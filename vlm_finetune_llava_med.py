#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune LLaVA(-Med) on HAM10000 framed as closed-ended VQA with LoRA.

Backends
--------
- HF (--backend hf): Hugging Face `LlavaForConditionalGeneration` + `AutoProcessor`.
  Uses `apply_chat_template` as recommended by HF's LLaVA docs.
- LLaVA-Med (--backend llavamed): Official loader (`load_pretrained_model`) with
  `mistral_instruct` conversation template and mm_utils. No AutoProcessor needed.

Key details
-----------
- Labels are masked to the *answer tokens only* so the model learns to emit one of
  the allowed class tokens.
- Validation uses closed-ended option scoring (sum of token log-likelihoods per option,
  softmax to probabilities) to compute Accuracy / macro-F1 / macro-AUC.
- **DataLoader uses a custom collate_fn that returns the list of examples as-is**,
  because default_collate cannot stack PIL images. We convert inside format_* funcs.

References
----------
- PyTorch DataLoader & collate_fn: https://pytorch.org/docs/stable/data.html
- LLaVA(-Med) usage & templates: https://github.com/microsoft/LLaVA-Med
"""

import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Project utilities
from skin_vqa.datasets.vqa_dataset import VQAPairs
from skin_vqa.constants import LABELS, LABEL2ID
from skin_vqa.utils.metrics import multiclass_metrics
from skin_vqa.utils.logging_utils import RunLogger
from skin_vqa.vlm.prompts import make_closed_ended_question


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 16
    weight_decay: float = 0.0
    max_new_tokens: int = 8


def get_warmup_linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    """
    Lightweight replacement for HF's `get_linear_schedule_with_warmup` to avoid importing
    transformers when using the LLaVA‑Med backend.
    """
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------
# HF backend: helpers
# ---------------------------
def format_batch_hf(processor, examples: List[Dict[str, Any]]):
    """
    Build batched inputs/labels for HF LLaVA using `apply_chat_template`.
    Labels mask prompt tokens (-100) and supervise only the answer tokens.
    """
    question = make_closed_ended_question() + " Respond with exactly one token."
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]

    images = [ex["image"] for ex in examples]
    prompts = [processor.apply_chat_template(messages, add_generation_prompt=True) for _ in examples]
    answers = [ex["answer"] for ex in examples]

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    ans_ids = [processor.tokenizer.encode(a, add_special_tokens=False) for a in answers]

    # Build labels: -100 for prompt tokens; answer tokens are targets
    labels = inputs["input_ids"].clone()
    labels[:] = -100
    for i in range(len(examples)):
        single = processor(text=prompts[i], images=images[i], return_tensors="pt")
        prompt_len = single["input_ids"].shape[1]
        ids = torch.tensor(ans_ids[i], dtype=labels.dtype)
        max_copy = min(ids.numel(), labels.shape[1] - prompt_len)
        labels[i, prompt_len:prompt_len + max_copy] = ids[:max_copy]

    return inputs, labels


# ---------------------------
# LLaVA‑Med backend: helpers
# ---------------------------
def pad_ids(id_lists: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1D LongTensors to a batch [B, L] + attention_mask [B, L].
    """
    max_len = max(ids.numel() for ids in id_lists)
    batch = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        L = ids.numel()
        batch[i, :L] = ids
        attn[i, :L] = 1
    return batch, attn


def get_projector_dtype(model):
    """
    Inspect the mm_projector dtype for dtype-safe image casting.
    """
    try:
        proj = model.get_model().mm_projector
    except AttributeError:
        proj = getattr(model, "mm_projector", None)
    if proj is not None:
        return next(p for p in proj.parameters()).dtype
    return next(model.parameters()).dtype


def format_batch_llavamed(tokenizer, image_processor, model, examples: List[Dict[str, Any]]):
    """
    Build batched `input_ids`, `attention_mask`, `images`, `labels` for LLaVA‑Med.
    Uses `mistral_instruct` conversation template and masks labels to answer tokens only.
    """
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    # 1) Compose prompts (one per example)
    question = make_closed_ended_question() + " Respond with exactly one token."
    prompts = []
    for _ in examples:
        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + question)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    # 2) Tokenize prompts -> list of 1D LongTensors
    prompt_ids_list: List[torch.Tensor] = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
        for p in prompts
    ]

    # 3) Answers
    answers = [ex["answer"] for ex in examples]
    answer_ids_list: List[torch.Tensor] = [
        tokenizer(a, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        for a in answers
    ]

    # 4) Concatenate prompt + answer per sample; build labels that mask the prompt
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids_cat = []
    labels_list = []
    for prompt_ids, ans_ids in zip(prompt_ids_list, answer_ids_list):
        cat = torch.cat([prompt_ids, ans_ids], dim=0)
        label = torch.full_like(cat, -100)
        label[-ans_ids.numel():] = ans_ids  # only answer tokens supervised
        input_ids_cat.append(cat)
        labels_list.append(label)

    # 5) Pad to batch
    input_ids, attention_mask = pad_ids(input_ids_cat, pad_id)
    labels, _ = pad_ids(labels_list, -100)  # second tensor unused

    # 6) Images -> projector dtype to avoid bf16/fp16 mismatch
    proj_dtype = get_projector_dtype(model)
    images = process_images([ex["image"] for ex in examples], image_processor, model.config)
    if isinstance(images, (list, tuple)):
        images = [img.to(model.device, dtype=proj_dtype) for img in images]
    else:
        images = images.to(model.device, dtype=proj_dtype)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "labels": labels,
    }
    return batch


def score_options_llavamed(tokenizer, model, image_processor, pil_image: Image.Image,
                           question_text: str, options: List[str], precision: str) -> List[float]:
    """
    Closed-ended option scoring for LLaVA‑Med (sum log-likelihood over answer tokens).
    """
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], "<image>\n" + question_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    # dtype-safe images
    proj_dtype = get_projector_dtype(model)
    images = process_images([pil_image], image_processor, model.config)
    if isinstance(images, (list, tuple)):
        images = [img.to(model.device, dtype=proj_dtype) for img in images]
    else:
        images = images.to(model.device, dtype=proj_dtype)

    scores = []
    for ans in options:
        ans_ids = tokenizer(ans, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        input_ids = torch.cat([prompt_ids, ans_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, images=images, labels=labels)
        n_tok = max(1, ans_ids.shape[1])
        scores.append(-out.loss.item() * n_tok)

    return torch.tensor(scores).softmax(dim=0).tolist()


def collate_identity(batch):
    """
    Return the batch as a Python list of examples (dicts).
    This avoids PyTorch's default_collate trying to stack PIL images into tensors.
    See DataLoader `collate_fn` docs.
    """
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='microsoft/llava-med-v1.5-mistral-7b')
    ap.add_argument('--backend', choices=['hf', 'llavamed'], default='hf',
                    help="Use HF Auto* (hf) or the official LLaVA‑Med loader (llavamed).")
    ap.add_argument('--vqa-root', type=str, required=True)
    ap.add_argument('--train-split', type=str, default='train')
    ap.add_argument('--val-split', type=str, default='val')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--precision', type=str, default='fp16', choices=['bf16','fp16','fp32'])
    ap.add_argument('--load-in-4bit', action='store_true',
                    help='HF backend only. Ignored for llavamed.')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    logger = RunLogger(args.outdir)
    logger.dump_config(vars(args))

    train_ds = VQAPairs(args.vqa_root, args.train_split)
    val_ds   = VQAPairs(args.vqa_root, args.val_split)

    # -------------------------
    # Load model (two backends; lazy imports to avoid BNB side-effects)
    # -------------------------
    backend = args.backend
    if backend == 'llavamed' or 'microsoft/llava-med' in args.model.lower():
        try:
            from llava.model.builder import load_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "Backend 'llavamed' requested but LLaVA‑Med is not importable. "
                "Install with: pip install -e git+https://github.com/microsoft/LLaVA-Med.git"
            ) from e

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=args.model,
            model_base=None,
            model_name="llava-med-v1.5-mistral-7b",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.config.use_cache = False
        processor = None  # not used in this branch

    else:
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as e:
            raise RuntimeError(
                "Failed to import HF LLaVA classes. If the traceback mentions "
                "'bitsandbytes' and 'CUDA Setup failed', either uninstall bitsandbytes "
                "or ensure CUDA libs are discoverable (`python -m bitsandbytes`)."
            ) from e

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        dtype = torch.bfloat16 if args.precision=='bf16' else (torch.float16 if args.precision=='fp16' else torch.float32)
        if args.load_in_4bit:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model, load_in_4bit=True, device_map='auto', trust_remote_code=True
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=dtype, device_map='auto', trust_remote_code=True
            )
        model.config.use_cache = False
        tokenizer = None
        image_processor = None

    # -------------------------
    # PEFT LoRA
    # -------------------------
    from peft import LoraConfig, get_peft_model, TaskType
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    )
    model = get_peft_model(model, lora)
    model.train()

    # -------------------------
    # Data loader with identity collate (critical fix for PIL images)
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_identity,  # <--- avoid default_collate on PIL Images
    )

    # -------------------------
    # Optimizer & schedule
    # -------------------------
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
    sched = get_warmup_linear_schedule(optim, warmup_steps=int(0.03 * total_steps), total_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.precision=='fp16' and torch.cuda.is_available()))
    global_step = 0
    best_macro_f1 = -1.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch_examples in enumerate(train_loader, start=1):
            if backend == 'hf':
                inputs, labels = format_batch_hf(processor, batch_examples)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels.to(model.device)

                with torch.amp.autocast('cuda', enabled=(args.precision=='fp16' and torch.cuda.is_available())):
                    out = model(**inputs, labels=labels)
                    loss = out.loss / args.grad_accum

            else:
                batch = format_batch_llavamed(tokenizer, image_processor, model, batch_examples)
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                images = batch["images"]  # already on device/dtype

                with torch.amp.autocast('cuda', enabled=(args.precision=='fp16' and torch.cuda.is_available())):
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                images=images, labels=labels)
                    loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

            running_loss += (out.loss.item())

        # -------------------------
        # Validation (closed-ended scoring)
        # -------------------------
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            if backend == 'hf':
                # Lazy import to avoid top-level side effects
                from skin_vqa.vlm.scoring import score_options
                for ex in val_ds:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": make_closed_ended_question()}]
                    }]
                    probs = score_options(model, processor, messages, LABELS, ex["image"], model.device)
                    y_prob.append(probs)
                    y_true.append(LABEL2ID[ex["answer"]])
            else:
                for ex in val_ds:
                    probs = score_options_llavamed(
                        tokenizer, model, image_processor, ex["image"],
                        make_closed_ended_question(), LABELS, args.precision
                    )
                    y_prob.append(probs)
                    y_true.append(LABEL2ID[ex["answer"]])

        y_true = np.array(y_true); y_prob = np.array(y_prob)
        metrics = multiclass_metrics(y_true, y_prob)
        logger.log_scalar(step=epoch, split='val',
                          loss=running_loss / max(1, len(train_loader)),
                          metrics=metrics)

        # Save best LoRA adapter
        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']
            ckpt_dir = os.path.join(args.outdir, 'checkpoints', 'adapter_model')
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)

    # Final dump
    logger.dump_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()



# import argparse, os, json
# from dataclasses import dataclass
# from typing import Dict, Any
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoProcessor, LlavaForConditionalGeneration, get_linear_schedule_with_warmup
# from peft import LoraConfig, get_peft_model, TaskType, PeftModel
# from tqdm import tqdm

# from skin_vqa.datasets.vqa_dataset import VQAPairs
# from skin_vqa.constants import LABELS, LABEL2ID
# from skin_vqa.utils.metrics import multiclass_metrics
# from skin_vqa.utils.logging_utils import RunLogger
# from skin_vqa.vlm.scoring import score_options
# from skin_vqa.vlm.prompts import make_closed_ended_question

# def set_seed(seed: int = 42):
#     import random, numpy as np, torch
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# @dataclass
# class TrainCfg:
#     lr: float = 2e-4
#     epochs: int = 1
#     batch_size: int = 1
#     grad_accum: int = 16
#     weight_decay: float = 0.0
#     max_new_tokens: int = 8

# def format_batch(processor, examples):
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": make_closed_ended_question() + " Respond with exactly one token."},
#         ]
#     }]
#     images = [ex['image'] for ex in examples]
#     prompts = [processor.apply_chat_template(messages, add_generation_prompt=True) for _ in examples]
#     answers = [ex['answer'] for ex in examples]

#     # Build tokenized inputs for prompt + answer
#     inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True)
#     # Tokenize answers separately (no special tokens)
#     ans_ids = [processor.tokenizer.encode(a, add_special_tokens=False) for a in answers]

#     # Build labels: -100 for prompt tokens, answer tokens as targets
#     labels = inputs['input_ids'].clone()
#     labels[:] = -100  # mask everything first
#     # because we used padding=True, compute prompt lengths separately per example
#     for i in range(len(examples)):
#         # Re-tokenize single prompt to get its length
#         single_prompt = processor(text=prompts[i], images=images[i], return_tensors='pt')
#         prompt_len = single_prompt['input_ids'].shape[1]
#         # place answer ids after prompt_len
#         ids = torch.tensor(ans_ids[i], dtype=labels.dtype)
#         # ensure fits in sequence
#         max_copy = min(ids.numel(), labels.shape[1] - prompt_len)
#         labels[i, prompt_len:prompt_len + max_copy] = ids[:max_copy]

#     return inputs, labels

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--model', type=str, default='microsoft/llava-med-v1.5-mistral-7b')
#     ap.add_argument('--vqa-root', type=str, required=True)
#     ap.add_argument('--train-split', type=str, default='train')
#     ap.add_argument('--val-split', type=str, default='val')
#     ap.add_argument('--epochs', type=int, default=1)
#     ap.add_argument('--batch-size', type=int, default=1)
#     ap.add_argument('--grad-accum', type=int, default=16)
#     ap.add_argument('--lr', type=float, default=2e-4)
#     ap.add_argument('--weight-decay', type=float, default=0.0)
#     ap.add_argument('--precision', type=str, default='bf16', choices=['bf16','fp16','fp32'])
#     ap.add_argument('--load-in-4bit', action='store_true')
#     ap.add_argument('--outdir', type=str, required=True)
#     ap.add_argument('--seed', type=int, default=42)
#     args = ap.parse_args()

#     set_seed(args.seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(args.outdir, exist_ok=True)
#     logger = RunLogger(args.outdir)
#     logger.dump_config(vars(args))

#     train_ds = VQAPairs(args.vqa_root, args.train_split)
#     val_ds   = VQAPairs(args.vqa_root, args.val_split)

#     processor = AutoProcessor.from_pretrained(args.model)

#     # Model loading with optional 4-bit
#     dtype = torch.bfloat16 if args.precision=='bf16' else (torch.float16 if args.precision=='fp16' else torch.float32)
#     if args.load_in_4bit:
#         model = LlavaForConditionalGeneration.from_pretrained(
#             args.model, load_in_4bit=True, device_map='auto'
#         )
#     else:
#         model = LlavaForConditionalGeneration.from_pretrained(
#             args.model, torch_dtype=dtype, device_map='auto'
#         )
#     model.config.use_cache = False

#     # LoRA config (typical for LLaMA/Mistral family)
#     lora = LoraConfig(
#         r=16, lora_alpha=32, lora_dropout=0.05,
#         bias='none',
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=[
#             'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'
#         ],
#     )
#     model = get_peft_model(model, lora)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)  # eval one by one for scoring

#     optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     total_steps = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
#     sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.03*total_steps), num_training_steps=total_steps)

#     scaler = torch.cuda.amp.GradScaler(enabled=(args.precision=='fp16'))
#     global_step = 0
#     best_macro_f1 = -1.0

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         running_loss = 0.0
#         for step, batch in enumerate(train_loader, start=1):
#             inputs, labels = format_batch(processor, batch)
#             inputs = {k: v.to(model.device) for k, v in inputs.items()}
#             labels = labels.to(model.device)

#             with torch.cuda.amp.autocast(enabled=(args.precision=='fp16')):
#                 out = model(**inputs, labels=labels)
#                 loss = out.loss / args.grad_accum

#             scaler.scale(loss).backward()

#             if step % args.grad_accum == 0:
#                 scaler.step(optim)
#                 scaler.update()
#                 optim.zero_grad(set_to_none=True)
#                 sched.step()
#                 global_step += 1
#             running_loss += out.loss.item()

#         # Validation via closed-ended scoring
#         model.eval()
#         y_true, y_prob = [], []
#         with torch.no_grad():
#             for ex in val_ds:
#                 messages = [{
#                     "role": "user",
#                     "content": [{"type": "image"}, {"type": "text", "text": make_closed_ended_question()}]
#                 }]
#                 probs = score_options(model, processor, messages, LABELS, ex['image'], model.device)
#                 y_prob.append(probs); y_true.append(LABEL2ID[ex['answer']])
#         import numpy as np
#         from skin_vqa.utils.metrics import multiclass_metrics
#         metrics = multiclass_metrics(np.array(y_true), np.array(y_prob))
#         logger.log_scalar(step=epoch, split='val', loss=running_loss/len(train_loader), metrics=metrics)

#         # Save best LoRA adapter
#         if metrics['macro_f1'] > best_macro_f1:
#             best_macro_f1 = metrics['macro_f1']
#             ckpt_dir = os.path.join(args.outdir, 'checkpoints', 'adapter_model')
#             os.makedirs(ckpt_dir, exist_ok=True)
#             model.save_pretrained(ckpt_dir)

#     # Final validation metrics dump
#     logger.dump_metrics(metrics)
#     print(json.dumps(metrics, indent=2))

# if __name__ == '__main__':
#     main()
