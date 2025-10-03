#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate VLMs on HAM10000 framed as closed-ended VQA.

Backends:
  - HF path (--backend hf): Hugging Face Transformers (AutoProcessor + LlavaForConditionalGeneration)
    Uses apply_chat_template for prompt formatting. [HF docs]
  - LLaVA-Med path (--backend llavamed): Official microsoft/LLaVA-Med loader with
    'mistral_instruct' conversation template and mm_utils.

References:
- HF LLaVA docs: https://huggingface.co/docs/transformers/model_doc/llava
- LLaVA-Med repo: https://github.com/microsoft/LLaVA-Med
"""

import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

# Repo utilities (pure-Python; safe to import at module import time)
from skin_vqa.datasets.vqa_dataset import VQAPairs
from skin_vqa.utils.metrics import multiclass_metrics
from skin_vqa.utils.logging_utils import RunLogger
from skin_vqa.utils.plots import save_confusion_matrix, save_roc_curves, save_pr_curves
from skin_vqa.constants import LABELS, LABEL2ID

# NOTE: Do NOT import heavy HF / LLaVA-Med modules at top level.
# We import them lazily inside the selected backend to avoid pulling bitsandbytes unexpectedly.


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score_options_llavamed(
    tokenizer,
    model,
    image_processor,
    pil_image: Image.Image,
    question_text: str,
    options,
    precision: str = "fp16",
):
    """
    Compute class probabilities for closed-ended answers using LLaVA‑Med.

    Prompt format (Mistral-style):
      user: <image>\n{question}
      assistant: (to be generated)

    For each candidate answer, we mask the loss to the answer tokens only,
    turn the summed log-likelihood into a score, and softmax across options.
    """
    # Lazily import LLaVA-Med conversation/template utilities
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], "<image>\n" + question_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Prompt tokens (with <image> placeholder)
    prompt_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )

    # Image tensor
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
        if precision == "fp16"
        else torch.float32
    )
    images = process_images([pil_image], image_processor, model.config)
    if isinstance(images, (list, tuple)):
        images = [img.to(model.device, dtype=dtype) for img in images]
    else:
        images = images.to(model.device, dtype=dtype)

    # Score each option by (negative) masked loss on answer tokens
    scores = []
    for ans in options:
        ans_ids = tokenizer(ans, return_tensors="pt", add_special_tokens=False).input_ids.to(
            model.device
        )
        input_ids = torch.cat([prompt_ids, ans_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100  # mask prompt

        with torch.no_grad():
            out = model(input_ids=input_ids, images=images, labels=labels)

        n_tokens = max(1, ans_ids.shape[1])
        sum_logprob = -out.loss.item() * n_tokens
        scores.append(sum_logprob)

    probs = torch.tensor(scores).softmax(dim=0).tolist()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    ap.add_argument("--vqa-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--backend",
        choices=["hf", "llavamed"],
        default="hf",
        help="Use HF Auto* (hf) or the official LLaVA‑Med loader (llavamed).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    logger = RunLogger(args.outdir)
    logger.dump_config(vars(args))

    ds = VQAPairs(args.vqa_root, args.split)

    # -------------------------
    # Load model (two backends)
    # -------------------------
    if args.backend == "llavamed" or "microsoft/llava-med" in args.model.lower():
        # Import LLaVA‑Med only when needed, to avoid pulling in HF/bitsandbytes via side effects.
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
        model.eval()
        backend = "llavamed"
        processor = None  # not used in this branch

    else:
        # HF path (import only here to avoid bitsandbytes import when not needed)
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as e:
            # If this crashes with a bitsandbytes CUDA setup error, see notes below.
            raise RuntimeError(
                "Failed to import HF LLaVA classes. If the traceback mentions "
                "'bitsandbytes' and 'CUDA Setup failed', either uninstall bitsandbytes "
                "(pip uninstall bitsandbytes) or ensure CUDA libs are discoverable "
                "(see 'python -m bitsandbytes')."
            ) from e

        dtype = (
            torch.bfloat16
            if args.precision == "bf16"
            else torch.float16
            if args.precision == "fp16"
            else torch.float32
        )
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model.eval()
        backend = "hf"
        image_processor = None  # not used in this branch
        tokenizer = None

    # -------------------------
    # Evaluate
    # -------------------------
    y_true, y_prob = [], []

    if backend == "hf":
        # Lazy import our HF scoring util now that processor/model exist.
        from skin_vqa.vlm.scoring import score_options  # uses apply_chat_template (HF docs)

    for ex in tqdm(ds, desc=f"Evaluating {args.model} [{backend}]"):
        question = ex["question"]
        if backend == "hf":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            probs = score_options(model, processor, messages, LABELS, ex["image"], model.device)
        else:
            probs = score_options_llavamed(
                tokenizer, model, image_processor, ex["image"], question, LABELS, args.precision
            )
        y_prob.append(probs)
        y_true.append(LABEL2ID[ex["answer"]])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Metrics + logging
    metrics = multiclass_metrics(y_true, y_prob)
    logger.log_scalar(step=0, split="test", loss=0.0, metrics=metrics)

    save_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        os.path.join(args.outdir, "confusion_matrix.png"),
    )
    save_roc_curves(y_true, y_prob, os.path.join(args.outdir, "roc_curves.png"))
    save_pr_curves(y_true, y_prob, os.path.join(args.outdir, "pr_curves.png"))
    logger.dump_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
