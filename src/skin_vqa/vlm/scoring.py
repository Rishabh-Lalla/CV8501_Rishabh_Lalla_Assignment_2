import torch
import torch.nn.functional as F
from typing import List, Dict
from PIL import Image

# Utilities to compute option probabilities by summing log-likelihoods of answer tokens

def ll_token_logprobs(model, processor, prompt_text: str, image: Image.Image, answer_ids: List[int], device):
    """Return summed log-prob of a candidate answer sequence given prompt + image."""
    with torch.no_grad():
        # tokenize prompt alone to get its length
        prompt_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        # Now tokenize prompt + answer to get full sequence
        full_inputs = processor(text=prompt_text + " " + processor.tokenizer.decode(answer_ids), images=image, return_tensors="pt")
        full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

        # Forward to get logits for next-token predictions over full sequence
        # We need logits to score the *answer* part only
        out = model(**full_inputs)
        logits = out.logits  # [1, seq_len, vocab]
        # positions corresponding to the answer
        seq_len = full_inputs["input_ids"].shape[1]
        prompt_len = prompt_inputs["input_ids"].shape[1]
        ans_len = seq_len - prompt_len
        # predicted logits for each answer token (shifted)
        # For token t at position p, the model predicts token at p given up to p-1
        # So we align logits[-ans_len-1 : -1] with answer_ids
        start = seq_len - ans_len - 1
        end = seq_len - 1
        step_logits = logits[:, start:end, :]  # [1, ans_len, vocab]
        target = torch.tensor(answer_ids, device=device).unsqueeze(0)  # [1, ans_len]
        log_probs = F.log_softmax(step_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, ans_len]
        return log_probs.sum().item()

def score_options(model, processor, messages: List[Dict], options: List[str], image: Image.Image, device) -> List[float]:
    """Return normalized probabilities over options for a single (image, messages)."""
    # Build prompt with chat template
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Tokenize each option to ids (without BOS/EOS)
    option_ids = [processor.tokenizer.encode(opt, add_special_tokens=False) for opt in options]
    scores = []
    for ids in option_ids:
        s = ll_token_logprobs(model, processor, prompt_text, image, ids, device)
        scores.append(s)
    # Normalize
    probs = torch.tensor(scores).softmax(dim=0).tolist()
    return probs
