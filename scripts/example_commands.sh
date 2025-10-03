#!/usr/bin/env bash

# Example one-shot run (baseline + VLM eval)

python -m skin_vqa.data_prep.make_splits --root data/ham10000 --seed 42
python -m skin_vqa.data_prep.make_vqa_pairs --root data/ham10000 --out data/vqa_pairs

python train_baseline.py \
  --data-root data/ham10000 \
  --splits data/ham10000/splits \
  --arch resnet50 \
  --epochs 1 --batch-size 8 \
  --outdir outputs/baseline_resnet50_quick

python vlm_eval_llava_med.py \
  --model microsoft/llava-med-v1.5-mistral-7b \
  --vqa-root data/vqa_pairs \
  --split test \
  --outdir outputs/vlm_pretrained_quick \
  --precision fp16
