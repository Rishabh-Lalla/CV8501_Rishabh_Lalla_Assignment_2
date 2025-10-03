# CV8501 — Skin Lesion Classification as Medical VQA (PyTorch)

A complete, reproducible PyTorch codebase that fulfills the **CV8501 Assignment 2** requirements:
- Baseline CNN/ViT classifier (7 classes on HAM10000).  
- VLM approach using **LLaVA‑Med** (pre‑trained evaluation) framed as closed‑ended VQA.  
- **Fine‑tuning** LLaVA‑Med with LoRA adapters on the generated VQA pairs.  
- Unified, fair **splits**, consistent **metrics** (Accuracy, F1, AUC), and thorough **logging** (TensorBoard, JSON/CSV, saved weights).

> **Clinical disclaimer**: This repository is for research/education only and **must not** be used for clinical decision-making.

---

## 1) Quick start

### 1.1. Environment

```bash
# Python 3.10+ recommended
conda create -n llavamed310 python=3.10 -y
conda activate llavamed310
pip install --upgrade pip
pip install -e git+https://github.com/microsoft/LLaVA-Med.git


cd cv8501_skin_vqa_pytorch
export PYTHONPATH="$PWD/src:$PYTHONPATH"

```

> If you plan to fine‑tune LLaVA‑Med on a single GPU with limited memory, prefer the 4‑bit path and keep batch sizes small.

### 1.2. Dataset: HAM10000

Download the dataset from Kaggle (requires Kaggle CLI authentication):

```bash
bash scripts/download_ham10000.sh
```

This places images and metadata under `data/ham10000/`:
```
data/ham10000/
  ├── images/  # merged from part_1 + part_2
  └── HAM10000_metadata.csv
```

### 1.3. Reproducible splits & VQA pairs

```bash
# 70/10/20 train/val/test with group-aware split by lesion_id
python -m skin_vqa.data_prep.make_splits --root data/ham10000 --seed 42

# Convert to closed-ended VQA (question–answer pairs)
python -m skin_vqa.data_prep.make_vqa_pairs --root data/ham10000 --out data/vqa_pairs
```

### 1.4. Train the **baseline** (CNN/ViT)

```bash
# ResNet50 baseline (default)
python train_baseline.py   --data-root data/ham10000   --splits data/ham10000/splits   --arch resnet50   --epochs 25 --batch-size 32   --outdir outputs/baseline_resnet50

# ViT-B/16 (torchvision) baseline
python train_baseline.py   --data-root data/ham10000   --splits data/ham10000/splits   --arch vit_b_16   --epochs 25 --batch-size 32   --outdir outputs/baseline_vit_b16
```

### 1.5. Evaluate **pre‑trained LLaVA‑Med** (VQA)

```bash
python vlm_eval_llava_med.py   --backend llavamed   --model microsoft/llava-med-v1.5-mistral-7b   --vqa-root data/vqa_pairs --split test   --outdir outputs/vlm_pretrained_med --precision fp16
```

### 1.6. **Fine‑tune** LLaVA‑Med with LoRA



```bash
python vlm_finetune_llava_med.py   --backend llavamed   --model microsoft/llava-med-v1.5-mistral-7b   --vqa-root data/vqa_pairs   --train-split train --val-split val   --epochs 1 --batch-size 1 --grad-accum 16   --lr 2e-4 --precision fp16   --outdir outputs/vlm_finetune_lora_med
```

### 1.7. **Fine‑tune** LLaVA‑Med with LoRA



```bash
python analyze_checkpoints.py   --data-root /l/users/rishabh.lalla/CV8501_Ag2/cv8501_skin_vqa_pytorch/data/ham10000   --splits /l/users/rishabh.lalla/CV8501_Ag2/cv8501_skin_vqa_pytorch/data/ham10000/splits   --baseline-checkpoint /l/users/rishabh.lalla/CV8501_Ag2/cv8501_skin_vqa_pytorch/outputs/baseline_resnet50/checkpoints/best.pt   --baseline-arch resnet50   --vlm-pretrained-id microsoft/llava-med-v1.5-mistral-7b   --vlm-lora-path /l/users/rishabh.lalla/CV8501_Ag2/cv8501_skin_vqa_pytorch/outputs/vlm_finetune_lora_med_2epochs/checkpoints/adapter_model   --outdir analyses/skin_vqa   --batch-size 32 --num-workers 4 --device cuda   --eval-splits train,val,test   --vlm-precision fp16   --verbose 2   --baseline-strict
```


## 2) Notes & choices

- **Fair splits**: group‑aware by `lesion_id` to reduce leakage.  
- **Class imbalance**: we compute class weights for loss and report macro metrics.  
- **VLM prob. scoring**: for AUC/F1, we derive class probabilities by summing token log‑likelihoods of each allowed option (normalized with softmax).  
- **LoRA fine‑tuning**: adapters on Mistral/LLaMA attention/MLP modules; optionally train `mm_projector`.  


---

## 3) Reproducibility

- All hyper‑params are CLI‑controlled and recorded to `outdir/config.json`.
- Seeds are set in data split + trainers; PyTorch deterministic flags are toggled.
- Since we provide the model checkpoints, you can just run steps 1.1 , 1.2, 1.3 and 1.7

---


