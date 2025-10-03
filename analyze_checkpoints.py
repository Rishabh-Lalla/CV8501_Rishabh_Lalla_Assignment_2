#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze baseline CNN, LLaVA‑Med (pretrained), and LLaVA‑Med + LoRA on HAM10000 with
comprehensive classification analysis.

What’s included
---------------
- Progress printing for each sample (or at intervals) + tqdm progress bars.
- Robust baseline checkpoint loading: EMA/state_dict preference, prefix cleanup,
  missing/unexpected key reporting, optional strict load, head weight stats.
- LLaVA‑Med evaluation via the official backend (mistral_instruct template), fp16 by default,
  batch size forced to 1 (pretrained & LoRA).
- Metrics/plots: classwise PRF report, confusion matrices, ROC/PR (OvR), macro‑AUC (OvR),
  calibration curves (reliability) + ECE, pairwise McNemar tests, bootstrap CIs.
- Class distribution plot for train/val/test.

References
----------
- PyTorch load_state_dict strict & key diagnostics.  # https://docs.pytorch.org/ (Module.load_state_dict)
- tqdm progress bars.                                # https://tqdm.github.io/
- scikit-learn metrics, calibration, reports.        # https://scikit-learn.org/
- LLaVA‑Med official repository & templates.         # https://github.com/microsoft/LLaVA-Med

Assignment alignment
--------------------
- Same split across methods; metrics include Accuracy, F1, AUC; analysis artifacts produced
  for error inspection—as required in your assignment brief (pp. 1–2)."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from tqdm import tqdm  # progress bars

# Optional stats packages
try:
    from mlxtend.evaluate import mcnemar as mcnemar_test
    HAVE_MCNEMAR = True
except Exception:
    HAVE_MCNEMAR = False

try:
    from scipy.stats import bootstrap as scipy_bootstrap
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -------------------------
# General utils
# -------------------------

HAM10000_CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_label(y: str) -> str:
    y = str(y).strip().lower()
    mapping = {
        "akiec":"akiec", "bcc":"bcc", "bkl":"bkl", "df":"df", "mel":"mel", "nv":"nv", "vasc":"vasc",
        "dermatofibroma":"df", "melanoma":"mel", "nevus":"nv", "vascular":"vasc",
        "basal cell carcinoma":"bcc", "actinic keratoses":"akiec", "benign keratosis":"bkl",
    }
    return mapping.get(y, y)

def try_make_image_path(data_root: Path, image_id: str) -> str:
    """
    Given an image_id like 'ISIC_0027419', try common HAM10000 layouts.
    Return a RELATIVE path (to be resolved later against --data-root).
    """
    candidates = [
        f"images/{image_id}.jpg",
        f"images/{image_id}.png",
        f"HAM10000_images_part_1/{image_id}.jpg",
        f"HAM10000_images_part_2/{image_id}.jpg",
        f"{image_id}.jpg",
    ]
    for rel in candidates:
        if (data_root / rel).exists():
            return rel
    return candidates[0]

def unify_split_df(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """
    Convert a split dataframe to include:
      - label      (from label/class/target/y/dx/diagnosis/answer)
      - image_path (from image_path/path/filepath/... or derived from image_id)
    """
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}

    # label
    label_candidates = ["label","class","target","y","dx","diagnosis","answer"]
    label_col = next((cols[c] for c in label_candidates if c in cols), None)
    if label_col is None:
        raise ValueError(f"Could not find a label column among {label_candidates}. Found: {list(d.columns)}")
    d["label"] = d[label_col].astype(str).str.lower().map(normalize_label)

    # image_path
    path_candidates = ["image_path","img_path","path","filepath","file_path","image","img","filename","fname","relpath","rel_path"]
    path_col = next((cols[c] for c in path_candidates if c in cols), None)
    if path_col is not None:
        d["image_path"] = d[path_col].astype(str)
    else:
        if "image_id" not in cols:
            raise ValueError("Neither image_path nor image_id found in splits.")
        img_id_col = cols["image_id"]
        d["image_path"] = d[img_id_col].astype(str).apply(lambda s: try_make_image_path(data_root, s))

    return d[["image_path","label"] + [c for c in d.columns if c not in {"image_path","label"}]]

def load_splits(splits_path: Path, data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Accept either:
      - a directory containing train.csv, val.csv (or valid.csv), test.csv
      - a single file: splits.csv with a 'split' column in {train,val,test}
    Automatically derives 'label' and 'image_path' as needed.
    """
    if splits_path.is_dir():
        train_df = pd.read_csv(splits_path / "train.csv")
        val_df = pd.read_csv(splits_path / "val.csv") if (splits_path / "val.csv").exists() else pd.read_csv(splits_path / "valid.csv")
        test_df = pd.read_csv(splits_path / "test.csv")
    else:
        df = pd.read_csv(splits_path)
        if "split" not in df.columns:
            raise ValueError("splits.csv must include a 'split' column with values in {train,val,test}")
        train_df = df[df["split"]=="train"].copy()
        val_df   = df[df["split"]=="val"].copy()
        test_df  = df[df["split"]=="test"].copy()

    train_df = unify_split_df(train_df, data_root)
    val_df   = unify_split_df(val_df,   data_root)
    test_df  = unify_split_df(test_df,  data_root)

    for d in (train_df, val_df, test_df):
        d["label"] = d["label"].astype(str).str.lower().map(normalize_label)

    return train_df, val_df, test_df

def maybe_override_classes_from_file(path: str, classes: List[str]) -> List[str]:
    """Optionally force class order from a .txt (one per line) or .json ([...]/{'classes':[...]}) file."""
    if not path:
        return classes
    p = Path(path)
    if not p.exists():
        print(f"[WARN] --classes-file not found: {path}; using default order {classes}")
        return classes
    try:
        if p.suffix.lower() == ".json":
            data = json.load(open(p))
            lst = data["classes"] if isinstance(data, dict) and "classes" in data else data
        else:
            lst = [line.strip() for line in open(p) if line.strip()]
        lst = [normalize_label(x) for x in lst]
        print(f"[INFO] Using class order from {path}: {lst}")
        return lst
    except Exception as e:
        print(f"[WARN] Failed to parse --classes-file {path}: {e}. Using default order.")
        return classes


# -------------------------
# Datasets
# -------------------------

class SimpleImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, classes: List[str], image_size: int = 224):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.image_size = image_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(self.data_root) / row["image_path"]
        y = self.class_to_idx[normalize_label(row["label"])]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32)/255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        arr = (arr-mean)/std
        arr = np.transpose(arr, (2,0,1))
        return {"image": torch.from_numpy(arr), "label": int(y), "path": str(path)}


# -------------------------
# Baseline (torchvision)
# -------------------------

def build_torchvision_model(arch: str,
                            num_classes: int,
                            *,
                            state_keys: Optional[List[str]] = None,
                            drop_rate: float = 0.2) -> nn.Module:
    """
    Build a torchvision model with a classifier head that matches the checkpoint.
    If `state_keys` indicate a Sequential head (e.g., 'fc.1.weight'), we create
    nn.Sequential(Dropout, Linear); otherwise a single Linear.

    This mirrors your training code:

        m.fc = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
        # ... or for ViT:
        m.heads.head = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))

    """
    import torchvision.models as models

    arch_l = arch.lower()
    want_seq_head = False
    if state_keys is not None:
        # Detect head variant from checkpoint keys
        if any(k.startswith("fc.1.") for k in state_keys):
            want_seq_head = True
        if any(k.startswith("heads.head.1.") for k in state_keys):
            want_seq_head = True
        if any(k == "fc.weight" for k in state_keys) or any(k == "heads.head.weight" for k in state_keys):
            # explicit linear keys present → override to linear
            want_seq_head = False

    if arch_l in {"resnet50", "resnet-50"}:
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        if want_seq_head:
            m.fc = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
        else:
            m.fc = nn.Linear(in_features, num_classes)
        return m

    elif arch_l in {"resnet18", "resnet-18"}:
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        if want_seq_head:
            m.fc = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
        else:
            m.fc = nn.Linear(in_features, num_classes)
        return m

    elif arch_l in {"vit_b_16", "vit-b-16", "vit"}:
        m = models.vit_b_16(weights=None)
        in_features = m.heads.head.in_features
        if want_seq_head:
            m.heads.head = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
        else:
            m.heads.head = nn.Linear(in_features, num_classes)
        return m

    else:
        raise ValueError(f"Unsupported baseline arch: {arch}")


@dataclass
class BaselineConfig:
    arch: str = "resnet50"
    num_classes: int = 7
    image_size: int = 224
    drop_rate: float = 0.2  # default matches your training code

class BaselineEvaluator:
    def __init__(self, checkpoint_path: Path, device: str,
                 cfg: Optional[BaselineConfig] = None,
                 strict: bool = False,
                 expected_num_classes: Optional[int] = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Prefer EMA state dicts if present (common training practice).
        state = None
        if isinstance(ckpt, dict):
            for key in ["ema_state_dict", "model_ema", "state_dict", "model_state", "model"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]; break
        if state is None:
            if isinstance(ckpt, dict):
                state = ckpt
            else:
                raise ValueError("Unrecognized baseline checkpoint format.")

        # Read metadata if available
        arch = (ckpt.get("arch", None) if isinstance(ckpt, dict) else None) or "resnet50"
        num_classes = (ckpt.get("num_classes", None) if isinstance(ckpt, dict) else None) or 7
        image_size = (ckpt.get("image_size", None) if isinstance(ckpt, dict) else None) or 224
        drop_rate = (ckpt.get("drop_rate", None) if isinstance(ckpt, dict) else None) or (cfg.drop_rate if cfg else 0.2)
        class_to_idx = ckpt.get("class_to_idx", None) if isinstance(ckpt, dict) else None
        if class_to_idx:
            ordered = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
            print(f"[INFO] Checkpoint class order: {ordered}")

        # Apply CLI overrides
        if cfg is not None:
            arch = cfg.arch or arch
            num_classes = cfg.num_classes or num_classes
            image_size = cfg.image_size or image_size

        if expected_num_classes is not None and expected_num_classes != num_classes:
            print(f"[WARN] Checkpoint num_classes={num_classes} != expected={expected_num_classes}")

        # Build model with a head that matches the checkpoint (Sequential vs Linear)
        self.image_size = image_size
        state_keys = list(state.keys())
        self.model = build_torchvision_model(arch, num_classes,
                                             state_keys=state_keys,
                                             drop_rate=drop_rate)

        # Normalize prefixes (DataParallel, Lightning, etc.)
        norm_state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}

        # Load with diagnostics
        missing, unexpected = self.model.load_state_dict(norm_state, strict=strict)
        print(f"[BASELINE] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys (strict={strict})")
        if missing:    print("  missing (first 20):", missing[:20])
        if unexpected: print("  unexpected (first 20):", unexpected[:20])
        if strict and (missing or unexpected):
            raise RuntimeError("Strict load failed: see missing/unexpected keys above.")

        # Head stats to confirm non-random weights got loaded
        head = getattr(self.model, "fc", None) or getattr(self.model, "classifier", None) or getattr(self.model, "heads", None)
        if hasattr(head, "weight"):            # Linear head
            w = head.weight.detach().float()
            print(f"[BASELINE] head weight: shape={tuple(w.shape)} mean={w.mean():.4f} std={w.std():.4f}")
        elif isinstance(head, nn.Sequential) and hasattr(head[-1], "weight"):  # Sequential(Dropout, Linear)
            w = head[-1].weight.detach().float()
            print(f"[BASELINE] head[-1] weight: shape={tuple(w.shape)} mean={w.mean():.4f} std={w.std():.4f}")

        self.model.eval().to(device)
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def predict(self, loader: DataLoader, verbose: int = 1, log_every: int = 25, tag: str = ""):
        y_true, y_pred, paths = [], [], []
        probs_all = []
        N = len(loader.dataset)
        seen = 0
        pbar = tqdm(total=N, desc=f"Baseline{(':'+tag) if tag else ''}", disable=(verbose == 0))
        for batch in loader:
            x = batch["image"].to(self.device, non_blocking=True)
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            y_true.extend(batch["label"].tolist())
            y_pred.extend(pred.cpu().tolist())
            probs_all.append(prob.cpu().numpy())
            paths.extend(batch["path"])

            for p in batch["path"]:
                seen += 1
                if verbose >= 2 or (verbose == 1 and (seen % log_every == 0 or seen == 1 or seen == N)):
                    print(f"[baseline] {seen}/{N} {p}")
            pbar.update(len(batch["path"]))
        pbar.close()
        proba = np.concatenate(probs_all, axis=0)
        return np.array(y_true), np.array(y_pred), proba, paths



# -------------------------
# LLaVA‑Med (official backend, closed‑ended scoring)
# -------------------------

VQA_PROMPT = "Which diagnosis best describes this lesion? Choose one of {classes}. Respond with exactly one token."

def build_closed_ended_question(classes: List[str]) -> str:
    return VQA_PROMPT.format(classes=", ".join(classes))

def get_mm_projector_dtype(model):
    try:
        proj = model.get_model().mm_projector
    except AttributeError:
        proj = getattr(model, "mm_projector", None)
    if proj is not None:
        return next(p for p in proj.parameters()).dtype
    return next(model.parameters()).dtype

class VLMEvaluatorLLavaMed:
    def __init__(self, base_model_id: str, device: str, precision: str = "fp16",
                 lora_adapter_path: Optional[str] = None):
        try:
            from llava.model.builder import load_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "LLaVA‑Med is not importable. Install:\n"
                "  pip install -e git+https://github.com/microsoft/LLaVA-Med.git"
            ) from e

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=base_model_id,
            model_base=None,
            model_name="llava-med-v1.5-mistral-7b",
            device=device if device.startswith("cuda") else "cpu",
        )
        self.model.eval()
        self.device = device
        self.precision = precision.lower()

        if lora_adapter_path:
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("peft not found; install with `pip install peft`.") from e
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model.eval()

    @torch.no_grad()
    def score_options(self, pil_image: Image.Image, question_text: str, options: List[str]) -> List[float]:
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

        proj_dtype = get_mm_projector_dtype(self.model)
        images = process_images([pil_image], self.image_processor, self.model.config)
        if isinstance(images, (list, tuple)):
            images = [img.to(self.model.device, dtype=proj_dtype) for img in images]
        else:
            images = images.to(self.model.device, dtype=proj_dtype)

        scores = []
        for ans in options:
            ans_ids = self.tokenizer(ans, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            input_ids = torch.cat([prompt_ids, ans_ids], dim=1)
            labels = input_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100  # supervise only answer tokens
            use_amp = (self.precision == "fp16" and self.device.startswith("cuda"))
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = self.model(input_ids=input_ids, images=images, labels=labels)
            n_tok = max(1, ans_ids.shape[1])
            scores.append(-out.loss.item() * n_tok)

        probs = torch.tensor(scores).softmax(dim=0).tolist()
        return probs

    @torch.no_grad()
    def predict_split(self, df: pd.DataFrame, data_root: Path, classes: List[str],
                      image_size: int = 224, verbose: int = 1, log_every: int = 25,
                      tag: str = ""):
        y_true, y_pred, y_proba, paths = [], [], [], []
        q = build_closed_ended_question(classes)
        N = len(df)
        pbar = tqdm(total=N, desc=f"LLaVA‑Med{(':'+tag) if tag else ''}", disable=(verbose == 0))
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            path = Path(data_root) / row["image_path"]
            img = Image.open(path).convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
            probs = self.score_options(img, q, classes)

            y_true.append(classes.index(normalize_label(row["label"])))
            y_pred.append(int(np.argmax(probs)))
            y_proba.append(np.array(probs)[None, :])
            paths.append(str(path))

            if verbose >= 2 or (verbose == 1 and (i % log_every == 0 or i == 1 or i == N)):
                print(f"[vlm] {i}/{N} {path}")
            pbar.update(1)
        pbar.close()
        return np.array(y_true), np.array(y_pred), np.concatenate(y_proba, axis=0), paths


# -------------------------
# Metrics & plots
# -------------------------

def compute_ece(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = prob.max(axis=1)
    predictions = prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; N = len(y_true)
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i+1])
        if not np.any(m): continue
        ece += (m.sum()/N) * abs(accuracies[m].mean() - confidences[m].mean())
    return float(ece)

def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")

def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray], classes: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    out = {"accuracy": float(acc), "report": report}
    if proba is not None:
        out["macro_auc_ovr"] = macro_ovr_auc(y_true, proba, len(classes))
        out["ece"] = compute_ece(proba, y_true)
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    return out

def plot_confusion(y_true, y_pred, classes, outpath: Path, normalize: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), normalize=normalize)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"; thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title("Confusion Matrix" + (f" (normalized={normalize})" if normalize else ""))
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_roc_ovr(y_true: np.ndarray, proba: np.ndarray, classes: List[str], outpath: Path):
    Y = label_binarize(y_true, classes=list(range(len(classes))))
    fig = plt.figure(figsize=(7,6))
    for i, c in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
            plt.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr,tpr):.3f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC (OvR)"); plt.legend(); fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_pr_ovr(y_true: np.ndarray, proba: np.ndarray, classes: List[str], outpath: Path):
    Y = label_binarize(y_true, classes=list(range(len(classes))))
    fig = plt.figure(figsize=(7,6))
    for i, c in enumerate(classes):
        try:
            precision, recall, _ = precision_recall_curve(Y[:, i], proba[:, i])
            ap = average_precision_score(Y[:, i], proba[:, i])
            plt.plot(recall, precision, label=f"{c} (AP={ap:.3f})")
        except Exception:
            continue
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (OvR)")
    plt.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_calibration_ovr(y_true: np.ndarray, proba: np.ndarray, classes: List[str], outpath: Path, n_bins: int = 10):
    Y = label_binarize(y_true, classes=list(range(len(classes))))
    fig = plt.figure(figsize=(7,6))
    for i, c in enumerate(classes):
        try:
            frac_pos, mean_pred = calibration_curve(Y[:, i], proba[:, i], n_bins=n_bins)
            plt.plot(mean_pred, frac_pos, marker="o", label=c)
        except Exception:
            continue
    plt.plot([0,1],[0,1],"--", linewidth=1)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration (Reliability) - OvR"); plt.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            classes: List[str], outpath: Path):
    train_counts = train_df["label"].astype(str).str.lower().value_counts()
    val_counts   = val_df["label"].astype(str).str.lower().value_counts()
    test_counts  = test_df["label"].astype(str).str.lower().value_counts()
    counts = pd.DataFrame({
        "class": classes,
        "train": [int(train_counts.get(c, 0)) for c in classes],
        "val":   [int(val_counts.get(c, 0)) for c in classes],
        "test":  [int(test_counts.get(c, 0)) for c in classes],
    })
    fig = plt.figure(figsize=(9,5))
    x = np.arange(len(classes)); w = 0.25
    plt.bar(x - w, counts["train"].values, width=w, label="train")
    plt.bar(x,     counts["val"].values,   width=w, label="val")
    plt.bar(x + w, counts["test"].values,  width=w, label="test")
    plt.xticks(x, classes); plt.ylabel("Count"); plt.title("Class distribution by split")
    plt.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)
    return counts

def mcnemar_compare(y_pred_a: np.ndarray, y_pred_b: np.ndarray, y_true: np.ndarray) -> Dict:
    if not HAVE_MCNEMAR:
        return {"available": False, "note": "mlxtend not installed"}
    both_correct = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    a_correct_b_wrong = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    a_wrong_b_correct = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    both_wrong = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
    table = np.array([[both_correct, a_correct_b_wrong],
                      [a_wrong_b_correct, both_wrong]], dtype=int)
    chi2, p = mcnemar_test(table, exact=False, corrected=True)
    return {"available": True, "chi2": float(chi2 if chi2 is not None else np.nan),
            "p_value": float(p), "table": table.tolist()}

def bootstrap_ci_accuracy_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    if not HAVE_SCIPY:
        return {"available": False}
    rng = np.random.default_rng(0)
    n = len(y_true); B = 2000
    acc_vals, f1_vals = [], []
    from sklearn.metrics import f1_score
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; yp = y_pred[idx]
        acc_vals.append((yt == yp).mean())
        f1_vals.append(f1_score(yt, yp, average="macro", zero_division=0))
    return {
        "available": True,
        "acc_ci_95": [float(np.quantile(acc_vals, 0.025)), float(np.quantile(acc_vals, 0.975))],
        "f1_macro_ci_95": [float(np.quantile(f1_vals, 0.025)), float(np.quantile(f1_vals, 0.975))]
    }


# -------------------------
# CLI / main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Analyze baseline & LLaVA‑Med (pretrained/finetuned) with rich classification analysis.")
    ap.add_argument("--data-root", type=str, required=True, help="Root folder to resolve relative image_path in CSVs.")
    ap.add_argument("--splits", type=str, required=True, help="Dir with train/val/test CSVs or a single splits.csv.")
    ap.add_argument("--outdir", type=str, required=True, help="Where to save outputs.")
    ap.add_argument("--classes", type=str, nargs="+", default=HAM10000_CLASSES)
    ap.add_argument("--classes-file", type=str, default="", help="Optional .txt/.json listing training-time class order.")
    ap.add_argument("--eval-splits", type=str, default="test", help="Comma list: 'test' or 'train,val,test'.")

    # Baseline
    ap.add_argument("--baseline-checkpoint", type=str, required=True)
    ap.add_argument("--baseline-arch", type=str, default="resnet50")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--baseline-strict", action="store_true", help="Fail on missing/unexpected keys when loading baseline.")

    # VLM (LLaVA‑Med)
    ap.add_argument("--vlm-pretrained-id", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    ap.add_argument("--vlm-lora-path", type=str, default="", help="Path to LoRA adapter dir (e.g., outputs/.../checkpoints/adapter_model)")
    ap.add_argument("--vlm-precision", type=str, choices=["fp16","bf16","fp32"], default="fp16", help="Precision for LLaVA‑Med forward.")
    ap.add_argument("--vlm-image-size", type=int, default=224)
    ap.add_argument("--vlm-batch-size", type=int, default=1, help="Forced to 1 for LLaVA‑Med.")

    # Verbosity
    ap.add_argument("--verbose", type=int, default=1, help="0: quiet | 1: tqdm + periodic prints | 2: print every sample path")
    ap.add_argument("--log-every", type=int, default=25, help="Print every N samples when --verbose=1")

    # Runtime
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.vlm_batch_size != 1:
        print("[INFO] Overriding --vlm-batch-size to 1 for LLaVA‑Med.")
        args.vlm_batch_size = 1
    outdir = Path(args.outdir); ensure_dir(outdir)

    # Load splits + plot distribution
    train_df, val_df, test_df = load_splits(Path(args.splits), Path(args.data_root))
    classes = maybe_override_classes_from_file(args.classes_file,
              [normalize_label(c) for c in args.classes])
    dist_df = plot_class_distribution(train_df, val_df, test_df, classes, outdir / "class_distribution.png")
    dist_df.to_csv(outdir / "class_distribution.csv", index=False)

    # Baseline evaluator + loader factory
    device = args.device
    baseline = BaselineEvaluator(
        Path(args.baseline_checkpoint),
        device=device,
        cfg=BaselineConfig(arch=args.baseline_arch, num_classes=len(classes), image_size=args.image_size),
        strict=args.baseline_strict,
        expected_num_classes=len(classes),
    )
    def make_loader(df):
        ds = SimpleImageDataset(df, Path(args.data_root), classes, image_size=baseline.image_size)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=("cuda" in device))

    # LLaVA‑Med evaluators (pretrained, finetuned/LoRA)
    vlm_pre = VLMEvaluatorLLavaMed(
        base_model_id=args.vlm_pretrained_id,
        device=device,
        precision=args.vlm_precision,
        lora_adapter_path=None,
    )
    vlm_ft = None
    if args.vlm_lora_path:
        vlm_ft = VLMEvaluatorLLavaMed(
            base_model_id=args.vlm_pretrained_id,
            device=device,
            precision=args.vlm_precision,
            lora_adapter_path=args.vlm_lora_path,
        )

    split_map = {"train": train_df, "val": val_df, "test": test_df}
    run_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip() in split_map]

    # Cache for pairwise tests
    test_preds_cache = {}

    for split in run_splits:
        df = split_map[split]
        split_dir = outdir / split; ensure_dir(split_dir)
        print(f"\n=== Evaluating split: {split} ({len(df)} samples) ===")

        # --- Baseline ---
        loader = make_loader(df)
        y_true_b, y_pred_b, proba_b, paths_b = baseline.predict(loader, verbose=args.verbose, log_every=args.log_every, tag=split)
        preds_b = pd.DataFrame({"image_path": paths_b,
                                "true": [classes[i] for i in y_true_b],
                                "pred": [classes[i] for i in y_pred_b],
                                "confidence": proba_b.max(axis=1)})
        for i,c in enumerate(classes): preds_b[f"p_{c}"] = proba_b[:, i]
        preds_b.to_csv(split_dir / "preds_baseline.csv", index=False)
        metrics_b = all_metrics(y_true_b, y_pred_b, proba_b, classes)
        (split_dir / "metrics_baseline.json").write_text(json.dumps(metrics_b, indent=2))
        plot_confusion(y_true_b, y_pred_b, classes, split_dir / "cm_baseline.png", normalize=None)
        plot_confusion(y_true_b, y_pred_b, classes, split_dir / "cm_baseline_norm_true.png", normalize="true")
        plot_roc_ovr(y_true_b, proba_b, classes, split_dir / "roc_baseline.png")
        plot_pr_ovr(y_true_b, proba_b, classes, split_dir / "pr_baseline.png")
        plot_calibration_ovr(y_true_b, proba_b, classes, split_dir / "calibration_baseline.png")

        # --- VLM pretrained ---
        y_true_p, y_pred_p, proba_p, paths_p = vlm_pre.predict_split(
            df, Path(args.data_root), classes, image_size=args.vlm_image_size,
            verbose=args.verbose, log_every=args.log_every, tag=f"pretrained-{split}"
        )
        preds_p = pd.DataFrame({"image_path": paths_p,
                                "true": [classes[i] for i in y_true_p],
                                "pred": [classes[i] for i in y_pred_p],
                                "confidence": proba_p.max(axis=1)})
        for i,c in enumerate(classes): preds_p[f"p_{c}"] = proba_p[:, i]
        preds_p.to_csv(split_dir / "preds_vlm_pretrained.csv", index=False)
        metrics_p = all_metrics(y_true_p, y_pred_p, proba_p, classes)
        (split_dir / "metrics_vlm_pretrained.json").write_text(json.dumps(metrics_p, indent=2))
        plot_confusion(y_true_p, y_pred_p, classes, split_dir / "cm_vlm_pretrained.png", normalize=None)
        plot_confusion(y_true_p, y_pred_p, classes, split_dir / "cm_vlm_pretrained_norm_true.png", normalize="true")
        plot_roc_ovr(y_true_p, proba_p, classes, split_dir / "roc_vlm_pretrained.png")
        plot_pr_ovr(y_true_p, proba_p, classes, split_dir / "pr_vlm_pretrained.png")
        plot_calibration_ovr(y_true_p, proba_p, classes, split_dir / "calibration_vlm_pretrained.png")

        # --- VLM finetuned (LoRA) ---
        if vlm_ft is not None:
            y_true_f, y_pred_f, proba_f, paths_f = vlm_ft.predict_split(
                df, Path(args.data_root), classes, image_size=args.vlm_image_size,
                verbose=args.verbose, log_every=args.log_every, tag=f"finetuned-{split}"
            )
            preds_f = pd.DataFrame({"image_path": paths_f,
                                    "true": [classes[i] for i in y_true_f],
                                    "pred": [classes[i] for i in y_pred_f],
                                    "confidence": proba_f.max(axis=1)})
            for i,c in enumerate(classes): preds_f[f"p_{c}"] = proba_f[:, i]
            preds_f.to_csv(split_dir / "preds_vlm_finetuned.csv", index=False)
            metrics_f = all_metrics(y_true_f, y_pred_f, proba_f, classes)
            (split_dir / "metrics_vlm_finetuned.json").write_text(json.dumps(metrics_f, indent=2))
            plot_confusion(y_true_f, y_pred_f, classes, split_dir / "cm_vlm_finetuned.png", normalize=None)
            plot_confusion(y_true_f, y_pred_f, classes, split_dir / "cm_vlm_finetuned_norm_true.png", normalize="true")
            plot_roc_ovr(y_true_f, proba_f, classes, split_dir / "roc_vlm_finetuned.png")
            plot_pr_ovr(y_true_f, proba_f, classes, split_dir / "pr_vlm_finetuned.png")
            plot_calibration_ovr(y_true_f, proba_f, classes, split_dir / "calibration_vlm_finetuned.png")

        # cache for test comparisons
        if split == "test":
            test_preds_cache["baseline"] = (y_true_b, y_pred_b)
            test_preds_cache["vlm_pretrained"] = (y_true_p, y_pred_p)
            if vlm_ft is not None:
                test_preds_cache["vlm_finetuned"] = (y_true_f, y_pred_f)

        # CIs
        for tag, (yt, yp) in {
            "baseline": (y_true_b, y_pred_b),
            "vlm_pretrained": (y_true_p, y_pred_p),
            **({"vlm_finetuned": (y_true_f, y_pred_f)} if ('vlm_ft' in locals() and vlm_ft is not None) else {})
        }.items():
            ci = bootstrap_ci_accuracy_f1(yt, yp)
            (split_dir / f"ci_{tag}.json").write_text(json.dumps(ci, indent=2))

    # McNemar on test split
    pairwise = {}
    if "baseline" in test_preds_cache and "vlm_pretrained" in test_preds_cache:
        yt, yb = test_preds_cache["baseline"]; yp = test_preds_cache["vlm_pretrained"][1]
        pairwise["baseline_vs_vlm_pretrained"] = mcnemar_compare(yb, yp, yt)
    if "baseline" in test_preds_cache and "vlm_finetuned" in test_preds_cache:
        yt, yb = test_preds_cache["baseline"]; yf = test_preds_cache["vlm_finetuned"][1]
        pairwise["baseline_vs_vlm_finetuned"] = mcnemar_compare(yb, yf, yt)
    if "vlm_pretrained" in test_preds_cache and "vlm_finetuned" in test_preds_cache:
        yt, yp = test_preds_cache["vlm_pretrained"]; yf = test_preds_cache["vlm_finetuned"][1]
        pairwise["vlm_pretrained_vs_vlm_finetuned"] = mcnemar_compare(yp, yf, yt)
    (outdir / "mcnemar_pairwise.json").write_text(json.dumps(pairwise, indent=2))

    print(f"\nDone. Results saved under: {outdir}")

if __name__ == "__main__":
    main()
