import argparse, os, random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from skin_vqa.datasets.ham10000 import HamDataset, build_transforms
from skin_vqa.models.baseline import build_baseline
from skin_vqa.utils.metrics import multiclass_metrics
from skin_vqa.utils.logging_utils import RunLogger
from skin_vqa.utils.plots import save_confusion_matrix, save_roc_curves, save_pr_curves
from skin_vqa.constants import LABELS, LABEL2ID

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_split(data_root, split_name):
    csv = os.path.join(data_root, 'splits', f'{split_name}.csv')
    return csv

def compute_class_weights(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df['dx'].value_counts().reindex(LABELS).fillna(0).values.astype(float)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def run(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.outdir, exist_ok=True)
    logger = RunLogger(args.outdir)
    logger.dump_config(vars(args))

    # Datasets
    train_ds = HamDataset(args.data_root, load_split(args.data_root, 'train'), transform=build_transforms(args.image_size, True))
    val_ds   = HamDataset(args.data_root, load_split(args.data_root, 'val'),   transform=build_transforms(args.image_size, False))
    test_ds  = HamDataset(args.data_root, load_split(args.data_root, 'test'),  transform=build_transforms(args.image_size, False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    model = build_baseline(args.arch, pretrained=not args.no_pretrained, drop_rate=args.drop)
    model.to(device)

    # Loss / Optim
    class_weights = compute_class_weights(load_split(args.data_root, 'train')).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.precision == 'fp16' and torch.cuda.is_available()))

    def evaluate(loader, step_tag='val'):
        model.eval()
        all_probs, all_labels = [], []
        total_loss = 0.0
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=(args.precision == 'fp16' and torch.cuda.is_available())):
                    logits = model(x)
                    loss = criterion(logits, y)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.cpu().numpy())
                total_loss += loss.item() * x.size(0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = multiclass_metrics(all_labels, all_probs)
        avg_loss = total_loss / len(loader.dataset)
        logger.log_scalar(step=current_epoch, split=step_tag, loss=avg_loss, metrics=metrics)
        return metrics, avg_loss, all_labels, all_probs

    best_f1 = -1.0
    for current_epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total = 0; total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(args.precision == 'fp16' and torch.cuda.is_available())):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += x.size(0)
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        # Val
        val_metrics, val_loss, y_val, p_val = evaluate(val_loader, 'val')
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            ckpt_dir = os.path.join(args.outdir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({'epoch': current_epoch, 'model': model.state_dict()}, os.path.join(ckpt_dir, 'best.pt'))

    # Final test evaluation with best weights
    ckpt_path = os.path.join(args.outdir, 'checkpoints', 'best.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])

    test_metrics, test_loss, y_true, y_prob = evaluate(test_loader, 'test')
    # Save plots
    from skin_vqa.utils.plots import save_confusion_matrix, save_roc_curves, save_pr_curves
    save_confusion_matrix(np.array(test_metrics['confusion_matrix']), os.path.join(args.outdir, 'confusion_matrix.png'))
    save_roc_curves(y_true, y_prob, os.path.join(args.outdir, 'roc_curves.png'))
    save_pr_curves(y_true, y_prob, os.path.join(args.outdir, 'pr_curves.png'))
    logger.dump_metrics(test_metrics)
    print(json.dumps(test_metrics, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--splits', type=str, default=None, help='(unused, kept for compatibility)')
    ap.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'vit_b_16'])
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=str, required=True)
    args = ap.parse_args()
    run(args)
