import json, os, csv
from datetime import datetime
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

class RunLogger:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=os.path.join(outdir, 'tensorboard'))
        self.csv_path = os.path.join(outdir, 'metrics.csv')
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'split', 'loss', 'accuracy', 'macro_f1', 'macro_auc'])

    def log_scalar(self, step: int, split: str, loss: float, metrics: Dict[str, Any]):
        self.tb.add_scalar(f"{split}/loss", loss, step)
        self.tb.add_scalar(f"{split}/accuracy", metrics.get('accuracy', float('nan')), step)
        self.tb.add_scalar(f"{split}/macro_f1", metrics.get('macro_f1', float('nan')), step)
        self.tb.add_scalar(f"{split}/macro_auc", metrics.get('macro_auc', float('nan')), step)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([step, split, loss, metrics.get('accuracy', ''), metrics.get('macro_f1', ''), metrics.get('macro_auc', '')])

    def dump_config(self, cfg: Dict[str, Any]):
        with open(os.path.join(self.outdir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

    def dump_metrics(self, metrics: Dict[str, Any]):
        with open(os.path.join(self.outdir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
