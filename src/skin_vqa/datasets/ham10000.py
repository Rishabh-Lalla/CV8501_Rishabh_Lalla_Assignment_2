import os
import pandas as pd
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..constants import LABELS, LABEL2ID

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(image_size: int = 224, is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

class HamDataset(Dataset):
    def __init__(self, root: str, csv_file: str, transform=None):
        self.root = root
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        # Expect columns: image_id, dx
        if "image_id" not in self.df.columns or "dx" not in self.df.columns:
            raise ValueError(f"CSV {csv_file} missing columns (need image_id, dx)")
        # map labels
        self.df["label"] = self.df["dx"].astype(str)
        bad = set(self.df["label"].unique()) - set(LABELS)
        if bad:
            raise ValueError(f"Unexpected labels in {csv_file}: {bad}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, "images", f"{row['image_id']}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = LABEL2ID[row["label"]]
        return img, label, row["image_id"]
