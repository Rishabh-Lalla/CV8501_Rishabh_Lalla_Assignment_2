import torch
import torch.nn as nn
from torchvision import models

from ..constants import LABELS

def build_baseline(arch: str = "resnet50", pretrained: bool = True, drop_rate: float = 0.2):
    num_classes = len(LABELS)

    if arch.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        return m

    elif arch.lower() == "vit_b_16":
        # torchvision ViT
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )
        return m

    else:
        raise ValueError(f"Unknown arch: {arch}")
