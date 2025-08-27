#hybrid_model.py

# Cell 2
import os, math, time, copy, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# >>>> EDIT THIS <<<<
DATASET_ROOT = "dataset2/"  # folder that contains Training/ and Testing/ (case-insensitive)

# training config
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 2
VAL_SPLIT_FROM_TRAIN = 0.15  # if there is no explicit Validation folder, split from Training
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # early stopping
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
print("Device:", DEVICE)


# Cell 4
# Standard ImageNet normalization for ResNet/ViT pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Cell 6
# We use torchvision's pretrained ResNet50 and ViT-B/16.
# We'll remove their classifier heads, concatenate pooled features, and train a small head on top.
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # keep everything except the final FC
        self.features = nn.Sequential(*(list(m.children())[:-1]))  # -> [B, 2048, 1, 1]
        self.out_dim = 2048
        for p in self.features.parameters():
            p.requires_grad = True  # fine-tune (can freeze for small data)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x  # [B, 2048]

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # vit returns logits if we call directly; use the encoder to get features
        self.patch_embed = m._modules['conv_proj']  # (B, 768, H/16, W/16)
        self.cls_token = m.class_token
        self.pos_embed = m.encoder.pos_embedding
        self.encoder = m.encoder
        self.ln = m.encoder.ln
        self.out_dim = 768

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 768, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 768)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.ln(x)
        return x[:, 0, :]  # CLS token, [B, 768]

class HybridResNetViT(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.res = ResNet50Backbone()
        self.vit = ViTBackbone()
        feat_dim = self.res.out_dim + self.vit.out_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f_res = self.res(x)
        f_vit = self.vit(x)
        f = torch.cat([f_res, f_vit], dim=1)
        return self.head(f)

# model = HybridResNetViT(num_classes).to(DEVICE)
# sum(p.numel() for p in model.parameters())/1e6


