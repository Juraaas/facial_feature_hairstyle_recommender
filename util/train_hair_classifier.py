import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BALANCED_DIR = "dataset/hair_dataset/balanced"
IMAGES_DIR   = f"{BALANCED_DIR}/images"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"

HAIR_CLASSES     = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

EPOCHS     = 30
LR         = 1e-3
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train_transform = T.compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

class HairDataset(Dataset):
    def __init__(self, csv_path, label_col, classes, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df[label_col].isin(classes)].reset_index(drop=True)
        self.label_col = label_col
        self.classes = classes
        self.transform = transform
        self.class2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]

        for folder in [IMAGES_DIR, TRAIN_IMAGES]:
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                break

        img = Image.open(path).convert("RGB")
        label = self.class2idx[row[self.label_col]]

        if self.transform:
            img = self.transform(img)

        return img, label