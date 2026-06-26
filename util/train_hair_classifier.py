import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix

BALANCED_DIR = "dataset/hair_dataset/balanced"
IMAGES_DIR = f"{BALANCED_DIR}/images"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"

HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train_transform = T.Compose([
    T.Resize((256, 256)), 
    T.RandomResizedCrop((224, 224), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(8),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.03),
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
    
class HairClassifier(nn.Module):
    def __init__(self, num_hair=4, num_hairline=3):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = 576

        self.head_hair = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.35),
            nn.Linear(128, num_hair)
        )
        self.head_hairline = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Hardswish(),
            nn.Dropout(0.35),
            nn.Linear(64, num_hairline),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.head_hair(x), self.head_hairline(x)
    
    def forward_hair(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.head_hair(x)

    def forward_hairline(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.head_hairline(x)
    
def train_head(model, head_name, train_csv, val_csv, label_col, classes):
    print(f"\n{'='*50}")
    print(f"Training: {head_name}")
    print(f"{'='*50}")

    train_ds = HairDataset(train_csv, label_col, classes, train_transform)
    val_ds   = HairDataset(val_csv,   label_col, classes, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_path = f"models/hair_{head_name}_best.pt"
    os.makedirs("models", exist_ok=True)

    if head_name == "hair_type":
        head_params = list(model.head_hair.parameters())
    else:
        head_params = list(model.head_hairline.parameters())

    WARMUP_EPOCHS = 8
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=1e-3)

    print(f"Phase 1: warmup {WARMUP_EPOCHS} epochs (backbone frozen)")
    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_hair(imgs) if head_name == "hair_type" \
                     else model.forward_hairline(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)
        print(f"Warmup {epoch:02d}/{WARMUP_EPOCHS} | "
              f"loss={loss_sum/total:.3f} | train_acc={correct/total:.3f}")

    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": LR},
        {"params": head_params, "lr": LR * 5},
    ], weight_decay=1e-3)

    FINETUNE_EPOCHS = EPOCHS - WARMUP_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6
    )
    best_acc = 0
    patience = 15
    no_improve = 0

    print(f"\nPhase 2: fine-tune {FINETUNE_EPOCHS} epochs (backbone unfrozen)")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if head_name == "hair_type":
                logits, _ = model(imgs)
            else:
                _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)

        train_acc = correct / total
        train_loss = loss_sum / total

        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if head_name == "hair_type":
                    logits, _ = model(imgs)
                else:
                    _, logits = model(imgs)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
              f"loss={train_loss:.3f} | "
              f"train_acc={train_acc:.3f} | "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            cm = confusion_matrix(all_labels, all_preds)
            print(f"Confusion matrix ({classes}):")
            print(cm)
            print(f"✓ saved best (val_acc={val_acc:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_acc for {head_name}: {best_acc:.3f}")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    return model
    
def export_onnx(model):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    out_path = "models/hair_classifier.onnx"

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["hair_type", "hairline"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nExported ONNX → {out_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(out_path,
                                providers=["CPUExecutionProvider"])
    dummy_np = dummy.cpu().numpy()
    out = sess.run(None, {"input": dummy_np})
    print(f"ONNX output shapes: {[o.shape for o in out]}")
    print("ONNX verification OK")


if __name__ == "__main__":
    model = HairClassifier(
        num_hair=len(HAIR_CLASSES),
        num_hairline=len(HAIRLINE_CLASSES)
    ).to(DEVICE)

    model = train_head(
        model,
        head_name = "hair_type",
        train_csv = f"{BALANCED_DIR}/train_hair.csv",
        val_csv = f"{BALANCED_DIR}/val_hair.csv",
        label_col = "hair_type",
        classes = HAIR_CLASSES,
    )

    model = train_head(
        model,
        head_name = "hairline",
        train_csv = f"{BALANCED_DIR}/train_hairline.csv",
        val_csv = f"{BALANCED_DIR}/val_hairline.csv",
        label_col = "hairline",
        classes = HAIRLINE_CLASSES,
    )

    export_onnx(model)