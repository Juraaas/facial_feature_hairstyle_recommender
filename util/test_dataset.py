import pandas as pd
import os
from PIL import Image

for csv_path in [
    "dataset/hair_dataset/balanced/train_hairline.csv",
    "dataset/hair_dataset/balanced/val_hairline.csv",
]:
    df = pd.read_csv(csv_path)
    print(f"\n{csv_path}")
    print(df["hairline"].value_counts())
    
    missing = 0
    for fname in df["filename"]:
        found = False
        for folder in [
            "dataset/hair_dataset/balanced/images",
            "dataset/hair_dataset/train_images",
            "dataset/hair_dataset/images",
        ]:
            if os.path.exists(os.path.join(folder, fname)):
                found = True
                break
        if not found:
            missing += 1
    print(f"Missing files: {missing}/{len(df)}")
    
    errors = 0
    for fname in df["filename"][:20]:
        for folder in [
            "dataset/hair_dataset/balanced/images",
            "dataset/hair_dataset/train_images", 
            "dataset/hair_dataset/images",
        ]:
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    _ = img.size
                except Exception as e:
                    print(f"  ERROR {fname}: {e}")
                    errors += 1
                break
    print(f"Image errors in first 20: {errors}")