import pandas as pd
import shutil
import os

df = pd.read_csv("dataset/hair_dataset/labels.csv")

os.makedirs("dataset/hair_dataset/train_images", exist_ok=True)

copied = 0
missing = 0

for _, row in df.iterrows():
    src = os.path.join("dataset/hair_dataset/images", row["filename"])
    dst = os.path.join("dataset/hair_dataset/train_images", row["filename"])
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f"Copied: {copied}")
print(f"Missing: {missing}")