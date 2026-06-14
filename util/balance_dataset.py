import cv2
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_CSV   = "dataset/hair_dataset/labels.csv"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"
OUTPUT_DIR   = "dataset/hair_dataset/balanced"

HAIR_CLASSES     = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

TARGET_HAIR     = 200
TARGET_HAIRLINE = 120

np.random.seed(42)

def augment(img, n):
    results = [img]
    ops = [
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=10),
        lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=-10),
        lambda x: cv2.GaussianBlur(x, (3, 3), 0),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=30),
    ]
    while len(results) < n:
        op  = np.random.choice(ops)
        src = results[np.random.randint(len(results))]
        results.append(op(src.copy()))
    return results[1:n]

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

df = pd.read_csv(LABELS_CSV)
aug_counter = 0

def process_class(subset, cls, target):
    global aug_counter
    records = []
    n = len(subset)

    rows = subset.sample(min(n, target), random_state=42) if n >= target else subset

    for _, row in rows.iterrows():
        src = os.path.join(TRAIN_IMAGES, row["filename"])
        dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            records.append({
                "filename":  row["filename"],
                "hair_type": row["hair_type"],
                "hairline":  row["hairline"],
                "augmented": False,
            })

    if n < target:
        need         = target - n
        aug_per_img  = max(1, int(np.ceil(need / n)))
        aug_done     = 0

        for _, row in subset.iterrows():
            if aug_done >= need:
                break
            src_path = os.path.join(TRAIN_IMAGES, row["filename"])
            if not os.path.exists(src_path):
                continue
            img = cv2.imread(src_path)
            if img is None:
                continue
            for aug_img in augment(img, aug_per_img + 1):
                if aug_done >= need:
                    break
                fname = f"aug_{aug_counter:05d}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname), aug_img)
                records.append({
                    "filename":  fname,
                    "hair_type": row["hair_type"],
                    "hairline":  row["hairline"],
                    "augmented": True,
                })
                aug_counter += 1
                aug_done    += 1

        print(f"  {cls}: {n} → +{aug_done} aug = {n + aug_done}")

    return records

print("=== Dataset A: hair_type ===")
records_hair = []
df_hair = df[df["hair_type"].isin(HAIR_CLASSES)].copy()

for cls in HAIR_CLASSES:
    subset = df_hair[df_hair["hair_type"] == cls]
    print(f"{cls}: {len(subset)}", end=" ")
    records_hair.extend(process_class(subset, cls, TARGET_HAIR))

df_hair_balanced = pd.DataFrame(records_hair)
df_hair_balanced.to_csv(f"{OUTPUT_DIR}/hair_type_balanced.csv", index=False)

print(f"\nHair type dataset: {len(df_hair_balanced)}")
print(df_hair_balanced["hair_type"].value_counts())

train_h, val_h = train_test_split(
    df_hair_balanced,
    test_size=0.15,
    stratify=df_hair_balanced["hair_type"],
    random_state=42,
)
train_h.to_csv(f"{OUTPUT_DIR}/train_hair.csv", index=False)
val_h.to_csv(f"{OUTPUT_DIR}/val_hair.csv", index=False)
print(f"Train: {len(train_h)}, Val: {len(val_h)}")


print("\n=== Dataset B: hairline ===")
records_hairline = []
df_hairline = df[df["hairline"].isin(HAIRLINE_CLASSES)].copy()

target_hl = max(TARGET_HAIRLINE, df_hairline["hairline"].value_counts().min() * 2)

for cls in HAIRLINE_CLASSES:
    subset = df_hairline[df_hairline["hairline"] == cls]
    print(f"{cls}: {len(subset)}", end=" ")
    records_hairline.extend(process_class(subset, cls, target_hl))

df_hairline_balanced = pd.DataFrame(records_hairline)
df_hairline_balanced.to_csv(f"{OUTPUT_DIR}/hairline_balanced.csv", index=False)

print(f"\nHairline dataset: {len(df_hairline_balanced)}")
print(df_hairline_balanced["hairline"].value_counts())

train_hl, val_hl = train_test_split(
    df_hairline_balanced,
    test_size=0.15,
    stratify=df_hairline_balanced["hairline"],
    random_state=42,
)
train_hl.to_csv(f"{OUTPUT_DIR}/train_hairline.csv", index=False)
val_hl.to_csv(f"{OUTPUT_DIR}/val_hairline.csv", index=False)
print(f"Train: {len(train_hl)}, Val: {len(val_hl)}")
print(f"Images in {OUTPUT_DIR}/images/: {len(os.listdir(OUTPUT_DIR+'/images'))}")