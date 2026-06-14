import cv2
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_CSV = "dataset/hair_dataset/labels.csv"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"
OUTPUT_DIR = "dataset/hair_dataset/balanced"
TARGET_HAIR = 120
TARGET_HAIRLINE = 60

HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

np.random.seed(42)

def augment(img, n):
    results = [img]
    ops = [
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=10),
        lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=-10),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.GaussianBlur(x, (3,3), 0),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=30),
    ]
    while len(results) < n:
        op = np.random.choice(ops)
        src = results[np.random.randint(len(results))]
        results.append(op(src))
    return results[1:n]

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

df = pd.read_csv(LABELS_CSV)
df_hair = df[df["hair_type"].isin(HAIR_CLASSES)].copy()
df_hairline = df[df["hairline"].isin(HAIRLINE_CLASSES)].copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

df = pd.read_csv(LABELS_CSV)
records = []
aug_counter = 0

def copy_or_augment(subset, cls, label_key, target, label_value):
    global aug_counter
    n = len(subset)
    added = []

    if n >= target:
        selected = subset.sample(target, random_state=42)
        for _, row in selected.iterrows():
            src = os.path.join(TRAIN_IMAGES, row["filename"])
            dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)
                added.append({
                    "filename": row["filename"],
                    "hair_type": row["hair_type"],
                    "hairline": row["hairline"],
                    "augmented": False,
                })
    else:
        need = target - n
        for _, row in subset.iterrows():
            src = os.path.join(TRAIN_IMAGES, row["filename"])
            dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)
                added.append({
                    "filename": row["filename"],
                    "hair_type": row["hair_type"],
                    "hairline": row["hairline"],
                    "augmented": False,
                })
        aug_per_img = max(1, int(np.ceil(need / max(n, 1))))
        aug_done = 0
        for _, row in subset.iterrows():
            if aug_done >= need:
                break
            src_path = os.path.join(TRAIN_IMAGES, row["filename"])
            if not os.path.exists(src_path):
                continue
            img = cv2.imread(src_path)
            if img is None:
                continue
            augs = augment(img, aug_per_img + 1)
            for aug_img in augs:
                if aug_done >= need:
                    break
                fname = f"aug_{aug_counter:05d}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname), aug_img)
                added.append({
                    "filename": fname,
                    "hair_type": row["hair_type"],
                    "hairline": row["hairline"],
                    "augmented": True,
                })
                aug_counter += 1
                aug_done += 1
        print(f"generated {aug_done} augmentations for {cls}")
    return added

print("=== Balancing hair_type ===")
df_hair = df[df["hair_type"].isin(HAIR_CLASSES)].copy()

for cls in HAIR_CLASSES:
    subset = df_hair[df_hair["hair_type"] == cls]
    print(f"{cls}: {len(subset)} → {TARGET_HAIR}")
    records.extend(copy_or_augment(subset, cls, "hair_type", TARGET_HAIR, cls))

print("=== Balancing hairline ===")
df_so_far = pd.DataFrame(records)

for cls in HAIRLINE_CLASSES:
    current = df_so_far[df_so_far["hairline"] == cls]
    n = len(current)
    print(f"{cls}: {n}", end="")

    if n < TARGET_HAIRLINE:
        need = TARGET_HAIRLINE - n 
        originals = df[df["hairline"] == cls]
        aug_done  = 0

        for _, row in originals.iterrows():
            if aug_done >= need:
                break
            src_path = os.path.join(TRAIN_IMAGES, row["filename"])
            if not os.path.exists(src_path):
                continue
            img = cv2.imread(src_path)
            if img is None:
                continue
            augs = augment(img, 4)
            for aug_img in augs:
                if aug_done >= need:
                    break
                fname = f"aug_{aug_counter:05d}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname), aug_img)
                records.append({
                    "filename":  fname,
                    "hair_type": row["hair_type"],
                    "hairline":  cls,
                    "augmented": True,
                })
                aug_counter += 1
                aug_done    += 1
    else:
        print(f" → OK")

df_balanced = pd.DataFrame(records)
assert list(df_balanced.columns) == ["filename", "hair_type", "hairline", "augmented"]

df_balanced.to_csv(f"{OUTPUT_DIR}/labels_balanced.csv", index=False)

print(f"\n=== Final dataset ===")
print(f"Total: {len(df_balanced)}")
print("\nHair type:")
print(df_balanced["hair_type"].value_counts())
print("\nHairline:")
print(df_balanced["hairline"].value_counts())

train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.15,
    stratify=df_balanced["hair_type"],
    random_state=42,
)
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)

print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")