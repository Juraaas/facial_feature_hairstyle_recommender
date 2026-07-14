import cv2
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_CSV = "dataset/hair_dataset/labels.csv"
LABELS_COV_CSV = "dataset/hair_dataset/labels_with_coverage.csv"
IMAGES_DIR = "dataset/hair_dataset/images"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"
OUTPUT_DIR = "dataset/hair_dataset/balanced"

HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

MAX_AUG_FACTOR = 3
MIN_CLASS_SIZE = 150

np.random.seed(42)

def find_image(filename):
    for folder in [TRAIN_IMAGES, IMAGES_DIR]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

def augment(img, n):
    results = [img]
    ops = [
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.convertScaleAbs(x, alpha=np.random.uniform(0.85, 1.15),
                                      beta=np.random.randint(-15, 15)),
        lambda x: cv2.GaussianBlur(x, (3, 3), 0),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=np.random.randint(-20, 20)),
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
df_cov = pd.read_csv(LABELS_COV_CSV)
aug_counter = 0

print("=== Dataset A: hair_type ===")
df_hair = df[df["hair_type"].isin(HAIR_CLASSES)].copy()

train_orig, val_orig = train_test_split(
    df_hair,
    test_size=0.15,
    stratify=df_hair["hair_type"],
    random_state=42,
)

print(f"Original train: {len(train_orig)}, val: {len(val_orig)}")
print("Train distribution:")
print(train_orig["hair_type"].value_counts())

train_records = []
val_records = []

for _, row in val_orig.iterrows():
    src = find_image(row["filename"])
    if src is None:
        continue
    dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
    shutil.copy2(src, dst)
    val_records.append({
        "filename": row["filename"],
        "hair_type": row["hair_type"],
        "hairline": row["hairline"],
        "augmented": False,
    })

class_counts = train_orig["hair_type"].value_counts()
max_count = class_counts.max()

for cls in HAIR_CLASSES:
    cls_df = train_orig[train_orig["hair_type"] == cls]
    n = len(cls_df)

    for _, row in cls_df.iterrows():
        src = find_image(row["filename"])
        if src is None:
            continue
        dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
        shutil.copy2(src, dst)
        train_records.append({
            "filename": row["filename"],
            "hair_type": cls,
            "hairline": row["hairline"],
            "augmented": False,
        })
    target = min(max(MIN_CLASS_SIZE, max_count), n * MAX_AUG_FACTOR)
    need = max(0, target - n)

    if need > 0:
        aug_per = max(1, int(np.ceil(need / n)))
        aug_done = 0
        for _, row in cls_df.iterrows():
            if aug_done >= need:
                break
            src_path = find_image(row["filename"])
            if not src_path:
                continue
            img = cv2.imread(src_path)
            if img is None:
                continue
            for aug_img in augment(img, aug_per + 1):
                if aug_done >= need:
                    break
                fname = f"aug_{aug_counter:05d}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname), aug_img)
                train_records.append({
                    "filename":  fname,
                    "hair_type": cls,
                    "hairline":  row["hairline"],
                    "augmented": True,
                })
                aug_counter += 1
                aug_done    += 1

        print(f"{cls}: {n} orig + {aug_done} aug = {n + aug_done}")
    else:
        print(f"{cls}: {n} orig (no aug needed)")

df_train_hair = pd.DataFrame(train_records)
df_val_hair = pd.DataFrame(val_records)

df_train_hair.to_csv(f"{OUTPUT_DIR}/train_hair.csv", index=False)
df_val_hair.to_csv(f"{OUTPUT_DIR}/val_hair.csv", index=False)

print(f"\nHair train: {len(df_train_hair)}")
print(df_train_hair["hair_type"].value_counts())
print(f"Hair val: {len(df_val_hair)}")
print(df_val_hair["hair_type"].value_counts())

print("\n=== Dataset B: hairline ===")

df_hl  = df_cov[df_cov["hairline"].isin(HAIRLINE_CLASSES)].copy()

train_hl_orig, val_hl_orig = train_test_split(
    df_hl,
    test_size=0.15,
    stratify=df_hl["hairline"],
    random_state=42
)

print(f"Original train: {len(train_hl_orig)}, val: {len(val_hl_orig)}")
print("Train distribution:")
print(train_hl_orig["hairline"].value_counts())
print("Coverage bins in train:")
print(pd.crosstab(train_hl_orig["hairline"], train_hl_orig["coverage_bin"]))

train_hl_records = []
val_hl_records = []

for _, row in val_hl_orig.iterrows():
    src = find_image(row["filename"])
    if src is None:
        continue
    dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
    shutil.copy2(src, dst)
    val_hl_records.append({
        "filename": row["filename"],
        "hair_type": row["hair_type"],
        "hairline": row["hairline"],
        "augmented": False,
    })

BIN_PROPORTIONS = {"long": 0.40, "medium": 0.35, "short": 0.25}
hl_class_counts = train_hl_orig["hairline"].value_counts()
hl_max = hl_class_counts.max()
HL_TARGET = min(hl_max, hl_class_counts.min() * MAX_AUG_FACTOR)

for cls in HAIRLINE_CLASSES:
    cls_df = train_hl_orig[train_hl_orig["hairline"] == cls]
    n = len(cls_df)

    for _, row in cls_df.iterrows():
        src = find_image(row["filename"])
        if src is None:
            continue
        dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
        shutil.copy2(src, dst)
        train_hl_records.append({
            "filename": row["filename"],
            "hair_type": row["hair_type"],
            "hairline": cls,
            "augmented": False,
        })

    need = max(0, HL_TARGET - n)
    if need > 0:
        aug_done = 0
        for bin_name, proportion in BIN_PROPORTIONS.items():
            bin_need = int(need * proportion)
            bin_df = cls_df[cls_df["coverage_bin"] == bin_name]
            bn = len(bin_df)
            if bn == 0 or bin_need == 0:
                continue

            bin_need = min(bin_need, bn * MAX_AUG_FACTOR)
            aug_per = max(1, int(np.ceil(bin_need / bn)))
            bin_done = 0

            for _, row in bin_df.iterrows():
                if bin_done >= bin_need:
                    break
                src_path = find_image(row["filename"])
                if not src_path:
                    continue
                img = cv2.imread(src_path)
                if img is None:
                    continue
                for aug_img in augment(img, aug_per + 1):
                    if bin_done >= bin_need:
                        break
                    fname = f"aug_{aug_counter:05d}.png"
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, "images", fname), aug_img
                    )
                    train_hl_records.append({
                        "filename": fname,
                        "hair_type": row["hair_type"],
                        "hairline": cls,
                        "augmented": True,
                    })
                    aug_counter += 1
                    bin_done += 1
                    aug_done += 1

        print(f"{cls}: {n} orig + {aug_done} aug = {n + aug_done}")
    else:
        print(f"{cls}: {n} orig (no aug needed)")

df_train_hl = pd.DataFrame(train_hl_records)
df_val_hl = pd.DataFrame(val_hl_records)

df_train_hl.to_csv(f"{OUTPUT_DIR}/train_hairline.csv", index=False)
df_val_hl.to_csv(f"{OUTPUT_DIR}/val_hairline.csv", index=False)

print(f"\nHairline train: {len(df_train_hl)}")
print(df_train_hl["hairline"].value_counts())
print(f"Hairline val: {len(df_val_hl)}")
print(df_val_hl["hairline"].value_counts())

print(f"\nImages in {OUTPUT_DIR}/images/: "
      f"{len(os.listdir(OUTPUT_DIR+'/images'))}")