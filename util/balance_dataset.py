import cv2, os, shutil, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

LABELS_CSV = "dataset/hair_dataset/labels_with_hard.csv"
IMAGES_DIR = "dataset/celeba/celeba_hq_256"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"
OUTPUT_DIR = "dataset/hair_dataset/balanced"

HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

MAX_AUG_FACTOR = 4
VAL_FRACTION = 0.15
SEED = 42
np.random.seed(SEED)
aug_counter = 0

def find_image(filename):
    for folder in [IMAGES_DIR, TRAIN_IMAGES,
                   "dataset/hair_dataset/images"]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

def augment_one(img):
    ops = [
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.convertScaleAbs(
            x,
            alpha=float(np.random.uniform(0.85, 1.15)),
            beta=int(np.random.randint(-15, 15)),
        ),
        lambda x: cv2.GaussianBlur(x, (3, 3), 0),
        lambda x: cv2.convertScaleAbs(
            x,
            alpha=float(np.random.uniform(0.9, 1.1)),
            beta=int(np.random.randint(-20, 20)),
        ),
        lambda x: cv2.convertScaleAbs(
            x,
            alpha=float(np.random.uniform(0.85, 1.0)),
            beta=0,
        ),
    ]
    return ops[np.random.randint(len(ops))](img.copy())


def process_class(subset_df, cls, target, label_col):
    global aug_counter
    records = []
    n = len(subset_df)
    copied = 0
    for _, row in subset_df.iterrows():
        src = find_image(row["filename"])
        if src is None:
            continue
        dst = os.path.join(OUTPUT_DIR, "images", row["filename"])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        records.append({
            "filename":  row["filename"],
            "hair_type": row.get("hair_type"),
            "hairline":  row.get("hairline"),
            "augmented": False,
        })
        copied += 1

    need = max(0, target - copied)
    aug_done = 0
    if need > 0:
        per_img = min(MAX_AUG_FACTOR, int(np.ceil(need / max(n, 1))) + 1)
        for _, row in subset_df.iterrows():
            if aug_done >= need:
                break
            src = find_image(row["filename"])
            if src is None:
                continue
            img = cv2.imread(src)
            if img is None:
                continue
            for _ in range(per_img):
                if aug_done >= need:
                    break
                aug_img = augment_one(img)
                fname = f"aug_{aug_counter:06d}.jpg"
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "images", fname),
                    aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                records.append({
                    "filename": fname,
                    "hair_type": row.get("hair_type"),
                    "hairline": row.get("hairline"),
                    "augmented": True,
                })
                aug_counter += 1
                aug_done += 1

    actual = copied + aug_done
    print(f"{cls:12s}: {n:4d} orig all kept"
          f" + {aug_done:4d} aug = {actual:5d}  (target {target})")
    return records

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

df_all = pd.read_csv(LABELS_CSV)

df_ht = df_all[df_all["hair_type"].isin(HAIR_CLASSES)].copy()
df_hl = df_all[df_all["hairline"].isin(HAIRLINE_CLASSES)].copy()


print("\n=== HAIR TYPE ===")
counts_ht = df_ht["hair_type"].value_counts()
print("Raw counts:\n", counts_ht.to_string())

minority_counts = {
    c: counts_ht[c] for c in HAIR_CLASSES
    if c != "straight" and c in counts_ht
}
target_ht = max(minority_counts.values())
print(f"\nTarget for minority classes: {target_ht}  "
      f"(largest minority = wavy)")
print("straight keeps all its originals — WeightedRandomSampler "
      "will balance during training.\n")

records_ht = []
for cls in HAIR_CLASSES:
    sub = df_ht[df_ht["hair_type"] == cls]
    t   = len(sub) if cls == "straight" else target_ht
    records_ht.extend(process_class(sub, cls, t, "hair_type"))

df_ht_bal = pd.DataFrame(records_ht)
print(f"\nFinal hair type counts:\n"
      f"{df_ht_bal['hair_type'].value_counts().to_string()}")

train_ht, val_ht = train_test_split(
    df_ht_bal,
    test_size = VAL_FRACTION,
    stratify  = df_ht_bal["hair_type"],
    random_state = SEED,
)
train_ht.to_csv(f"{OUTPUT_DIR}/train_hair.csv", index=False)
val_ht.to_csv(  f"{OUTPUT_DIR}/val_hair.csv", index=False)
df_ht_bal.to_csv(f"{OUTPUT_DIR}/hair_type_balanced.csv", index=False)
print(f"Train: {len(train_ht)}  Val: {len(val_ht)}")


print("\n=== HAIRLINE ===")
counts_hl = df_hl["hairline"].value_counts()
print("Raw counts:\n", counts_hl.to_string())

minority_hl = {
    c: counts_hl[c] for c in HAIRLINE_CLASSES
    if c != "normal" and c in counts_hl
}
target_hl = max(minority_hl.values())
print(f"\nTarget for minority classes: {target_hl}  "
      f"(largest minority = uneven)")
print("normal keeps all its originals.\n")

records_hl = []
for cls in HAIRLINE_CLASSES:
    sub = df_hl[df_hl["hairline"] == cls]
    t = len(sub) if cls == "normal" else target_hl
    records_hl.extend(process_class(sub, cls, t, "hairline"))

df_hl_bal = pd.DataFrame(records_hl)
print(f"\nFinal hairline counts:\n"
      f"{df_hl_bal['hairline'].value_counts().to_string()}")

train_hl, val_hl = train_test_split(
    df_hl_bal,
    test_size = VAL_FRACTION,
    stratify = df_hl_bal["hairline"],
    random_state = SEED,
)
train_hl.to_csv(f"{OUTPUT_DIR}/train_hairline.csv", index=False)
val_hl.to_csv(  f"{OUTPUT_DIR}/val_hairline.csv", index=False)
df_hl_bal.to_csv(f"{OUTPUT_DIR}/hairline_balanced.csv", index=False)
print(f"Train: {len(train_hl)}  Val: {len(val_hl)}")

print(f"\nImages in balanced dir: "
      f"{len(os.listdir(OUTPUT_DIR + '/images'))}")
print("Done.")