import pandas as pd
import shutil
import os

LABELS_PATH = "dataset/hair_dataset/labels.csv"
DEST_DIR = "dataset/hair_dataset/train_images"

SOURCE_DIRS = [
    "dataset/hair_dataset/images",
    "dataset/short_hair_candidates",
]

MISSING_REPORT = "dataset/hair_dataset/missing_images.csv"

df = pd.read_csv(LABELS_PATH)

print(f"Rows in labels.csv: {len(df)}")
print(f"Unique filenames: {df['filename'].nunique()}")

duplicates = df[df.duplicated("filename", keep=False)]

if len(duplicates) > 0:
    print(f"\nFound {duplicates['filename'].nunique()} duplicated filenames")
    duplicates.to_csv(
        "dataset/hair_dataset/duplicate_labels.csv",
        index=False
    )
else:
    print("No duplicate filenames found")

os.makedirs(DEST_DIR, exist_ok=True)

copied = 0
already_present = 0
missing_files = []

for _, row in df.iterrows():
    filename = row["filename"]

    found_path = None

    for source_dir in SOURCE_DIRS:
        candidate = os.path.join(source_dir, filename)

        if os.path.exists(candidate):
            found_path = candidate
            break

    if found_path is None:
        missing_files.append(filename)
        continue

    dst = os.path.join(DEST_DIR, filename)

    if not os.path.exists(dst):
        shutil.copy2(found_path, dst)
        copied += 1
    else:
        already_present += 1

print(f"Copied: {copied}")
print(f"Already present: {already_present}")
print(f"Missing: {len(missing_files)}")

if missing_files:
    pd.DataFrame({
        "filename": missing_files
    }).to_csv(MISSING_REPORT, index=False)

    print(f"Missing report saved to:")
    print(MISSING_REPORT)

train_images_count = len([
    f for f in os.listdir(DEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Labels: {df['filename'].nunique()}")
print(f"Train images: {train_images_count}")

if train_images_count + len(missing_files) == df["filename"].nunique():
    print("Dataset is consistent")
else:
    print("Counts do not match")