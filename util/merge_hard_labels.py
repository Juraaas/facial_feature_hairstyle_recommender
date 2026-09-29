import pandas as pd

LABELS = "dataset/hair_dataset/labels.csv"
HARD = "dataset/hair_dataset/hard_case_labels.csv"
TEST = "dataset/hair_dataset/test_set.csv"
OUT = "dataset/hair_dataset/labels_with_hard.csv"

labels = pd.read_csv(LABELS)
hard = pd.read_csv(HARD)
test = pd.read_csv(TEST)

print(f"Original labels: {len(labels)}")
print(f"Hard labels: {len(hard)}")
print(f"Test set: {len(test)}")

hard = hard[
    [
        "filename",
        "hair_type",
        "hairline",
    ]
].copy()

overlap = set(labels["filename"]) & set(hard["filename"])

print(f"Already labeled in original dataset: {len(overlap)}")

labels = labels[~labels["filename"].isin(overlap)].copy()

merged = pd.concat(
    [labels, hard],
    ignore_index=True
)

test_files = set(test["filename"])

before_test_removal = len(merged)

merged = merged[
    ~merged["filename"].isin(test_files)
].copy()

removed_test = before_test_removal - len(merged)

merged = merged.drop_duplicates(
    subset=["filename"],
    keep="last"
).reset_index(drop=True)

merged.to_csv(OUT, index=False)

print()
print("=== MERGED TRAINING LABELS ===")
print(f"Saved: {OUT}")
print(f"Total: {len(merged)}")
print(f"Removed test images: {removed_test}")

print("\nHair type:")
print(
    merged["hair_type"]
    .value_counts(dropna=False)
)

print("\nHairline:")
print(
    merged["hairline"]
    .value_counts(dropna=False)
)