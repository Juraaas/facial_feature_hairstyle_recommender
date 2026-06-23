import sys
sys.path.insert(0, "backend")
import os
import time
import cv2
import numpy as np
import pandas as pd
from src.hair_segmentation import segment_face

LABELS_PATH = "dataset/hair_dataset/labels.csv"
OUT_PATH = "dataset/hair_dataset/labels_with_coverage.csv"
PARTIAL_PATH = "dataset/hair_dataset/labels_with_coverage_partial.csv"

IMAGES_DIR = "dataset/hair_dataset/images"
TRAIN_IMAGES = "dataset/hair_dataset/train_images"

RESIZE_TO = 160
SAVE_EVERY = 50
PRINT_EVERY = 10

def find_image_path(filename):
    for folder in [TRAIN_IMAGES, IMAGES_DIR]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

def coverage_to_bin(cov):
    if cov is None:
        return "unknown"
    if cov < 0.04:
        return "short"
    if cov < 0.10:
        return "medium"
    return "long"

def load_existing_results():
    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
    elif os.path.exists(PARTIAL_PATH):
        old = pd.read_csv(PARTIAL_PATH)
    else:
        return {}

    if "filename" not in old.columns or "coverage_bin" not in old.columns:
        return {}

    return {
        row["filename"]: {
            "coverage": row.get("coverage", np.nan),
            "coverage_bin": row.get("coverage_bin", "unknown"),
        }
        for _, row in old.iterrows()
        if pd.notna(row.get("coverage_bin", np.nan))
    }

def process_one(filename):
    path = find_image_path(filename)
    if path is None:
        return None, "unknown", "missing"
    img = cv2.imread(path)
    if img is None:
        return None, "unknown", "unreadable"

    img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
    mask, _ = segment_face(img)

    if mask is None:
        return None, "unknown", "segmentation_failed"
    cov = float(np.sum(mask > 0) / mask.size)
    cov_bin = coverage_to_bin(cov)

    return cov, cov_bin, "ok"

def save_checkpoint(df, coverages, bins, path):
    temp = df.copy()
    temp["coverage"] = coverages
    temp["coverage_bin"] = bins
    temp.to_csv(path, index=False)

def main():
    df = pd.read_csv(LABELS_PATH)
    existing = load_existing_results()

    coverages = []
    bins = []

    start = time.time()
    processed_now = 0
    reused = 0

    total = len(df)

    for i, row in df.iterrows():
        filename = row["filename"]

        if filename in existing:
            cov = existing[filename]["coverage"]
            cov_bin = existing[filename]["coverage_bin"]

            coverages.append(cov)
            bins.append(cov_bin)
            reused += 1
            continue

        cov, cov_bin, status = process_one(filename)

        coverages.append(cov if cov is not None else np.nan)
        bins.append(cov_bin)

        processed_now += 1

        if (i + 1) % PRINT_EVERY == 0 or status != "ok":
            elapsed = time.time() - start
            print(
                f"[{i+1}/{total}] "
                f"file={filename} "
                f"status={status} "
                f"cov={cov if cov is not None else 'NA'} "
                f"bin={cov_bin} "
                f"processed_now={processed_now} "
                f"reused={reused} "
                f"elapsed={elapsed/60:.1f} min"
            )

        if (i + 1) % SAVE_EVERY == 0:
            save_checkpoint(df.iloc[:i+1].copy(), coverages, bins, PARTIAL_PATH)
            print(f"Checkpoint saved: {PARTIAL_PATH}")

    save_checkpoint(df, coverages, bins, OUT_PATH)

    print("\nSaved final file:", OUT_PATH)
    print("\nCounts:")
    print(pd.crosstab(df["hairline"], pd.Series(bins, name="coverage_bin")))

    print("\nPercent by hairline:")
    print(
        pd.crosstab(
            df["hairline"],
            pd.Series(bins, name="coverage_bin"),
            normalize="index"
        ).round(3) * 100
    )

    result_df = df.copy()
    result_df["coverage"] = coverages
    result_df["coverage_bin"] = bins

    print("\nCoverage stats by hairline:")
    print(result_df.groupby("hairline")["coverage"].describe())


if __name__ == "__main__":
    main()

