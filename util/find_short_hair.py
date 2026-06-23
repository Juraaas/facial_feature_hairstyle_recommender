import sys
sys.path.insert(0, "backend")

import os
import cv2
import time
import random
import numpy as np
import pandas as pd

from src.hair_segmentation import segment_face, get_hair_coverage

IMAGES_DIRS = [
    "dataset/hair_dataset/images",
    "dataset/hair_dataset/train_images",
]

LABELS_PATH = "dataset/hair_dataset/labels.csv"
OUT_PATH = "dataset/hair_dataset/unlabeled_short_candidates.txt"

TARGET_N = 200
MAX_COVERAGE = 0.04
MIN_COVERAGE = 0.001
RESIZE_TO = 128
PRINT_EVERY = 50
RANDOM_SEED = 42


def load_labeled():
    if not os.path.exists(LABELS_PATH):
        return set()

    df = pd.read_csv(LABELS_PATH)
    return set(df["filename"].dropna().astype(str).tolist())


def collect_unlabeled_files(labeled):
    files = []

    for folder in IMAGES_DIRS:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if fname in labeled:
                continue

            files.append((fname, os.path.join(folder, fname)))

    seen = set()
    unique = []
    for fname, path in files:
        if fname not in seen:
            unique.append((fname, path))
            seen.add(fname)

    return unique


def main():
    labeled = load_labeled()
    files = collect_unlabeled_files(labeled)

    random.seed(RANDOM_SEED)
    random.shuffle(files)

    print(f"Labeled files: {len(labeled)}")
    print(f"Unlabeled files to scan: {len(files)}")
    print(f"Target candidates: {TARGET_N}")
    print(f"Coverage range: {MIN_COVERAGE} - {MAX_COVERAGE}")

    candidates = []
    start = time.time()

    for i, (fname, path) in enumerate(files, start=1):
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(
            img,
            (RESIZE_TO, RESIZE_TO),
            interpolation=cv2.INTER_AREA
        )

        hair_mask, _ = segment_face(img)

        if hair_mask is None:
            continue

        coverage = get_hair_coverage(hair_mask)

        if MIN_COVERAGE <= coverage <= MAX_COVERAGE:
            candidates.append((fname, coverage))
            print(f"FOUND {len(candidates)}/{TARGET_N}: {fname} cov={coverage:.4f}")

            if len(candidates) >= TARGET_N:
                break

        if i % PRINT_EVERY == 0:
            elapsed = time.time() - start
            print(
                f"[{i}/{len(files)}] "
                f"found={len(candidates)} "
                f"elapsed={elapsed/60:.1f} min"
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for fname, _ in candidates:
            f.write(fname + "\n")

    debug_csv = OUT_PATH.replace(".txt", "_debug.csv")
    pd.DataFrame(candidates, columns=["filename", "coverage"]).to_csv(debug_csv, index=False)

    print(f"\nSaved candidates: {OUT_PATH}")
    print(f"Saved debug CSV: {debug_csv}")
    print(f"Found {len(candidates)} candidates")


if __name__ == "__main__":
    main()