"""
Evaluate hair classifier on a held-out test set.
Produces accuracy, macro F1, confusion matrix,
and hard-case rate (hair_conf < 0.70).

Usage:
    python util/evaluate_classifier.py
    python util/evaluate_classifier.py --csv path/to/test_labels.csv
    python util/evaluate_classifier.py --csv path/to/test_labels.csv --images path/to/images
"""

import sys, os, argparse
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.src.hair_classifier import classify_hair
from backend.src.hair_segmentation import segment_face

DEFAULT_CSV    = "dataset/hair_dataset/labels.csv"
DEFAULT_IMAGES = "dataset/celeba/celeba_hq_256"
IMAGE_DIRS = [
    "dataset/hair_dataset/images",
    "dataset/short_hair_candidates",
    "dataset/celeba/celeba_hq_256",
    "dataset/hair_dataset/train_images",
]
HARD_THRESHOLD = 0.70
TEST_FRACTION  = 0.15
TEST_SEED = 99 

HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

def find_image(filename):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, filename)

        if os.path.exists(path):
            return path

    return None

def load_test_set(
    csv_path: str,
    fraction: float,
    seed: int,
    fixed_test: str = "",
) -> pd.DataFrame:

    if fixed_test:
        test = pd.read_csv(fixed_test)

        print(
            f"Fixed test set: {len(test)} images "
            f"loaded from {fixed_test}"
        )

        return test.reset_index(drop=True)

    df = pd.read_csv(csv_path)

    df = df[
        df["hair_type"].isin(HAIR_CLASSES) |
        df["hairline"].isin(HAIRLINE_CLASSES)
    ].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < fraction

    test = df[mask].reset_index(drop=True)

    print(
        f"Test set: {len(test)} images "
        f"({fraction*100:.0f}% of {len(df)} labeled)"
    )

    return test


def run_evaluation(df: pd.DataFrame, images_dir: str) -> dict:
    rows = []
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating"
    ):
        fname = row["filename"]
        true_ht = str(row.get("hair_type", "")).strip()
        true_hl = str(row.get("hairline", "")).strip()
        image_path = find_image(fname)

        if image_path is None:
            rows.append({
                "filename": fname,

                "hair_true": true_ht if true_ht in HAIR_CLASSES else None,
                "hair_pred": None,
                "hair_conf": None,
                "hair_status": "missing",
                "hair_correct": False,

                "hairline_true": true_hl if true_hl in HAIRLINE_CLASSES else None,
                "hairline_pred": None,
                "hairline_conf": None,
                "hairline_status": "missing",
                "hairline_correct": False,

                "error": "image_not_found",
            })
            continue

        img = cv2.imread(image_path)

        if img is None:
            rows.append({
                "filename": fname,

                "hair_true": true_ht if true_ht in HAIR_CLASSES else None,
                "hair_pred": None,
                "hair_conf": None,
                "hair_status": "error",
                "hair_correct": False,

                "hairline_true": true_hl if true_hl in HAIRLINE_CLASSES else None,
                "hairline_pred": None,
                "hairline_conf": None,
                "hairline_status": "error",
                "hairline_correct": False,

                "error": "cv2_imread_failed",
            })
            continue

        try:
            hair_mask, _ = segment_face(img)
            result = classify_hair(img, hair_mask)

        except Exception as e:
            rows.append({
                "filename": fname,

                "hair_true": true_ht if true_ht in HAIR_CLASSES else None,
                "hair_pred": None,
                "hair_conf": None,
                "hair_status": "error",
                "hair_correct": False,

                "hairline_true": true_hl if true_hl in HAIRLINE_CLASSES else None,
                "hairline_pred": None,
                "hairline_conf": None,
                "hairline_status": "error",
                "hairline_correct": False,

                "error": f"{type(e).__name__}: {str(e)}",
            })
            continue

        if true_ht in HAIR_CLASSES:

            pred_ht = result["hair_type"]
            conf_ht = result["hair_conf"]

            if pred_ht is None:
                hair_status = "abstain"
                hair_correct = False
            else:
                hair_status = "ok"
                hair_correct = pred_ht == true_ht

        else:
            pred_ht = None
            conf_ht = None
            hair_status = "not_labeled"
            hair_correct = False


        if true_hl in HAIRLINE_CLASSES:

            pred_hl = result["hairline"]
            conf_hl = result["hairline_conf"]

            if pred_hl is None:
                hairline_status = "abstain"
                hairline_correct = False
            else:
                hairline_status = "ok"
                hairline_correct = pred_hl == true_hl

        else:
            pred_hl = None
            conf_hl = None
            hairline_status = "not_labeled"
            hairline_correct = False

        rows.append({
            "filename": fname,

            "hair_true": true_ht if true_ht in HAIR_CLASSES else None,
            "hair_pred": pred_ht,
            "hair_conf": conf_ht,
            "hair_status": hair_status,
            "hair_correct": hair_correct,

            "hairline_true": true_hl if true_hl in HAIRLINE_CLASSES else None,
            "hairline_pred": pred_hl,
            "hairline_conf": conf_hl,
            "hairline_status": hairline_status,
            "hairline_correct": hairline_correct,

            "error": None,
        })

    return {
        "rows": rows,
    }


def print_report(res: dict, tag: str = ""):
    sep = "=" * 60

    if tag:
        print(f"\n{sep}\n  {tag}\n{sep}")

    df = pd.DataFrame(res["rows"])

    hair = df[df["hair_true"].isin(HAIR_CLASSES)].copy()

    if not hair.empty:

        total = len(hair)

        ok = hair[hair["hair_status"] == "ok"]
        abstain = hair[hair["hair_status"] == "abstain"]
        errors = hair[hair["hair_status"].isin(["error", "missing"])]

        coverage = len(ok) / total
        abstention_rate = len(abstain) / total
        error_rate = len(errors) / total

        print(f"\n── HAIR TYPE ({total} labeled samples) ──")

        print(f"  Predictions:       {len(ok)}/{total}")
        print(f"  Abstentions:       {len(abstain)}/{total} "
              f"({abstention_rate*100:.1f}%)")
        print(f"  Errors/missing:    {len(errors)}/{total} "
              f"({error_rate*100:.1f}%)")
        print(f"  Coverage:          {coverage*100:.1f}%")

        if not ok.empty:

            acc = accuracy_score(
                ok["hair_true"],
                ok["hair_pred"]
            )

            bal = balanced_accuracy_score(
                ok["hair_true"],
                ok["hair_pred"]
            )

            f1 = f1_score(
                ok["hair_true"],
                ok["hair_pred"],
                labels=HAIR_CLASSES,
                average="macro",
                zero_division=0
            )

            mean_conf = ok["hair_conf"].mean()

            hard = (ok["hair_conf"] < HARD_THRESHOLD).sum()

            print("\n  Metrics among predictions:")
            print(f"    Accuracy:          {acc:.3f}")
            print(f"    Balanced accuracy: {bal:.3f}")
            print(f"    Macro F1:          {f1:.3f}")
            print(f"    Mean confidence:   {mean_conf:.3f}")
            print(f"    Hard cases:        {hard}/{len(ok)} "
                  f"({hard/len(ok)*100:.1f}%)")

            print("\n  Classification report:")
            print(
                classification_report(
                    ok["hair_true"],
                    ok["hair_pred"],
                    labels=HAIR_CLASSES,
                    target_names=HAIR_CLASSES,
                    zero_division=0
                )
            )

            cm = confusion_matrix(
                ok["hair_true"],
                ok["hair_pred"],
                labels=HAIR_CLASSES
            )

            print("  Confusion matrix (rows=true, cols=pred):")
            print(f"  {'':10s} " +
                  " ".join(f"{c:>8s}" for c in HAIR_CLASSES))

            for i, row_label in enumerate(HAIR_CLASSES):
                print(
                    f"  {row_label:10s} " +
                    " ".join(
                        f"{cm[i,j]:8d}"
                        for j in range(len(HAIR_CLASSES))
                    )
                )

        end_to_end_acc = hair["hair_correct"].mean()

        print("\n  End-to-end:")
        print(f"    Accuracy including abstentions/errors: "
              f"{end_to_end_acc:.3f}")

    else:
        print("\n── HAIR TYPE: no labeled samples ──")

    hairline = df[
        df["hairline_true"].isin(HAIRLINE_CLASSES)
    ].copy()

    if not hairline.empty:

        total = len(hairline)

        ok = hairline[hairline["hairline_status"] == "ok"]
        abstain = hairline[
            hairline["hairline_status"] == "abstain"
        ]
        errors = hairline[
            hairline["hairline_status"].isin(["error", "missing"])
        ]

        coverage = len(ok) / total
        abstention_rate = len(abstain) / total
        error_rate = len(errors) / total

        print(f"\n── HAIRLINE ({total} labeled samples) ──")

        print(f"  Predictions:       {len(ok)}/{total}")
        print(f"  Abstentions:       {len(abstain)}/{total} "
              f"({abstention_rate*100:.1f}%)")
        print(f"  Errors/missing:    {len(errors)}/{total} "
              f"({error_rate*100:.1f}%)")
        print(f"  Coverage:          {coverage*100:.1f}%")

        if not ok.empty:

            acc = accuracy_score(
                ok["hairline_true"],
                ok["hairline_pred"]
            )

            bal = balanced_accuracy_score(
                ok["hairline_true"],
                ok["hairline_pred"]
            )

            f1 = f1_score(
                ok["hairline_true"],
                ok["hairline_pred"],
                labels=HAIRLINE_CLASSES,
                average="macro",
                zero_division=0
            )

            mean_conf = ok["hairline_conf"].mean()

            hard = (
                ok["hairline_conf"] < HARD_THRESHOLD
            ).sum()

            print("\n  Metrics among predictions:")
            print(f"    Accuracy:          {acc:.3f}")
            print(f"    Balanced accuracy: {bal:.3f}")
            print(f"    Macro F1:          {f1:.3f}")
            print(f"    Mean confidence:   {mean_conf:.3f}")
            print(f"    Hard cases:        {hard}/{len(ok)} "
                  f"({hard/len(ok)*100:.1f}%)")

            print("\n  Classification report:")
            print(
                classification_report(
                    ok["hairline_true"],
                    ok["hairline_pred"],
                    labels=HAIRLINE_CLASSES,
                    target_names=HAIRLINE_CLASSES,
                    zero_division=0
                )
            )

            cm = confusion_matrix(
                ok["hairline_true"],
                ok["hairline_pred"],
                labels=HAIRLINE_CLASSES
            )

            print("Confusion matrix (rows=true, cols=pred):")
            print(f"  {'':10s} " +
                  " ".join(f"{c:>8s}" for c in HAIRLINE_CLASSES))

            for i, row_label in enumerate(HAIRLINE_CLASSES):
                print(
                    f"  {row_label:10s} " +
                    " ".join(
                        f"{cm[i,j]:8d}"
                        for j in range(len(HAIRLINE_CLASSES))
                    )
                )

        end_to_end_acc = hairline["hairline_correct"].mean()

        print("\n  End-to-end:")
        print(f"Accuracy including abstentions/errors: "
              f"{end_to_end_acc:.3f}")

    else:
        print("\n── HAIRLINE: no labeled samples ──")

    print("\n── PIPELINE ──")

    status_counts = df["error"].notna().sum()

    print(f"  Test images evaluated: {len(df)}")
    print(f"  Images with pipeline errors: {status_counts}")

    if status_counts:
        print("\n  Error types:")

        error_types = (
            df.loc[df["error"].notna(), "error"]
            .str.split(":")
            .str[0]
            .value_counts()
        )

        for error_type, count in error_types.items():
            print(f"{error_type}: {count}")


def save_results(res: dict, out_path: str):
    """
    Save one row per test image.

    This intentionally keeps abstentions and errors so that
    before/after evaluations can be compared on the same test set.
    """

    df = pd.DataFrame(res["rows"])

    for col in ["hair_conf", "hairline_conf"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    df.to_csv(out_path, index=False)

    print(f"\nPer-image results saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="Labeled CSV (filename, hair_type, hairline)")
    ap.add_argument("--images", default=DEFAULT_IMAGES,
                    help="Directory with face images")
    ap.add_argument("--tag", default="CURRENT MODEL",
                    help="Label printed in report header")
    ap.add_argument("--out", default="",
                    help="Optional path to save per-image CSV results")
    ap.add_argument("--fraction", type=float, default=TEST_FRACTION,
                    help=f"Fraction used as test set (default {TEST_FRACTION})")
    ap.add_argument("--fixed-test", default="",
                    help="Use an existing fixed test CSV instead of sampling a new test set"
)
    args = ap.parse_args()

    df = load_test_set(
        args.csv,
        args.fraction,
        TEST_SEED,
        args.fixed_test,
    )

    res = run_evaluation(df, args.images)

    print_report(res, tag=args.tag)

    if args.out:
        save_results(res, args.out)
    else:
        default_out = (
            f"dataset/hair_dataset/"
            f"eval_{args.tag.lower().replace(' ', '_')}.csv"
        )
        save_results(res, default_out)