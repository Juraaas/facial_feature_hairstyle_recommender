import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    balanced_accuracy_score,
)

HARD_THRESHOLD = 0.70
HAIR_CLASSES     = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]


def metrics(df: pd.DataFrame, head: str, classes: list) -> dict:
    sub = df[df["head"] == head].copy()
    if sub.empty:
        return {}
    t = sub["true"].tolist()
    p = sub["pred"].tolist()
    c = sub["conf"].tolist()
    hard = sub["hard_case"].sum()
    n    = len(sub)
    return {
        "n":        n,
        "acc":      accuracy_score(t, p),
        "bal_acc":  balanced_accuracy_score(t, p),
        "macro_f1": f1_score(t, p, average="macro",
                             labels=classes, zero_division=0),
        "hard":     int(hard),
        "hard_pct": hard / n * 100,
        "mean_conf":np.mean(c),
        "per_class_f1": {
            cls: f1_score(
                [x == cls for x in t],
                [x == cls for x in p],
                average="binary", zero_division=0
            )
            for cls in classes
        },
    }


def fmt_delta(val, unit="", higher_is_better=True):
    sign  = "+" if val >= 0 else ""
    color = "↑" if (val > 0) == higher_is_better else ("↓" if val != 0 else "")
    return f"{sign}{val:.3f}{unit}  {color}"


def compare(before_csv: str, after_csv: str):
    b = pd.read_csv(before_csv)
    a = pd.read_csv(after_csv)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BEFORE: {before_csv}")
    print(f"  AFTER:  {after_csv}")
    print(f"{sep}")

    for head, classes in [
        ("hair_type", HAIR_CLASSES),
        ("hairline",  HAIRLINE_CLASSES),
    ]:
        mb = metrics(b, head, classes)
        ma = metrics(a, head, classes)

        if not mb or not ma:
            print(f"\n── {head.upper()}: missing data in one of the files")
            continue

        print(f"\n── {head.upper()} ──")
        print(f"  {'Metric':<22s}  {'Before':>8s}  {'After':>8s}  {'Delta':>14s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*14}")

        rows = [
            ("Accuracy",         mb["acc"],       ma["acc"],       True),
            ("Balanced accuracy",mb["bal_acc"],   ma["bal_acc"],   True),
            ("Macro F1",         mb["macro_f1"],  ma["macro_f1"],  True),
            ("Mean confidence",  mb["mean_conf"], ma["mean_conf"], True),
            ("Hard-case %",      mb["hard_pct"],  ma["hard_pct"],  False),
        ]
        for label, bval, aval, hib in rows:
            delta = aval - bval
            print(f"  {label:<22s}  {bval:>8.3f}  {aval:>8.3f}  "
                  f"{fmt_delta(delta, higher_is_better=hib):>14s}")

        print(f"\n  Hard cases: {mb['hard']}/{mb['n']} → "
              f"{ma['hard']}/{ma['n']}  "
              f"(delta: {ma['hard'] - mb['hard']:+d})")

        print(f"\n  Per-class F1:")
        print(f"  {'Class':<12s}  {'Before':>8s}  {'After':>8s}  {'Delta':>14s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*14}")
        for cls in classes:
            bf = mb["per_class_f1"].get(cls, 0)
            af = ma["per_class_f1"].get(cls, 0)
            d  = af - bf
            flag = ""
            if abs(d) >= 0.05:
                flag = "  ← significant"
            print(f"  {cls:<12s}  {bf:>8.3f}  {af:>8.3f}  "
                  f"{fmt_delta(d):>14s}{flag}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python util/compare_models.py before.csv after.csv")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])