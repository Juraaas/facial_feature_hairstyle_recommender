import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

HARD_THRESHOLD = 0.70
HAIR_CLASSES = ["straight", "wavy", "curly", "coily"]
HAIRLINE_CLASSES = ["normal", "receding", "uneven"]

def metrics_from_df(df: pd.DataFrame, head: str) -> dict:
    if head == "hair_type":
        true_col  = "hair_true"
        pred_col  = "hair_pred"
        conf_col  = "hair_conf"
        status_col = "hair_status"
        classes   = HAIR_CLASSES
    else:
        true_col  = "hairline_true"
        pred_col  = "hairline_pred"
        conf_col  = "hairline_conf"
        status_col = "hairline_status"
        classes   = HAIRLINE_CLASSES

    labeled = df[df[true_col].isin(classes)].copy()
    if labeled.empty:
        return {}

    ok = labeled[labeled[status_col] == "ok"].copy()
    if ok.empty:
        return {}

    t = ok[true_col].tolist()
    p = ok[pred_col].tolist()
    c = ok[conf_col].tolist()
    hard = int((ok[conf_col] < HARD_THRESHOLD).sum())
    n_labeled = len(labeled)
    n_ok      = len(ok)

    return {
        "n_labeled":   n_labeled,
        "n_ok":        n_ok,
        "abstentions": n_labeled - n_ok,
        "acc":         accuracy_score(t, p),
        "bal_acc":     balanced_accuracy_score(t, p),
        "macro_f1":    f1_score(t, p, average="macro",
                                labels=classes, zero_division=0),
        "hard":        hard,
        "hard_pct":    hard / n_ok * 100,
        "mean_conf":   float(np.mean(c)),
        "end2end_acc": float(labeled["hair_correct" if head == "hair_type"
                                     else "hairline_correct"].mean()),
        "per_class_f1": {
            cls: f1_score(
                [x == cls for x in t],
                [x == cls for x in p],
                average="binary", zero_division=0,
            )
            for cls in classes
        },
    }


def fmt_delta(val, higher_is_better=True):
    sign  = "+" if val >= 0 else ""
    arrow = ""
    if val > 0.001:
        arrow = " ↑" if higher_is_better else " ↓"
    elif val < -0.001:
        arrow = " ↓" if higher_is_better else " ↑"
    return f"{sign}{val:.3f}{arrow}"


def compare(before_csv: str, after_csv: str):
    b = pd.read_csv(before_csv)
    a = pd.read_csv(after_csv)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  BEFORE : {before_csv}  ({len(b)} rows)")
    print(f"  AFTER  : {after_csv}  ({len(a)} rows)")
    print(f"{sep}")

    for head, classes in [
        ("hair_type", HAIR_CLASSES),
        ("hairline",  HAIRLINE_CLASSES),
    ]:
        mb = metrics_from_df(b, head)
        ma = metrics_from_df(a, head)

        if not mb or not ma:
            print(f"\n── {head.upper()}: missing data in one of the files")
            continue

        print(f"\n── {head.upper()} ──")
        print(f"  Samples  : {mb['n_labeled']} → {ma['n_labeled']}")
        print(f"  Predicted: {mb['n_ok']}/{mb['n_labeled']} "
              f"→ {ma['n_ok']}/{ma['n_labeled']}")
        abstb = mb['abstentions'] / mb['n_labeled'] * 100
        absta = ma['abstentions'] / ma['n_labeled'] * 100
        print(f"  Abstentions: {abstb:.1f}% → {absta:.1f}%  "
              f"({fmt_delta(absta - abstb, higher_is_better=False)})")

        print(f"\n  {'Metric':<24s}  {'Before':>8s}  {'After':>8s}  {'Delta':>12s}")
        print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*12}")

        rows = [
            ("Accuracy",          mb["acc"],       ma["acc"],       True ),
            ("Balanced accuracy", mb["bal_acc"],   ma["bal_acc"],   True ),
            ("Macro F1",          mb["macro_f1"],  ma["macro_f1"],  True ),
            ("Mean confidence",   mb["mean_conf"], ma["mean_conf"], True ),
            ("Hard-case %",       mb["hard_pct"],  ma["hard_pct"],  False),
            ("End-to-end acc",    mb["end2end_acc"],ma["end2end_acc"],True),
        ]
        for label, bv, av, hib in rows:
            d = av - bv
            print(f"  {label:<24s}  {bv:>8.3f}  {av:>8.3f}  "
                  f"{fmt_delta(d, hib):>12s}")

        print(f"\n  Per-class F1:")
        print(f"  {'Class':<12s}  {'Before':>8s}  {'After':>8s}  {'Delta':>12s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*12}")
        for cls in classes:
            bf = mb["per_class_f1"].get(cls, 0)
            af = ma["per_class_f1"].get(cls, 0)
            d  = af - bf
            flag = "  ← significant" if abs(d) >= 0.05 else ""
            print(f"  {cls:<12s}  {bf:>8.3f}  {af:>8.3f}  "
                  f"{fmt_delta(d):>12s}{flag}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python util/compare_models.py before.csv after.csv")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])