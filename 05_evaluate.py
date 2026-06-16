"""
05_evaluate.py — evaluation curves + diagnostics.

Outputs:
  - ROC curve
  - PR curve
  - Confusion matrix at threshold = 0.5
  - Calibration plot (binned predicted prob vs observed rate)
  - Threshold sweep table for picking operating point
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score,
)

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = PROC / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    pred = pd.read_csv(PROC / "test_predictions.csv")
    y = pred["actual"].values
    p = pred["prob"].values

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curve")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "01_roc.png", dpi=120); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}", color="darkorange", linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "02_pr.png", dpi=120); plt.close()

    # Calibration
    bins = np.linspace(0, 1, 11)
    pred["bin"] = pd.cut(pred["prob"], bins=bins, include_lowest=True)
    cal = pred.groupby("bin", observed=True).agg(
        avg_pred=("prob", "mean"),
        actual_rate=("actual", "mean"),
        n=("actual", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.scatter(cal["avg_pred"], cal["actual_rate"], s=cal["n"]/30, alpha=0.7, color="seagreen")
    ax.set_xlabel("Predicted prob (bin avg)")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration — bubble size = bin count")
    plt.tight_layout(); plt.savefig(OUT / "03_calibration.png", dpi=120); plt.close()

    cal.to_csv(OUT / "calibration.csv", index=False)

    # Threshold sweep
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        rows.append({
            "threshold": round(t, 2),
            "predicted_positive": int(tp + fp),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)

    print(f"AUC: {auc:.4f}   AP: {ap:.4f}")
    print("\nthreshold sweep (head):")
    print(sweep.head(10).to_string(index=False))
    print(f"\nplots saved to {OUT}")


if __name__ == "__main__":
    main()
