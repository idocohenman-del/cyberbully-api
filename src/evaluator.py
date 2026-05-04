"""
evaluator.py
────────────
Evaluates a trained pipeline on a held-out split.
Prints a full classification report and saves a confusion matrix + ROC curve.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score,
)


def evaluate(
    pipeline,
    X_test,
    y_test,
    save_dir: str = "models",
    split_name: str = "Test",
) -> dict:
    """
    Run the pipeline on X_test, print metrics, and save plots.

    Parameters
    ----------
    split_name : label shown in the console header and plot titles
                 (e.g. "Validation" or "Test")

    Returns
    -------
    dict with numeric metric values for programmatic use.
    """
    y_pred = pipeline.predict(X_test)

    # ROC-AUC requires probability scores; LinearSVC without calibration lacks them
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
    except AttributeError:
        y_prob = None
        auc    = None

    print("\n" + "═" * 55)
    print(f"  {split_name.upper()} EVALUATION RESULTS")
    print("═" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Bullying (1)"],
        digits=4,
    ))
    if auc is not None:
        print(f"  ROC-AUC Score : {auc:.4f}")
    print("═" * 55 + "\n")

    _plot_confusion_matrix(y_test, y_pred, save_dir, split_name)
    if y_prob is not None:
        _plot_roc_curve(y_test, y_prob, auc, save_dir, split_name)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "roc_auc":   auc,
    }


def _plot_confusion_matrix(y_test, y_pred, save_dir: str, split_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Safe", "Bullying"],
        yticklabels=["Safe", "Bullying"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name}")
    fig.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{split_name.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


def _plot_roc_curve(y_test, y_prob, auc: float, save_dir: str, split_name: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(save_dir, f"roc_curve_{split_name.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved → {path}")
