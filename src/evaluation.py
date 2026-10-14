"""
Stage 3 — evaluation helpers shared by both domains.

Computes metrics from real test-set predictions only.
Saves confusion matrix, ROC curve and PR curve figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import FIGURES_DIR, ensure_project_dirs


def compute_metrics(y_true, y_pred, y_prob, label: str) -> dict:
    """Return a flat metrics dict from held-out test predictions."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "label":     label,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["class_0", "class_1"]
        ),
    }


def save_confusion_matrix(y_true, y_pred, label: str, domain: str) -> Path:
    ensure_project_dirs()
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"{domain} — {label}\nConfusion Matrix")
    fig.tight_layout()
    path = FIGURES_DIR / f"{domain.lower()}_{label.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_roc_curve(results: list[dict], domain: str, y_tests: list, y_probs: list) -> Path:
    """Overlay ROC curves for all models in one domain on a single figure."""
    ensure_project_dirs()
    fig, ax = plt.subplots(figsize=(6, 5))
    for res, y_true, y_prob in zip(results, y_tests, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{res['label']} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{domain} — ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / f"{domain.lower()}_roc_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_pr_curve(results: list[dict], domain: str, y_tests: list, y_probs: list) -> Path:
    """Overlay Precision-Recall curves for all models in one domain."""
    ensure_project_dirs()
    fig, ax = plt.subplots(figsize=(6, 5))
    for res, y_true, y_prob in zip(results, y_tests, y_probs):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{res['label']} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{domain} — Precision-Recall Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = FIGURES_DIR / f"{domain.lower()}_pr_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
