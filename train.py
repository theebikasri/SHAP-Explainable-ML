"""
Stage 3 — top-level training runner.

Trains all four models (LR + XGBoost for healthcare and finance),
saves artifacts and writes comparison reports.

Run from project root with the venv active:
    .venv\\Scripts\\python.exe train.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.train_healthcare import run as run_healthcare
from src.train_finance import run as run_finance
from src.utils import REPORTS_DIR, ensure_project_dirs


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_comparison_csv(all_metrics: list[dict]) -> Path:
    path = REPORTS_DIR / "model_comparison.csv"
    fields = ["domain", "label", "accuracy", "precision", "recall",
              "f1", "roc_auc", "tn", "fp", "fn", "tp",
              "cv_roc_auc_mean", "cv_roc_auc_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    return path


def save_metrics_json(all_metrics: list[dict]) -> Path:
    path = REPORTS_DIR / "model_metrics.json"
    # classification_report is verbose — keep it but nest it
    path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return path


def save_training_summary(all_metrics: list[dict]) -> Path:
    lines = [
        "Stage 3 — Training Summary",
        "==========================",
        f"Generated: {_timestamp()}",
        "",
    ]
    for domain in ("healthcare", "finance"):
        domain_metrics = [m for m in all_metrics if m["domain"] == domain]
        lines.append(f"{domain.capitalize()} results (held-out test set)")
        lines.append("-" * 45)
        header = f"  {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}  CV-AUC"
        lines.append(header)
        for m in domain_metrics:
            lines.append(
                f"  {m['label']:<25} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
                f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f}  "
                f"{m['cv_roc_auc_mean']:.4f}±{m['cv_roc_auc_std']:.4f}"
            )
        lines.append("")
        for m in domain_metrics:
            lines.append(f"  {m['label']} confusion matrix  "
                         f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
        lines.append("")
        for m in domain_metrics:
            lines.append(f"  {m['label']} best params: {m['best_params']}")
        lines.append("")

    path = REPORTS_DIR / "training_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_project_dirs()
    print("=" * 60)
    print("Stage 3 — Model Training")
    print("=" * 60)

    hc_results  = run_healthcare()
    fin_results = run_finance()
    all_metrics = hc_results + fin_results

    csv_path  = save_comparison_csv(all_metrics)
    json_path = save_metrics_json(all_metrics)
    txt_path  = save_training_summary(all_metrics)

    print()
    print("Artifacts saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print()
    print("Training complete. Run validate_stage3.py to verify all artifacts.")


if __name__ == "__main__":
    main()
