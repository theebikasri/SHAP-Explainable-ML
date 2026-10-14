"""
Stage 2 — dataset validation for healthcare (breast cancer) and finance (credit risk).

Reads raw CSVs only (no overwriting). Writes text reports under outputs/reports/.
Run: python -m src.data_validation   (from project root, with .venv active)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    BRCA_CSV,
    CREDIT_RISK_CSV,
    REPORTS_DIR,
    ensure_project_dirs,
    require_datasets,
)


def _report_header(title: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{title}\n{'=' * len(title)}\nGenerated: {ts}\n\n"


def validate_healthcare_brca() -> dict:
    """
    Validate dataset/brca.csv per Stage 1 audit.

    Expectations: 569 rows; target column 'y' with B=357, M=212;
    leading unnamed index column; features named x.* for SHAP readability.
    """
    df = pd.read_csv(BRCA_CSV)
    n_rows = len(df)
    unnamed_cols = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]

    if "y" not in df.columns:
        raise ValueError("brca.csv must contain column 'y'.")

    y_counts = df["y"].value_counts().to_dict()
    n_b = int(y_counts.get("B", 0))
    n_m = int(y_counts.get("M", 0))

    # File is class-sorted in the raw CSV — check whether rows are grouped by label
    y_series = df["y"].astype(str)
    block_changes = int((y_series != y_series.shift()).sum()) - 1
    class_sorted = block_changes <= 2  # typically one block of B then one of M

    feature_cols = [c for c in df.columns if c not in unnamed_cols and c != "y"]
    missing = df[feature_cols + ["y"]].isna().sum().sum()

    stats = {
        "path": str(BRCA_CSV),
        "n_rows": n_rows,
        "n_features_raw": len(feature_cols),
        "unnamed_index_columns": unnamed_cols,
        "y_B": n_b,
        "y_M": n_m,
        "class_sorted_in_file": class_sorted,
        "label_block_transitions": block_changes,
        "total_missing_values": int(missing),
        "expected_rows": 569,
        "expected_B": 357,
        "expected_M": 212,
    }
    return stats


def validate_finance_credit_risk() -> dict:
    """
    Validate dataset/credit_risk_dataset.csv per Stage 1 audit.

    Expectations: 32581 rows; loan_status with 0=25473, 1=7108;
    person_emp_length and loan_int_rate may contain missing values (handled in pipeline).
    """
    df = pd.read_csv(CREDIT_RISK_CSV)
    n_rows = len(df)

    if "loan_status" not in df.columns:
        raise ValueError("credit_risk_dataset.csv must contain column 'loan_status'.")

    status_counts = df["loan_status"].value_counts().sort_index().to_dict()
    n0 = int(status_counts.get(0, 0))
    n1 = int(status_counts.get(1, 0))

    # Pre-split data quality flags (documented before train/test split in preprocessing)
    age_over_100 = int((df["person_age"] > 100).sum())
    emp_gt_age = int((df["person_emp_length"] > df["person_age"]).sum())

    miss_emp = int(df["person_emp_length"].isna().sum())
    miss_rate = int(df["loan_int_rate"].isna().sum())

    required_keep = ["loan_grade", "loan_int_rate"]
    missing_required = [c for c in required_keep if c not in df.columns]

    stats = {
        "path": str(CREDIT_RISK_CSV),
        "n_rows": n_rows,
        "loan_status_0": n0,
        "loan_status_1": n1,
        "missing_person_emp_length": miss_emp,
        "missing_loan_int_rate": miss_rate,
        "rows_person_age_gt_100": age_over_100,
        "rows_person_emp_length_gt_person_age": emp_gt_age,
        "columns": list(df.columns),
        "missing_required_columns": missing_required,
        "expected_rows": 32581,
        "expected_status_0": 25473,
        "expected_status_1": 7108,
    }
    return stats


def _format_healthcare_report(stats: dict) -> str:
    lines = [
        _report_header("Healthcare — Breast Cancer (brca.csv) Validation"),
        f"File: {stats['path']}\n",
        f"Row count: {stats['n_rows']} (expected {stats['expected_rows']})",
        f"Target 'y': B={stats['y_B']} (expected {stats['expected_B']}), "
        f"M={stats['y_M']} (expected {stats['expected_M']})",
        f"Unnamed index column(s) to drop: {stats['unnamed_index_columns']}",
        f"Feature columns (excluding index and y): {stats['n_features_raw']}",
        f"Missing values (features + y): {stats['total_missing_values']}",
        f"Class-sorted in raw file: {stats['class_sorted_in_file']} "
        f"(label block transitions: {stats['label_block_transitions']})",
        "",
        "Encoding plan (preprocessing): M -> 1, B -> 0.",
        "Split plan: stratified train/test with shuffle=True (required because file is sorted).",
    ]
    ok = (
        stats["n_rows"] == stats["expected_rows"]
        and stats["y_B"] == stats["expected_B"]
        and stats["y_M"] == stats["expected_M"]
    )
    lines.append(f"\nAudit check: {'PASS' if ok else 'REVIEW — counts differ from Stage 1 audit'}")
    return "\n".join(lines)


def _format_finance_report(stats: dict) -> str:
    lines = [
        _report_header("Finance — Credit Risk Validation"),
        f"File: {stats['path']}\n",
        f"Row count: {stats['n_rows']} (expected {stats['expected_rows']})",
        f"Target loan_status: 0={stats['loan_status_0']} (expected {stats['expected_status_0']}), "
        f"1={stats['loan_status_1']} (expected {stats['expected_status_1']})",
        f"Missing person_emp_length: {stats['missing_person_emp_length']} "
        "(imputed in train-only finance pipeline)",
        f"Missing loan_int_rate: {stats['missing_loan_int_rate']} "
        "(imputed in train-only finance pipeline)",
        f"Rows with person_age > 100 (before split): {stats['rows_person_age_gt_100']}",
        f"Rows with person_emp_length > person_age (before split): "
        f"{stats['rows_person_emp_length_gt_person_age']}",
        "",
        "Pre-split cleaning (preprocessing):",
        "  - person_age > 100: cap at 100 (preserves rows; documented in preprocessing report).",
        "  - person_emp_length > person_age: drop those rows (logical inconsistency).",
        "",
        f"Required columns present (loan_grade, loan_int_rate): "
        f"{len(stats['missing_required_columns']) == 0}",
        "Categorical encoding plan: OneHotEncoder(handle_unknown='ignore').",
    ]
    ok = (
        stats["n_rows"] == stats["expected_rows"]
        and stats["loan_status_0"] == stats["expected_status_0"]
        and stats["loan_status_1"] == stats["expected_status_1"]
        and not stats["missing_required_columns"]
    )
    lines.append(f"\nAudit check: {'PASS' if ok else 'REVIEW — counts differ from Stage 1 audit'}")
    return "\n".join(lines)


def write_validation_reports(
    healthcare_stats: dict,
    finance_stats: dict,
) -> tuple[Path, Path]:
    ensure_project_dirs()
    hc_path = REPORTS_DIR / "healthcare_validation.txt"
    fin_path = REPORTS_DIR / "finance_validation.txt"
    hc_path.write_text(_format_healthcare_report(healthcare_stats), encoding="utf-8")
    fin_path.write_text(_format_finance_report(finance_stats), encoding="utf-8")
    return hc_path, fin_path


def main() -> None:
    require_datasets()
    ensure_project_dirs()
    hc = validate_healthcare_brca()
    fin = validate_finance_credit_risk()
    hc_path, fin_path = write_validation_reports(hc, fin)
    print(_format_healthcare_report(hc))
    print("\n" + "-" * 60 + "\n")
    print(_format_finance_report(fin))
    print(f"\nReports saved:\n  {hc_path}\n  {fin_path}")


if __name__ == "__main__":
    main()
