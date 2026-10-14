"""
Shared paths and small helpers for the Explainable ML project (Stage 2).

All paths are resolved relative to the project root so scripts work from
any working directory on Windows PowerShell.
"""

from __future__ import annotations

from pathlib import Path

# Project layout (E:\\SHAP-Explainable-ML)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

BRCA_CSV = DATASET_DIR / "brca.csv"
CREDIT_RISK_CSV = DATASET_DIR / "credit_risk_dataset.csv"

# Reproducible splits and preprocessing (Stage 2)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Filenames for saved artifacts (domains kept separate)
HEALTHCARE_PIPELINE_PATH = MODELS_DIR / "healthcare_preprocessor.joblib"
FINANCE_PIPELINE_PATH    = MODELS_DIR / "finance_preprocessor.joblib"
HEALTHCARE_SPLIT_PATH   = MODELS_DIR / "healthcare_train_test_split.joblib"
FINANCE_SPLIT_PATH       = MODELS_DIR / "finance_train_test_split.joblib"

# Stage 3 — trained model artifacts
HC_LR_PATH  = MODELS_DIR / "healthcare_logistic_regression.joblib"
HC_XGB_PATH = MODELS_DIR / "healthcare_xgboost.joblib"
FIN_LR_PATH  = MODELS_DIR / "finance_logistic_regression.joblib"
FIN_XGB_PATH = MODELS_DIR / "finance_xgboost.joblib"


def ensure_project_dirs() -> None:
    """Create output and model directories if they do not exist."""
    for path in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_datasets() -> None:
    """
    Raise FileNotFoundError if either required CSV is missing.
    Stage 2 must stop rather than substitute data.
    """
    missing = [p for p in (BRCA_CSV, CREDIT_RISK_CSV) if not p.is_file()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required dataset(s) not found: {names}. "
            "Place files under dataset/ and re-run validation."
        )
