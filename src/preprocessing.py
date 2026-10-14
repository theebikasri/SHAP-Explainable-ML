"""
Stage 2 — domain-specific preprocessing pipelines (no model training).

Healthcare and finance are handled separately. All transformers are fit on
training data only after a stratified shuffle split.

Run: python -m src.preprocessing
"""

from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    BRCA_CSV,
    CREDIT_RISK_CSV,
    FINANCE_PIPELINE_PATH,
    FINANCE_SPLIT_PATH,
    HEALTHCARE_PIPELINE_PATH,
    HEALTHCARE_SPLIT_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    ensure_project_dirs,
    require_datasets,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_healthcare_frame() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load BRCA data: drop index column, encode target, keep x.* feature names."""
    df = pd.read_csv(BRCA_CSV)
    drop_cols = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "y" not in df.columns:
        raise ValueError("Column 'y' missing in brca.csv")

    y = df["y"].map({"M": 1, "B": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), "y"].unique()
        raise ValueError(f"Unexpected y labels (expected B/M only): {bad}")

    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].copy()
    return X, y.astype(int), feature_cols


def build_healthcare_preprocessor() -> Pipeline:
    """Numeric-only breast cancer features — scale for downstream linear models."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_healthcare() -> dict:
    """Stratified split, fit scaler on train, save pipeline and split bundle."""
    X, y, feature_cols = load_healthcare_frame()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )

    pipeline = build_healthcare_preprocessor()
    pipeline.fit(X_train)

    X_train_t = pipeline.transform(X_train)
    X_test_t = pipeline.transform(X_test)

    ensure_project_dirs()
    joblib.dump(pipeline, HEALTHCARE_PIPELINE_PATH)
    joblib.dump(
        {
            "domain": "healthcare",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_processed": X_train_t,
            "X_test_processed": X_test_t,
            "feature_names": feature_cols,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        HEALTHCARE_SPLIT_PATH,
    )

    return {
        "n_rows": len(X),
        "n_features": len(feature_cols),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "processed_train_shape": tuple(np.shape(X_train_t)),
        "processed_test_shape": tuple(np.shape(X_test_t)),
        "pipeline_path": str(HEALTHCARE_PIPELINE_PATH),
        "split_path": str(HEALTHCARE_SPLIT_PATH),
    }


def load_finance_frame() -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load credit risk data and apply documented cleaning BEFORE the train/test split.

    Step 1 — Cap person_age > 100 at 100 (preserves rows; 5 affected).
    Step 2 — Drop rows where person_emp_length > person_age (impossible; 2 rows).
    Step 3 — Drop exact full-row duplicates (inflated training signal; 165 rows).

    All three steps are applied before the stratified split so no leakage occurs.
    Row counts are recorded after each step for the preprocessing report.
    """
    df = pd.read_csv(CREDIT_RISK_CSV)
    n_initial = len(df)

    # Step 1: cap implausible ages (keep rows, correct the value)
    age_over_100_mask = df["person_age"] > 100
    n_age_capped = int(age_over_100_mask.sum())
    df.loc[age_over_100_mask, "person_age"] = 100
    n_after_age_cap = len(df)  # unchanged — rows kept, value corrected

    # Step 2: drop rows with logically impossible employment length
    emp_gt_age_mask = df["person_emp_length"] > df["person_age"]
    n_emp_dropped = int(emp_gt_age_mask.sum())
    df = df.loc[~emp_gt_age_mask].reset_index(drop=True)
    n_after_emp_drop = len(df)

    # Step 3: drop exact full-row duplicates
    n_before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicates_dropped = n_before_dedup - len(df)
    n_after_dedup = len(df)

    y = df["loan_status"].astype(int)
    X = df.drop(columns=["loan_status"])

    cleaning = {
        "n_initial_rows": n_initial,
        "n_age_capped_at_100": n_age_capped,
        "n_after_age_cap": n_after_age_cap,
        "n_rows_dropped_emp_length_gt_age": n_emp_dropped,
        "n_after_emp_drop": n_after_emp_drop,
        "n_duplicates_dropped": n_duplicates_dropped,
        "n_after_dedup": n_after_dedup,
        "n_rows_after_cleaning": n_after_dedup,
    }
    return X, y, cleaning


def build_finance_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Train-only imputation for person_emp_length and loan_int_rate (via numeric pipeline
    median imputer). Categoricals: OneHot with unknown levels ignored at scoring time.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
    )


def preprocess_finance() -> dict:
    """Stratified split after pre-split cleaning; fit finance pipeline on train only."""
    X, y, cleaning = load_finance_frame()

    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    for col in numeric_features + categorical_features:
        if col not in X.columns:
            raise ValueError(f"Expected column missing in credit dataset: {col}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )

    pipeline = build_finance_preprocessor(numeric_features, categorical_features)
    pipeline.fit(X_train)

    X_train_t = pipeline.transform(X_train)
    X_test_t = pipeline.transform(X_test)

    # Feature names after one-hot (for later SHAP interpretability)
    try:
        transformed_names = pipeline.get_feature_names_out().tolist()
    except AttributeError:
        transformed_names = None

    ensure_project_dirs()
    joblib.dump(pipeline, FINANCE_PIPELINE_PATH)
    joblib.dump(
        {
            "domain": "finance",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_processed": X_train_t,
            "X_test_processed": X_test_t,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "transformed_feature_names": transformed_names,
            "cleaning": cleaning,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        FINANCE_SPLIT_PATH,
    )

    return {
        **cleaning,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
        "processed_train_shape": tuple(np.shape(X_train_t)),
        "processed_test_shape": tuple(np.shape(X_test_t)),
        "n_transformed_features": int(X_train_t.shape[1]),
        "pipeline_path": str(FINANCE_PIPELINE_PATH),
        "split_path": str(FINANCE_SPLIT_PATH),
    }


def write_preprocessing_report(hc: dict, fin: dict) -> None:
    """Write combined preprocessing summary with real run statistics."""
    lines = [
        "Stage 2 — Preprocessing Summary",
        "================================",
        f"Generated: {_timestamp()}",
        "",
        "No classifier or SHAP steps were run in this stage.",
        "",
        "Healthcare (breast cancer)",
        "---------------------------",
        f"Rows: {hc['n_rows']} | Features: {hc['n_features']}",
        f"Train/test: {hc['train_rows']} / {hc['test_rows']} (stratified, shuffle=True)",
        f"Malignant rate train/test: {hc['train_pos_rate']:.4f} / {hc['test_pos_rate']:.4f}",
        f"Processed shapes train/test: {hc['processed_train_shape']} / {hc['processed_test_shape']}",
        f"Pipeline: {hc['pipeline_path']}",
        f"Split bundle: {hc['split_path']}",
        "",
        "Finance (credit risk)",
        "---------------------",
        f"Rows before cleaning: {fin['n_initial_rows']}",
        f"Step 1 — Age > 100 capped at 100: {fin['n_age_capped_at_100']} rows corrected | rows remaining: {fin['n_after_age_cap']}",
        f"Step 2 — Dropped (emp_length > age): {fin['n_rows_dropped_emp_length_gt_age']} rows | rows remaining: {fin['n_after_emp_drop']}",
        f"Step 3 — Dropped exact duplicates: {fin['n_duplicates_dropped']} rows | rows remaining: {fin['n_after_dedup']}",
        f"Rows after all cleaning: {fin['n_rows_after_cleaning']}",
        f"Train/test: {fin['train_rows']} / {fin['test_rows']} (stratified, shuffle=True)",
        f"Default rate train/test: {fin['train_default_rate']:.4f} / {fin['test_default_rate']:.4f}",
        f"Processed shapes train/test: {fin['processed_train_shape']} / {fin['processed_test_shape']}",
        f"Transformed feature count: {fin['n_transformed_features']}",
        f"Pipeline: {fin['pipeline_path']}",
        f"Split bundle: {fin['split_path']}",
    ]
    path = REPORTS_DIR / "preprocessing_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved: {path}")


def main() -> None:
    require_datasets()
    ensure_project_dirs()
    hc_stats = preprocess_healthcare()
    fin_stats = preprocess_finance()
    write_preprocessing_report(hc_stats, fin_stats)


if __name__ == "__main__":
    main()
