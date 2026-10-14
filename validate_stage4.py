"""
Stage 4 validation script.
Run: .venv/Scripts/python.exe validate_stage4.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import shap

ROOT = Path(__file__).parent
errors = []

def chk(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        errors.append(label)


# ── Load artifacts ────────────────────────────────────────────────────────────
hc_bundle  = joblib.load(ROOT / "models/healthcare_train_test_split.joblib")
fin_bundle = joblib.load(ROOT / "models/finance_train_test_split.joblib")
hc_xgb     = joblib.load(ROOT / "models/healthcare_xgboost.joblib")
fin_xgb    = joblib.load(ROOT / "models/finance_xgboost.joblib")
hc_lr      = joblib.load(ROOT / "models/healthcare_logistic_regression.joblib")
fin_lr     = joblib.load(ROOT / "models/finance_logistic_regression.joblib")

chk(hc_bundle["domain"]  == "healthcare", "HC bundle domain key")
chk(fin_bundle["domain"] == "finance",    "FIN bundle domain key")

# ── Splits unchanged from Stage 2 ────────────────────────────────────────────
chk(hc_bundle["X_test_processed"].shape  == (114, 30),   "HC test shape unchanged")
chk(fin_bundle["X_test_processed"].shape == (6483, 26),  "FIN test shape unchanged")

# ── Feature name counts ───────────────────────────────────────────────────────
chk(len(hc_bundle["feature_names"])                == 30, "HC feature_names count=30")
chk(len(fin_bundle["transformed_feature_names"])   == 26, "FIN feature_names count=26")

# ── TreeExplainer — Healthcare XGBoost ───────────────────────────────────────
X_hc = hc_bundle["X_test_processed"]
ex_hc = shap.TreeExplainer(hc_xgb)
sv_hc = ex_hc(X_hc)
if sv_hc.values.ndim == 3:
    sv_vals_hc = sv_hc.values[:, :, 1]
else:
    sv_vals_hc = sv_hc.values

chk(sv_vals_hc.shape == (114, 30),          "HC XGB SHAP shape (114,30)")
chk(np.isfinite(sv_vals_hc).all(),          "HC XGB SHAP values finite")
chk(sv_vals_hc.shape[1] == len(hc_bundle["feature_names"]),
                                             "HC XGB SHAP cols == feature count")

# Verify SHAP values align with model predictions
y_prob_hc  = hc_xgb.predict_proba(X_hc)[:, 1]
y_pred_hc  = hc_xgb.predict(X_hc)
y_true_hc  = hc_bundle["y_test"].to_numpy()
chk(y_pred_hc.shape == (114,),              "HC XGB pred shape (114,)")
chk(set(np.unique(y_pred_hc)).issubset({0,1}), "HC XGB pred labels 0/1")
chk(np.isfinite(y_prob_hc).all(),           "HC XGB probs finite")

# ── TreeExplainer — Finance XGBoost ──────────────────────────────────────────
X_fin = fin_bundle["X_test_processed"]
ex_fin = shap.TreeExplainer(fin_xgb)
sv_fin = ex_fin(X_fin)
if sv_fin.values.ndim == 3:
    sv_vals_fin = sv_fin.values[:, :, 1]
else:
    sv_vals_fin = sv_fin.values

chk(sv_vals_fin.shape == (6483, 26),        "FIN XGB SHAP shape (6483,26)")
chk(np.isfinite(sv_vals_fin).all(),         "FIN XGB SHAP values finite")
chk(sv_vals_fin.shape[1] == len(fin_bundle["transformed_feature_names"]),
                                             "FIN XGB SHAP cols == feature count")

y_prob_fin = fin_xgb.predict_proba(X_fin)[:, 1]
y_pred_fin = fin_xgb.predict(X_fin)
y_true_fin = fin_bundle["y_test"].to_numpy()
chk(y_pred_fin.shape == (6483,),            "FIN XGB pred shape (6483,)")
chk(set(np.unique(y_pred_fin)).issubset({0,1}), "FIN XGB pred labels 0/1")
chk(np.isfinite(y_prob_fin).all(),          "FIN XGB probs finite")

# ── LinearExplainer — LR models ──────────────────────────────────────────────
masker_hc  = shap.maskers.Independent(hc_bundle["X_train_processed"],  max_samples=200)
masker_fin = shap.maskers.Independent(fin_bundle["X_train_processed"], max_samples=200)
ex_lr_hc   = shap.LinearExplainer(hc_lr,  masker_hc)
ex_lr_fin  = shap.LinearExplainer(fin_lr, masker_fin)
sv_lr_hc   = ex_lr_hc(X_hc)
sv_lr_fin  = ex_lr_fin(X_fin)

chk(sv_lr_hc.values.shape  == (114, 30),   "HC LR SHAP shape (114,30)")
chk(sv_lr_fin.values.shape == (6483, 26),  "FIN LR SHAP shape (6483,26)")
chk(np.isfinite(sv_lr_hc.values).all(),    "HC LR SHAP values finite")
chk(np.isfinite(sv_lr_fin.values).all(),   "FIN LR SHAP values finite")

# ── Global importance sanity ──────────────────────────────────────────────────
mean_abs_hc  = np.abs(sv_vals_hc).mean(axis=0)
mean_abs_fin = np.abs(sv_vals_fin).mean(axis=0)
chk(len(mean_abs_hc)  == 30, "HC global importance length=30")
chk(len(mean_abs_fin) == 26, "FIN global importance length=26")
chk(float(mean_abs_hc.max())  > 0, "HC max mean|SHAP| > 0")
chk(float(mean_abs_fin.max()) > 0, "FIN max mean|SHAP| > 0")

# ── Local example consistency ─────────────────────────────────────────────────
# Pick first TP in each domain and verify SHAP sum ≈ log-odds
tp_hc  = np.where((y_true_hc == 1) & (y_pred_hc == 1))[0][0]
tp_fin = np.where((y_true_fin == 1) & (y_pred_fin == 1))[0][0]

# SHAP additivity: base_value + sum(shap_values) ≈ raw margin
raw_hc  = hc_xgb.predict(X_hc[[tp_hc]],  output_margin=True)[0]
raw_fin = fin_xgb.predict(X_fin[[tp_fin]], output_margin=True)[0]
shap_sum_hc  = float(sv_hc.base_values[tp_hc])  + float(sv_vals_hc[tp_hc].sum())
shap_sum_fin = float(sv_fin.base_values[tp_fin]) + float(sv_vals_fin[tp_fin].sum())

chk(abs(shap_sum_hc  - raw_hc)  < 0.05, f"HC SHAP additivity check (diff={abs(shap_sum_hc-raw_hc):.4f})")
chk(abs(shap_sum_fin - raw_fin) < 0.05, f"FIN SHAP additivity check (diff={abs(shap_sum_fin-raw_fin):.4f})")

# ── Figure files ──────────────────────────────────────────────────────────────
figures = [
    "healthcare_shap_beeswarm.png",
    "healthcare_shap_importance.png",
    "healthcare_shap_waterfall.png",
    "healthcare_shap_dependence.png",
    "healthcare_local_explanation.png",
    "healthcare_lr_shap_importance.png",
    "finance_shap_beeswarm.png",
    "finance_shap_importance.png",
    "finance_shap_waterfall.png",
    "finance_shap_dependence.png",
    "finance_local_explanation.png",
    "finance_lr_shap_importance.png",
]
for fname in figures:
    p = ROOT / "outputs/figures" / fname
    chk(p.exists() and p.stat().st_size > 2000, f"Figure non-empty: {fname}")

# ── Report files ──────────────────────────────────────────────────────────────
reports = [
    "shap_global_importance.csv",
    "shap_local_examples.csv",
    "shap_interpretation_summary.txt",
]
for fname in reports:
    p = ROOT / "outputs/reports" / fname
    chk(p.exists() and p.stat().st_size > 100, f"Report non-empty: {fname}")

# ── CSV row counts ────────────────────────────────────────────────────────────
import csv
with open(ROOT / "outputs/reports/shap_global_importance.csv", encoding="utf-8") as f:
    gi_rows = list(csv.DictReader(f))
chk(len(gi_rows) == (30 + 26) * 2,   # 4 model×domain combos × features
    f"shap_global_importance.csv row count={len(gi_rows)} (expected {(30+26)*2})")

with open(ROOT / "outputs/reports/shap_local_examples.csv", encoding="utf-8") as f:
    le_rows = list(csv.DictReader(f))
chk(len(le_rows) > 0, f"shap_local_examples.csv has rows ({len(le_rows)})")

# ── No retraining check — model files predate this script ────────────────────
import time
model_mtime = (ROOT / "models/healthcare_xgboost.joblib").stat().st_mtime
script_mtime = (ROOT / "validate_stage4.py").stat().st_mtime
chk(model_mtime < script_mtime, "HC XGB model predates validation script (no retraining)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"ALL {len([l for l in open(__file__).readlines() if 'chk(' in l])} CHECKS PASSED")
