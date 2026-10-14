"""
Stage 5 validation script — tests all dashboard dependencies headlessly.
Run: .venv/Scripts/python.exe validate_stage5.py
"""
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).parent
errors = []

def chk(condition: bool, label: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        errors.append(label)


# ── 1. app.py syntax ──────────────────────────────────────────────────────────
import py_compile, tempfile
try:
    py_compile.compile(str(ROOT / "app.py"), doraise=True)
    chk(True, "app.py syntax valid")
except py_compile.PyCompileError as e:
    chk(False, f"app.py syntax error: {e}")

# ── 2. Artifact existence ─────────────────────────────────────────────────────
required_files = [
    "models/healthcare_train_test_split.joblib",
    "models/finance_train_test_split.joblib",
    "models/healthcare_xgboost.joblib",
    "models/healthcare_logistic_regression.joblib",
    "models/finance_xgboost.joblib",
    "models/finance_logistic_regression.joblib",
    "models/healthcare_preprocessor.joblib",
    "models/finance_preprocessor.joblib",
    "outputs/reports/model_metrics.json",
    "outputs/reports/shap_global_importance.csv",
]
for f in required_files:
    chk((ROOT / f).exists(), f"Exists: {f}")

# ── 3. Load artifacts ─────────────────────────────────────────────────────────
hc_bundle  = joblib.load(ROOT / "models/healthcare_train_test_split.joblib")
fin_bundle = joblib.load(ROOT / "models/finance_train_test_split.joblib")
hc_xgb     = joblib.load(ROOT / "models/healthcare_xgboost.joblib")
hc_lr      = joblib.load(ROOT / "models/healthcare_logistic_regression.joblib")
fin_xgb    = joblib.load(ROOT / "models/finance_xgboost.joblib")
fin_lr     = joblib.load(ROOT / "models/finance_logistic_regression.joblib")
hc_pipe    = joblib.load(ROOT / "models/healthcare_preprocessor.joblib")
fin_pipe   = joblib.load(ROOT / "models/finance_preprocessor.joblib")
chk(True, "All 8 artifacts loaded")

# ── 4. Bundle key verification ────────────────────────────────────────────────
chk("feature_names"            in hc_bundle,  "HC bundle has feature_names")
chk("X_test"                   in hc_bundle,  "HC bundle has X_test (raw)")
chk("X_test_processed"         in hc_bundle,  "HC bundle has X_test_processed")
chk("numeric_features"         in fin_bundle, "FIN bundle has numeric_features")
chk("categorical_features"     in fin_bundle, "FIN bundle has categorical_features")
chk("transformed_feature_names" in fin_bundle,"FIN bundle has transformed_feature_names")
chk("X_test"                   in fin_bundle, "FIN bundle has X_test (raw)")
chk("X_test_processed"         in fin_bundle, "FIN bundle has X_test_processed")

# ── 5. Metrics JSON ───────────────────────────────────────────────────────────
metrics = json.loads((ROOT / "outputs/reports/model_metrics.json").read_text())
chk(len(metrics) == 4, f"model_metrics.json has 4 entries")
hc_m  = [m for m in metrics if m["domain"] == "healthcare"]
fin_m = [m for m in metrics if m["domain"] == "finance"]
chk(len(hc_m)  == 2, "2 healthcare metric entries")
chk(len(fin_m) == 2, "2 finance metric entries")
for m in metrics:
    chk("cv_roc_auc_mean" in m, f"{m['domain']}/{m['label']} has cv_roc_auc_mean")

# ── 6. Healthcare pipeline: raw DataFrame → preprocessor → model ──────────────
row_hc  = hc_bundle["X_test"].iloc[[1]]
proc_hc = hc_pipe.transform(row_hc)
pred_hc = hc_xgb.predict(proc_hc)
prob_hc = hc_xgb.predict_proba(proc_hc)[:, 1]
chk(proc_hc.shape == (1, 30),                  f"HC pipe output shape (1,30)")
chk(int(pred_hc[0]) in {0, 1},                 "HC XGB pred is 0 or 1")
chk(0.0 <= float(prob_hc[0]) <= 1.0,           "HC XGB prob in [0,1]")
chk(int(pred_hc[0]) == 1,                      "HC sample 1 predicted Malignant (known TP)")

# ── 7. Finance pipeline: raw DataFrame → preprocessor → model ────────────────
row_fin  = fin_bundle["X_test"].iloc[[4]]
proc_fin = fin_pipe.transform(row_fin)
pred_fin = fin_xgb.predict(proc_fin)
prob_fin = fin_xgb.predict_proba(proc_fin)[:, 1]
chk(proc_fin.shape == (1, 26),                 f"FIN pipe output shape (1,26)")
chk(int(pred_fin[0]) in {0, 1},                "FIN XGB pred is 0 or 1")
chk(0.0 <= float(prob_fin[0]) <= 1.0,          "FIN XGB prob in [0,1]")
chk(int(pred_fin[0]) == 1,                     "FIN sample 4 predicted Default (known TP)")

# ── 8. Finance manual-input row (simulates form submission) ───────────────────
manual_fin = pd.DataFrame([{
    "person_age": 30, "person_income": 50000, "person_emp_length": 5.0,
    "loan_amnt": 10000, "loan_int_rate": 12.5, "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 4,
    "person_home_ownership": "RENT", "loan_intent": "PERSONAL",
    "loan_grade": "B", "cb_person_default_on_file": "N",
}])
proc_manual = fin_pipe.transform(manual_fin)
pred_manual = fin_xgb.predict(proc_manual)
prob_manual = fin_xgb.predict_proba(proc_manual)[:, 1]
chk(proc_manual.shape == (1, 26),              "FIN manual form pipe shape (1,26)")
chk(int(pred_manual[0]) in {0, 1},             "FIN manual pred is 0 or 1")
chk(0.0 <= float(prob_manual[0]) <= 1.0,       "FIN manual prob in [0,1]")

# ── 9. SHAP for single HC prediction ─────────────────────────────────────────
ex_hc = shap.TreeExplainer(hc_xgb)
sv_hc = ex_hc(proc_hc)
sv_vals_hc = sv_hc.values[0, :, 1] if sv_hc.values.ndim == 3 else sv_hc.values[0]
chk(sv_vals_hc.shape == (30,),                 "HC single SHAP shape (30,)")
chk(np.isfinite(sv_vals_hc).all(),             "HC single SHAP values finite")

# ── 10. SHAP for single FIN prediction ───────────────────────────────────────
ex_fin = shap.TreeExplainer(fin_xgb)
sv_fin = ex_fin(proc_fin)
sv_vals_fin = sv_fin.values[0, :, 1] if sv_fin.values.ndim == 3 else sv_fin.values[0]
chk(sv_vals_fin.shape == (26,),                "FIN single SHAP shape (26,)")
chk(np.isfinite(sv_vals_fin).all(),            "FIN single SHAP values finite")

# ── 11. SHAP additivity ───────────────────────────────────────────────────────
raw_margin_hc  = float(hc_xgb.predict(proc_hc,  output_margin=True)[0])
raw_margin_fin = float(fin_xgb.predict(proc_fin, output_margin=True)[0])
base_hc  = float(sv_hc.base_values[0, 1])  if sv_hc.base_values.ndim  == 2 else float(sv_hc.base_values[0])
base_fin = float(sv_fin.base_values[0, 1]) if sv_fin.base_values.ndim == 2 else float(sv_fin.base_values[0])
chk(abs(base_hc  + sv_vals_hc.sum()  - raw_margin_hc)  < 0.05, "HC SHAP additivity")
chk(abs(base_fin + sv_vals_fin.sum() - raw_margin_fin) < 0.05, "FIN SHAP additivity")

# ── 12. Feature name alignment ────────────────────────────────────────────────
disp_hc  = [n.replace("x.", "")                              for n in hc_bundle["feature_names"]]
disp_fin = [n.replace("num__","").replace("cat__","")        for n in fin_bundle["transformed_feature_names"]]
chk(len(disp_hc)  == 30, "HC display names count=30")
chk(len(disp_fin) == 26, "FIN display names count=26")

# ── 13. Required figures ──────────────────────────────────────────────────────
figs = [
    "healthcare_shap_beeswarm.png", "healthcare_shap_importance.png",
    "healthcare_shap_waterfall.png", "healthcare_shap_dependence.png",
    "healthcare_local_explanation.png",
    "finance_shap_beeswarm.png", "finance_shap_importance.png",
    "finance_shap_waterfall.png", "finance_shap_dependence.png",
    "finance_local_explanation.png",
    "healthcare_roc_curves.png", "finance_roc_curves.png",
    "healthcare_pr_curves.png",  "finance_pr_curves.png",
    "healthcare_xgboost_confusion_matrix.png",
    "finance_xgboost_confusion_matrix.png",
]
for f in figs:
    p = ROOT / "outputs/figures" / f
    chk(p.exists() and p.stat().st_size > 1000, f"Figure non-empty: {f}")

# ── 14. Invalid input handling (finance) ─────────────────────────────────────
try:
    bad = pd.DataFrame([{
        "person_age": -5, "person_income": 0, "person_emp_length": np.nan,
        "loan_amnt": 500, "loan_int_rate": np.nan, "loan_percent_income": 0.0,
        "cb_person_cred_hist_length": 0,
        "person_home_ownership": "RENT", "loan_intent": "PERSONAL",
        "loan_grade": "A", "cb_person_default_on_file": "N",
    }])
    proc_bad = fin_pipe.transform(bad)
    pred_bad = fin_xgb.predict(proc_bad)
    chk(int(pred_bad[0]) in {0, 1}, "FIN invalid input (NaN) handled gracefully by pipeline")
except Exception as e:
    chk(False, f"FIN invalid input raised exception: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED ({sum(1 for l in open(__file__) if 'chk(' in l)} assertions)")
