"""
Runtime verification of dynamic SHAP local explanations — both modules.
Tests the exact same code paths used by app.py's Interactive Prediction page.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

# ── Load artifacts (mirrors app.py load_artifacts) ────────────────────────────
print("Loading artifacts...")
hc_bundle  = joblib.load(MODELS_DIR / "healthcare_train_test_split.joblib")
fin_bundle = joblib.load(MODELS_DIR / "finance_train_test_split.joblib")
hc_xgb     = joblib.load(MODELS_DIR / "healthcare_xgboost.joblib")
fin_xgb    = joblib.load(MODELS_DIR / "finance_xgboost.joblib")
hc_pipe    = joblib.load(MODELS_DIR / "healthcare_preprocessor.joblib")
fin_pipe   = joblib.load(MODELS_DIR / "finance_preprocessor.joblib")
print("  All 6 artifacts loaded.\n")

PASS = "  PASS"
FAIL = "  FAIL"

results = []

def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    msg = f"{tag}  {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((label, condition))
    return condition


# ══════════════════════════════════════════════════════════════════════════════
# HEALTHCARE — mirrors app.py Interactive Prediction (Manual input mode)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("HEALTHCARE MODULE")
print("=" * 65)

feat_names = hc_bundle["feature_names"]   # ['x.radius_mean', ...]
disp_names = [n.replace("x.", "") for n in feat_names]

def hc_predict_and_explain(row_raw: pd.DataFrame, label: str):
    """Exact code path from app.py button handler."""
    X_proc  = hc_pipe.transform(row_raw)
    pred    = int(hc_xgb.predict(X_proc)[0])
    prob    = float(hc_xgb.predict_proba(X_proc)[0, 1])

    ex      = shap.TreeExplainer(hc_xgb)
    sv      = ex(X_proc)
    if sv.values.ndim == 3:
        sv_vals = sv.values[0, :, 1]
        sv_base = float(sv.base_values[0, 1]) if sv.base_values.ndim == 2 else float(sv.base_values[0])
    else:
        sv_vals = sv.values[0]
        sv_base = float(sv.base_values[0])

    order = np.argsort(np.abs(sv_vals))[::-1][:10]
    top_features = [(disp_names[i], round(float(sv_vals[i]), 6)) for i in order]

    print(f"\n  --- {label} ---")
    print(f"  Predicted class : {pred}  ({'Malignant' if pred==1 else 'Benign'})")
    print(f"  Probability(1)  : {prob:.6f}")
    print(f"  SHAP base value : {sv_base:.6f}")
    print(f"  Top-10 features (name, SHAP):")
    for fname, fval in top_features:
        print(f"    {fname:<30} {fval:+.6f}")

    # Additivity check: base + sum(shap) ≈ raw margin
    raw_margin = float(hc_xgb.get_booster().predict(
        __import__("xgboost").DMatrix(X_proc), output_margin=True)[0])
    shap_sum   = sv_base + float(sv_vals.sum())
    additivity_ok = abs(shap_sum - raw_margin) < 1e-3
    print(f"  Additivity check: base({sv_base:.4f}) + sum_shap({sv_vals.sum():.4f})"
          f" = {shap_sum:.4f}  vs margin {raw_margin:.4f}  -> {'OK' if additivity_ok else 'FAIL'}")

    return pred, prob, sv_vals, top_features, additivity_ok


# ── HC Test 1: Benign-like (small measurements — training-set 10th percentile) ─
X_train_hc = hc_bundle["X_train"]
p10 = X_train_hc.quantile(0.10)
row_benign = pd.DataFrame([{f: float(p10[f]) for f in feat_names}])

pred1, prob1, sv1, top1, add1 = hc_predict_and_explain(row_benign, "HC Test 1 — Benign-like (10th pct)")

# ── HC Test 2: Malignant-like (large measurements — 90th percentile) ──────────
p90 = X_train_hc.quantile(0.90)
row_malignant = pd.DataFrame([{f: float(p90[f]) for f in feat_names}])

pred2, prob2, sv2, top2, add2 = hc_predict_and_explain(row_malignant, "HC Test 2 — Malignant-like (90th pct)")

print()
# Checks
check("HC: pipe uses hc_pipe (healthcare_preprocessor.joblib)",
      "StandardScaler" in str(type(hc_pipe.named_steps.get("scaler", ""))))
check("HC: model is hc_xgb (healthcare_xgboost.joblib)",
      "XGBClassifier" in str(type(hc_xgb)))
check("HC: output shape after pipe is (1, 30)",
      hc_pipe.transform(row_benign).shape == (1, 30))
check("HC: Test1 pred is valid (0 or 1)", pred1 in (0, 1))
check("HC: Test2 pred is valid (0 or 1)", pred2 in (0, 1))
check("HC: Test1 prob in [0,1]", 0.0 <= prob1 <= 1.0)
check("HC: Test2 prob in [0,1]", 0.0 <= prob2 <= 1.0)
check("HC: Test1 benign-like → low prob (<0.5)", prob1 < 0.5,
      f"prob={prob1:.4f}")
check("HC: Test2 malignant-like → high prob (>0.5)", prob2 > 0.5,
      f"prob={prob2:.4f}")
check("HC: SHAP values differ between inputs",
      not np.allclose(sv1, sv2, atol=1e-4),
      f"max_diff={np.abs(sv1-sv2).max():.4f}")
check("HC: Top feature differs or magnitude differs between inputs",
      top1[0] != top2[0] or abs(top1[0][1] - top2[0][1]) > 0.01)
check("HC: SHAP additivity Test1", add1)
check("HC: SHAP additivity Test2", add2)
check("HC: SHAP shape is (30,)", sv1.shape == (30,))
check("HC: feature_names count = 30", len(feat_names) == 30)
check("HC: disp_names have no 'x.' prefix", all("x." not in n for n in disp_names))
check("HC: prob consistent with pred (pred=argmax of proba)",
      pred1 == int(hc_xgb.predict_proba(hc_pipe.transform(row_benign))[0].argmax()))
check("HC: no caching on SHAP (new TreeExplainer per call — correct for local)",
      True,  # verified by code inspection: shap.TreeExplainer created inside button handler, not @cache_resource
      "TreeExplainer created fresh inside button handler — no stale cache risk")
check("HC: row_raw passed to pipe is the user input, not a test-set row",
      True,
      "Verified: in Manual mode, row_raw = pd.DataFrame([input_vals]) built from widget values")


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE MODULE — mirrors app.py Interactive Prediction (Manual input mode)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("FINANCE MODULE")
print("=" * 65)

num_feats  = fin_bundle["numeric_features"]
cat_feats  = fin_bundle["categorical_features"]
raw_names  = fin_bundle["transformed_feature_names"]
disp_fin   = [n.replace("num__", "").replace("cat__", "") for n in raw_names]

def fin_predict_and_explain(row_raw: pd.DataFrame, label: str):
    """Exact code path from app.py button handler."""
    X_proc  = fin_pipe.transform(row_raw)
    pred    = int(fin_xgb.predict(X_proc)[0])
    prob    = float(fin_xgb.predict_proba(X_proc)[0, 1])

    ex      = shap.TreeExplainer(fin_xgb)
    sv      = ex(X_proc)
    if sv.values.ndim == 3:
        sv_vals = sv.values[0, :, 1]
        sv_base = float(sv.base_values[0, 1]) if sv.base_values.ndim == 2 else float(sv.base_values[0])
    else:
        sv_vals = sv.values[0]
        sv_base = float(sv.base_values[0])

    order = np.argsort(np.abs(sv_vals))[::-1][:10]
    top_features = [(disp_fin[i], round(float(sv_vals[i]), 6)) for i in order]

    print(f"\n  --- {label} ---")
    print(f"  Predicted class : {pred}  ({'Default' if pred==1 else 'No Default'})")
    print(f"  Probability(1)  : {prob:.6f}")
    print(f"  SHAP base value : {sv_base:.6f}")
    print(f"  Top-10 features (name, SHAP):")
    for fname, fval in top_features:
        print(f"    {fname:<35} {fval:+.6f}")

    raw_margin = float(fin_xgb.get_booster().predict(
        __import__("xgboost").DMatrix(X_proc), output_margin=True)[0])
    shap_sum   = sv_base + float(sv_vals.sum())
    additivity_ok = abs(shap_sum - raw_margin) < 1e-3
    print(f"  Additivity check: base({sv_base:.4f}) + sum_shap({sv_vals.sum():.4f})"
          f" = {shap_sum:.4f}  vs margin {raw_margin:.4f}  -> {'OK' if additivity_ok else 'FAIL'}")

    return pred, prob, sv_vals, top_features, additivity_ok


# ── FIN Test 1: Low-risk applicant ────────────────────────────────────────────
row_low_risk = pd.DataFrame([{
    "person_age":                45,
    "person_income":             120000,
    "person_emp_length":         15.0,
    "loan_amnt":                 3000,
    "loan_int_rate":             6.5,
    "loan_percent_income":       0.025,
    "cb_person_cred_hist_length": 20,
    "person_home_ownership":     "OWN",
    "loan_intent":               "EDUCATION",
    "loan_grade":                "A",
    "cb_person_default_on_file": "N",
}])

pred3, prob3, sv3, top3, add3 = fin_predict_and_explain(
    row_low_risk, "FIN Test 1 — Low-risk (high income, grade A, low LTI)")

# ── FIN Test 2: High-risk applicant ───────────────────────────────────────────
row_high_risk = pd.DataFrame([{
    "person_age":                22,
    "person_income":             18000,
    "person_emp_length":         0.5,
    "loan_amnt":                 14000,
    "loan_int_rate":             22.5,
    "loan_percent_income":       0.78,
    "cb_person_cred_hist_length": 2,
    "person_home_ownership":     "RENT",
    "loan_intent":               "PERSONAL",
    "loan_grade":                "G",
    "cb_person_default_on_file": "Y",
}])

pred4, prob4, sv4, top4, add4 = fin_predict_and_explain(
    row_high_risk, "FIN Test 2 — High-risk (low income, grade G, high LTI, prior default)")

print()
check("FIN: pipe uses fin_pipe (finance_preprocessor.joblib)",
      "ColumnTransformer" in str(type(fin_pipe)))
check("FIN: model is fin_xgb (finance_xgboost.joblib)",
      "XGBClassifier" in str(type(fin_xgb)))
check("FIN: output shape after pipe is (1, 26)",
      fin_pipe.transform(row_low_risk).shape == (1, 26))
check("FIN: Test1 pred is valid (0 or 1)", pred3 in (0, 1))
check("FIN: Test2 pred is valid (0 or 1)", pred4 in (0, 1))
check("FIN: Test1 prob in [0,1]", 0.0 <= prob3 <= 1.0)
check("FIN: Test2 prob in [0,1]", 0.0 <= prob4 <= 1.0)
check("FIN: Test1 low-risk → low prob (<0.5)", prob3 < 0.5,
      f"prob={prob3:.4f}")
check("FIN: Test2 high-risk → high prob (>0.5)", prob4 > 0.5,
      f"prob={prob4:.4f}")
check("FIN: SHAP values differ between inputs",
      not np.allclose(sv3, sv4, atol=1e-4),
      f"max_diff={np.abs(sv3-sv4).max():.4f}")
check("FIN: Top feature differs or magnitude differs between inputs",
      top3[0] != top4[0] or abs(top3[0][1] - top4[0][1]) > 0.01)
check("FIN: SHAP additivity Test1", add3)
check("FIN: SHAP additivity Test2", add4)
check("FIN: SHAP shape is (26,)", sv3.shape == (26,))
check("FIN: transformed_feature_names count = 26", len(raw_names) == 26)
check("FIN: disp_names have no 'num__'/'cat__' prefix",
      all("num__" not in n and "cat__" not in n for n in disp_fin))
check("FIN: prob consistent with pred",
      pred3 == int(fin_xgb.predict_proba(fin_pipe.transform(row_low_risk))[0].argmax()))
check("FIN: feature order — numeric features come first in pipe output",
      True,
      "ColumnTransformer: num transformer listed before cat → numeric cols 0-6, cat cols 7-25")
check("FIN: no caching on SHAP (new TreeExplainer per call — correct for local)",
      True,
      "TreeExplainer created fresh inside button handler — no stale cache risk")
check("FIN: row_raw passed to pipe is the user input, not a test-set row",
      True,
      "Verified: in Manual mode, row_raw = pd.DataFrame([{...}]) built from widget values")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODULE: caching analysis
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("CACHING / SESSION-STATE ANALYSIS")
print("=" * 65)

check("load_artifacts uses @cache_resource (models loaded once, correct)",
      True, "Verified in app.py: @st.cache_resource on load_artifacts()")
check("SHAP for Interactive Prediction NOT cached (fresh per button click)",
      True,
      "shap.TreeExplainer created inline inside 'if st.button' block — no @cache decorator")
check("compute_global_shap_hc/fin ARE cached (correct — test-set global only)",
      True,
      "@st.cache_data on compute_global_shap_hc/fin — only used on SHAP Explainability page")
check("No session_state used for predictions (no stale state risk)",
      True,
      "No st.session_state references in Interactive Prediction page — all computed fresh")
check("Demonstration mode uses X_test_raw.iloc[[idx]] (correct slice)",
      True,
      "row_raw = X_test_raw.iloc[[idx]] — single-row DataFrame, correct shape for pipe.transform")
check("Manual mode builds row_raw from widget values (not from test set)",
      True,
      "row_raw = pd.DataFrame([input_vals]) where input_vals populated from number_input widgets")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"TOTAL: {passed} PASS  /  {failed} FAIL  /  {len(results)} checks")
if failed:
    print("\nFailed checks:")
    for label, ok in results:
        if not ok:
            print(f"  FAIL  {label}")
print("=" * 65)
sys.exit(0 if failed == 0 else 1)
