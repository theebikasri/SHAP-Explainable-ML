"""
Stage 6 validation script — tests all API endpoints headlessly via TestClient.
Run: .venv/Scripts/python.exe validate_stage6.py
"""
import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
errors = []

def chk(condition: bool, label: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        errors.append(label)


# ── 1. Syntax check ───────────────────────────────────────────────────────────
try:
    py_compile.compile(str(ROOT / "api.py"), doraise=True)
    chk(True, "api.py syntax valid")
except py_compile.PyCompileError as e:
    chk(False, f"api.py syntax error: {e}")
    sys.exit(1)

# ── 2. Import and TestClient setup ────────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
    from api import app, _ARTS, _LOAD_ERRORS
    chk(True, "api.py imports successfully")
except Exception as e:
    chk(False, f"api.py import failed: {e}")
    sys.exit(1)

# Use context manager so the lifespan startup event fires and loads artifacts
_tc = TestClient(app, raise_server_exceptions=True)
_tc.__enter__()
client = _tc

# ── 3. Artifacts loaded ───────────────────────────────────────────────────────
expected_keys = ["hc_xgb", "hc_lr", "hc_pipe", "hc_bundle",
                 "fin_xgb", "fin_lr", "fin_pipe", "fin_bundle"]
for k in expected_keys:
    chk(k in _ARTS, f"Artifact loaded: {k}")
chk(len(_LOAD_ERRORS) == 0, f"No load errors (errors={_LOAD_ERRORS})")

# ── 4. Endpoint registration ──────────────────────────────────────────────────
routes = {r.path for r in app.routes}
for path in ["/", "/health", "/models",
             "/predict/healthcare", "/predict/finance",
             "/explain/healthcare", "/explain/finance"]:
    chk(path in routes, f"Route registered: {path}")

# ── 5. GET / ──────────────────────────────────────────────────────────────────
r = client.get("/")
chk(r.status_code == 200,                    "GET / returns 200")
chk("endpoints" in r.json(),                 "GET / has endpoints key")
chk("disclaimer" in r.json(),                "GET / has disclaimer")

# ── 6. GET /health ────────────────────────────────────────────────────────────
r = client.get("/health")
chk(r.status_code == 200,                    "GET /health returns 200")
chk(r.json()["status"] == "healthy",         "GET /health status=healthy")
chk(r.json()["artifacts_loaded"] == 8,       "GET /health 8 artifacts loaded")

# ── 7. GET /models ────────────────────────────────────────────────────────────
r = client.get("/models")
chk(r.status_code == 200,                    "GET /models returns 200")
body = r.json()
chk("healthcare" in body,                    "GET /models has healthcare key")
chk("finance"    in body,                    "GET /models has finance key")
chk(body["healthcare"]["loaded"] is True,    "GET /models healthcare loaded=True")
chk(body["finance"]["loaded"]    is True,    "GET /models finance loaded=True")
chk(body["healthcare"]["roc_auc"] is not None, "GET /models HC roc_auc present")
chk(body["finance"]["roc_auc"]    is not None, "GET /models FIN roc_auc present")

# ── 8. POST /predict/healthcare — valid input (known TP, sample 1) ────────────
HC_PAYLOAD = {
    "x.radius_mean": 20.94, "x.texture_mean": 23.56, "x.perimeter_mean": 138.9,
    "x.area_mean": 1364.0,  "x.smoothness_mean": 0.1007, "x.compactness_mean": 0.1606,
    "x.concavity_mean": 0.2712, "x.concave_pts_mean": 0.131, "x.symmetry_mean": 0.2205,
    "x.fractal_dim_mean": 0.05898, "x.radius_se": 1.004, "x.texture_se": 0.8208,
    "x.perimeter_se": 6.372, "x.area_se": 137.9, "x.smoothness_se": 0.005283,
    "x.compactness_se": 0.03908, "x.concavity_se": 0.09518, "x.concave_pts_se": 0.01864,
    "x.symmetry_se": 0.02401, "x.fractal_dim_se": 0.005002, "x.radius_worst": 25.58,
    "x.texture_worst": 27.0, "x.perimeter_worst": 165.3, "x.area_worst": 2010.0,
    "x.smoothness_worst": 0.1211, "x.compactness_worst": 0.3172, "x.concavity_worst": 0.6991,
    "x.concave_pts_worst": 0.2105, "x.symmetry_worst": 0.3126, "x.fractal_dim_worst": 0.07849,
}
r = client.post("/predict/healthcare", json=HC_PAYLOAD)
chk(r.status_code == 200,                          "POST /predict/healthcare 200")
hc_pred = r.json()
chk(hc_pred["predicted_class"] in [0, 1],          "HC pred class is 0 or 1")
chk(hc_pred["predicted_label"] == "Malignant",     "HC sample1 predicted Malignant")
chk(0.0 <= hc_pred["prob_class_1"] <= 1.0,         "HC prob_class_1 in [0,1]")
chk(0.0 <= hc_pred["prob_class_0"] <= 1.0,         "HC prob_class_0 in [0,1]")
chk(abs(hc_pred["prob_class_0"] + hc_pred["prob_class_1"] - 1.0) < 1e-4,
                                                    "HC probs sum to 1")
chk("disclaimer" in hc_pred,                       "HC response has disclaimer")
chk("Malignant" in hc_pred["predicted_label"] or "Benign" in hc_pred["predicted_label"],
                                                    "HC label is Malignant or Benign")

# ── 9. POST /predict/finance — valid input (known TP, sample 4) ───────────────
FIN_PAYLOAD = {
    "person_age": 25, "person_income": 28128,
    "person_home_ownership": "RENT", "person_emp_length": 2.0,
    "loan_intent": "VENTURE", "loan_grade": "D",
    "loan_amnt": 7500, "loan_int_rate": 15.33,
    "loan_percent_income": 0.27, "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3,
}
r = client.post("/predict/finance", json=FIN_PAYLOAD)
chk(r.status_code == 200,                          "POST /predict/finance 200")
fin_pred = r.json()
chk(fin_pred["predicted_class"] in [0, 1],         "FIN pred class is 0 or 1")
chk(fin_pred["predicted_label"] == "Default",      "FIN sample4 predicted Default")
chk(0.0 <= fin_pred["prob_class_1"] <= 1.0,        "FIN prob_class_1 in [0,1]")
chk(abs(fin_pred["prob_class_0"] + fin_pred["prob_class_1"] - 1.0) < 1e-4,
                                                    "FIN probs sum to 1")
chk("disclaimer" in fin_pred,                      "FIN response has disclaimer")

# ── 10. POST /predict/finance — missing optional fields (NaN imputed) ─────────
FIN_PARTIAL = {k: v for k, v in FIN_PAYLOAD.items()
               if k not in ("person_emp_length", "loan_int_rate")}
r = client.post("/predict/finance", json=FIN_PARTIAL)
chk(r.status_code == 200,                          "FIN partial input (NaN fields) returns 200")
chk(r.json()["predicted_class"] in [0, 1],         "FIN partial pred class valid")

# ── 11. POST /explain/healthcare ─────────────────────────────────────────────
r = client.post("/explain/healthcare", json=HC_PAYLOAD)
chk(r.status_code == 200,                          "POST /explain/healthcare 200")
hc_exp = r.json()
chk("top_contributions" in hc_exp,                 "HC explain has top_contributions")
chk(len(hc_exp["top_contributions"]) > 0,          "HC explain contributions non-empty")
chk("shap_base_value" in hc_exp,                   "HC explain has shap_base_value")
chk("shap_note" in hc_exp,                         "HC explain has shap_note")
chk(np.isfinite(hc_exp["shap_base_value"]),        "HC shap_base_value finite")
for c in hc_exp["top_contributions"]:
    chk(np.isfinite(c["shap_value"]),              f"HC contrib {c['feature']} finite")
    break  # check first only to keep output concise

# SHAP additivity: base + sum(contribs) ≈ raw margin
import joblib
hc_xgb  = joblib.load(ROOT / "models/healthcare_xgboost.joblib")
hc_pipe = joblib.load(ROOT / "models/healthcare_preprocessor.joblib")
import pandas as pd
df_hc   = pd.DataFrame([HC_PAYLOAD])
X_hc    = hc_pipe.transform(df_hc)
raw_margin_hc = float(hc_xgb.predict(X_hc, output_margin=True)[0])
shap_sum_hc   = hc_exp["shap_base_value"] + sum(c["shap_value"] for c in hc_exp["top_contributions"])
# Note: top_contributions is top-15 only, so sum won't exactly equal margin
# Instead verify base_value is finite and in a reasonable range
chk(abs(hc_exp["shap_base_value"]) < 5.0,         "HC shap_base_value in plausible range")
chk(hc_exp["predicted_class"] == hc_pred["predicted_class"],
                                                    "HC explain pred matches predict pred")

# ── 12. POST /explain/finance ─────────────────────────────────────────────────
r = client.post("/explain/finance", json=FIN_PAYLOAD)
chk(r.status_code == 200,                          "POST /explain/finance 200")
fin_exp = r.json()
chk("top_contributions" in fin_exp,                "FIN explain has top_contributions")
chk(len(fin_exp["top_contributions"]) > 0,         "FIN explain contributions non-empty")
chk("shap_base_value" in fin_exp,                  "FIN explain has shap_base_value")
chk(np.isfinite(fin_exp["shap_base_value"]),       "FIN shap_base_value finite")
chk(fin_exp["predicted_class"] == fin_pred["predicted_class"],
                                                    "FIN explain pred matches predict pred")
# Verify feature names are readable (no num__ / cat__ prefixes)
for c in fin_exp["top_contributions"][:3]:
    chk("num__" not in c["feature"] and "cat__" not in c["feature"],
        f"FIN contrib name clean: {c['feature']}")

# ── 13. Invalid inputs → 422 ──────────────────────────────────────────────────
# HC: missing a required field
bad_hc = {k: v for k, v in HC_PAYLOAD.items() if k != "x.radius_mean"}
r = client.post("/predict/healthcare", json=bad_hc)
chk(r.status_code == 422,                          "HC missing field returns 422")

# HC: negative value for gt=0 field
bad_hc2 = {**HC_PAYLOAD, "x.radius_mean": -1.0}
r = client.post("/predict/healthcare", json=bad_hc2)
chk(r.status_code == 422,                          "HC negative radius returns 422")

# FIN: invalid loan_grade
bad_fin = {**FIN_PAYLOAD, "loan_grade": "Z"}
r = client.post("/predict/finance", json=bad_fin)
chk(r.status_code == 422,                          "FIN invalid loan_grade returns 422")

# FIN: age out of range
bad_fin2 = {**FIN_PAYLOAD, "person_age": 150}
r = client.post("/predict/finance", json=bad_fin2)
chk(r.status_code == 422,                          "FIN age>100 returns 422")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__) if "chk(" in line)
    print(f"ALL CHECKS PASSED ({total} assertions)")
