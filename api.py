"""
Stage 6 — FastAPI prediction and explanation API.

Endpoints
---------
GET  /                      Service information
GET  /health                Artifact-loading health check
GET  /models                Available domains and model metadata
POST /predict/healthcare    Breast-cancer prediction (30 numeric features)
POST /predict/finance       Credit-risk prediction (raw schema, 11 fields)
POST /explain/healthcare    Prediction + local SHAP explanation
POST /explain/finance       Prediction + local SHAP explanation

Run from project root with venv active:
    .venv\\Scripts\\uvicorn.exe api:app --reload --port 8000

DISCLAIMER
----------
All predictions are for educational and research purposes only.
Healthcare outputs are not medical diagnoses.
Finance outputs are not official lending decisions.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"

_ARTIFACT_PATHS = {
    "hc_xgb":    MODELS_DIR / "healthcare_xgboost.joblib",
    "hc_lr":     MODELS_DIR / "healthcare_logistic_regression.joblib",
    "hc_pipe":   MODELS_DIR / "healthcare_preprocessor.joblib",
    "hc_bundle": MODELS_DIR / "healthcare_train_test_split.joblib",
    "fin_xgb":   MODELS_DIR / "finance_xgboost.joblib",
    "fin_lr":    MODELS_DIR / "finance_logistic_regression.joblib",
    "fin_pipe":  MODELS_DIR / "finance_preprocessor.joblib",
    "fin_bundle": MODELS_DIR / "finance_train_test_split.joblib",
}

# ── Shared state (loaded once at startup) ─────────────────────────────────────
_ARTS: dict = {}
_LOAD_ERRORS: list[str] = []


def _load_all() -> None:
    for key, path in _ARTIFACT_PATHS.items():
        try:
            _ARTS[key] = joblib.load(path)
        except Exception as e:
            _LOAD_ERRORS.append(f"{key}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Explainable ML — SHAP Prediction API",
    description=(
        "Prediction and SHAP explanation API for breast-cancer diagnosis (healthcare) "
        "and credit-risk assessment (finance). "
        "All outputs are for educational purposes only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit dev origin only
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DISCLAIMER_HC  = ("Educational use only. Not a medical diagnosis or clinical recommendation. "
                  "Consult a qualified clinician for health decisions.")
DISCLAIMER_FIN = ("Educational use only. Not an official lending decision or financial advice. "
                  "Do not use to approve or deny credit applications.")


# ── Feature-name helpers ───────────────────────────────────────────────────────
def _hc_display(raw: list[str]) -> list[str]:
    return [n.replace("x.", "") for n in raw]

def _fin_display(raw: list[str]) -> list[str]:
    return [n.replace("num__", "").replace("cat__", "") for n in raw]


# ── SHAP helper ────────────────────────────────────────────────────────────────
def _shap_for_row(model, X_proc: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return (shap_values_1d, base_value) for a single processed row.
    Values are in log-odds (margin) space.
    """
    ex = shap.TreeExplainer(model)
    sv = ex(X_proc)
    if sv.values.ndim == 3:          # multi-class guard
        vals = sv.values[0, :, 1]
        base = float(sv.base_values[0, 1]) if sv.base_values.ndim == 2 else float(sv.base_values[0])
    else:
        vals = sv.values[0]
        base = float(sv.base_values[0])
    return vals, base


def _guard_loaded(key: str):
    if key not in _ARTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Artifact '{key}' failed to load at startup. Check server logs.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

# ── Healthcare input ───────────────────────────────────────────────────────────
# Feature names contain dots (x.radius_mean) which are invalid Python identifiers.
# We use Field(alias=...) and model_config populate_by_name=True so callers
# can send either the aliased JSON key or the Python field name.

class HealthcareInput(BaseModel):
    model_config = {"populate_by_name": True}

    radius_mean:        float = Field(..., alias="x.radius_mean",        gt=0)
    texture_mean:       float = Field(..., alias="x.texture_mean",       gt=0)
    perimeter_mean:     float = Field(..., alias="x.perimeter_mean",     gt=0)
    area_mean:          float = Field(..., alias="x.area_mean",          gt=0)
    smoothness_mean:    float = Field(..., alias="x.smoothness_mean",    gt=0)
    compactness_mean:   float = Field(..., alias="x.compactness_mean",   ge=0)
    concavity_mean:     float = Field(..., alias="x.concavity_mean",     ge=0)
    concave_pts_mean:   float = Field(..., alias="x.concave_pts_mean",   ge=0)
    symmetry_mean:      float = Field(..., alias="x.symmetry_mean",      gt=0)
    fractal_dim_mean:   float = Field(..., alias="x.fractal_dim_mean",   gt=0)
    radius_se:          float = Field(..., alias="x.radius_se",          ge=0)
    texture_se:         float = Field(..., alias="x.texture_se",         ge=0)
    perimeter_se:       float = Field(..., alias="x.perimeter_se",       ge=0)
    area_se:            float = Field(..., alias="x.area_se",            ge=0)
    smoothness_se:      float = Field(..., alias="x.smoothness_se",      ge=0)
    compactness_se:     float = Field(..., alias="x.compactness_se",     ge=0)
    concavity_se:       float = Field(..., alias="x.concavity_se",       ge=0)
    concave_pts_se:     float = Field(..., alias="x.concave_pts_se",     ge=0)
    symmetry_se:        float = Field(..., alias="x.symmetry_se",        ge=0)
    fractal_dim_se:     float = Field(..., alias="x.fractal_dim_se",     ge=0)
    radius_worst:       float = Field(..., alias="x.radius_worst",       gt=0)
    texture_worst:      float = Field(..., alias="x.texture_worst",      gt=0)
    perimeter_worst:    float = Field(..., alias="x.perimeter_worst",    gt=0)
    area_worst:         float = Field(..., alias="x.area_worst",         gt=0)
    smoothness_worst:   float = Field(..., alias="x.smoothness_worst",   gt=0)
    compactness_worst:  float = Field(..., alias="x.compactness_worst",  ge=0)
    concavity_worst:    float = Field(..., alias="x.concavity_worst",    ge=0)
    concave_pts_worst:  float = Field(..., alias="x.concave_pts_worst",  ge=0)
    symmetry_worst:     float = Field(..., alias="x.symmetry_worst",     gt=0)
    fractal_dim_worst:  float = Field(..., alias="x.fractal_dim_worst",  gt=0)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame with the original x.* column names."""
        return pd.DataFrame([{
            "x.radius_mean":       self.radius_mean,
            "x.texture_mean":      self.texture_mean,
            "x.perimeter_mean":    self.perimeter_mean,
            "x.area_mean":         self.area_mean,
            "x.smoothness_mean":   self.smoothness_mean,
            "x.compactness_mean":  self.compactness_mean,
            "x.concavity_mean":    self.concavity_mean,
            "x.concave_pts_mean":  self.concave_pts_mean,
            "x.symmetry_mean":     self.symmetry_mean,
            "x.fractal_dim_mean":  self.fractal_dim_mean,
            "x.radius_se":         self.radius_se,
            "x.texture_se":        self.texture_se,
            "x.perimeter_se":      self.perimeter_se,
            "x.area_se":           self.area_se,
            "x.smoothness_se":     self.smoothness_se,
            "x.compactness_se":    self.compactness_se,
            "x.concavity_se":      self.concavity_se,
            "x.concave_pts_se":    self.concave_pts_se,
            "x.symmetry_se":       self.symmetry_se,
            "x.fractal_dim_se":    self.fractal_dim_se,
            "x.radius_worst":      self.radius_worst,
            "x.texture_worst":     self.texture_worst,
            "x.perimeter_worst":   self.perimeter_worst,
            "x.area_worst":        self.area_worst,
            "x.smoothness_worst":  self.smoothness_worst,
            "x.compactness_worst": self.compactness_worst,
            "x.concavity_worst":   self.concavity_worst,
            "x.concave_pts_worst": self.concave_pts_worst,
            "x.symmetry_worst":    self.symmetry_worst,
            "x.fractal_dim_worst": self.fractal_dim_worst,
        }])


# ── Finance input ──────────────────────────────────────────────────────────────
class FinanceInput(BaseModel):
    person_age:                 int   = Field(..., ge=18, le=100)
    person_income:              int   = Field(..., gt=0)
    person_home_ownership:      Literal["MORTGAGE", "OWN", "RENT", "OTHER"]
    person_emp_length:          Optional[float] = Field(None, ge=0, le=60)
    loan_intent:                Literal["DEBTCONSOLIDATION", "EDUCATION",
                                        "HOMEIMPROVEMENT", "MEDICAL",
                                        "PERSONAL", "VENTURE"]
    loan_grade:                 Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt:                  int   = Field(..., ge=500, le=35000)
    loan_int_rate:              Optional[float] = Field(None, ge=5.0, le=25.0)
    loan_percent_income:        float = Field(..., ge=0.0, le=1.0)
    cb_person_default_on_file:  Literal["N", "Y"]
    cb_person_cred_hist_length: int   = Field(..., ge=0, le=30)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "person_age":                self.person_age,
            "person_income":             self.person_income,
            "person_home_ownership":     self.person_home_ownership,
            "person_emp_length":         self.person_emp_length,
            "loan_intent":               self.loan_intent,
            "loan_grade":                self.loan_grade,
            "loan_amnt":                 self.loan_amnt,
            "loan_int_rate":             self.loan_int_rate,
            "loan_percent_income":       self.loan_percent_income,
            "cb_person_default_on_file": self.cb_person_default_on_file,
            "cb_person_cred_hist_length": self.cb_person_cred_hist_length,
        }])


# ── Response schemas ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    domain:          str
    model_name:      str
    predicted_class: int
    predicted_label: str
    prob_class_0:    float
    prob_class_1:    float
    disclaimer:      str


class FeatureContribution(BaseModel):
    feature:       str
    shap_value:    float
    direction:     str


class ExplanationResponse(BaseModel):
    domain:              str
    model_name:          str
    predicted_class:     int
    predicted_label:     str
    prob_class_0:        float
    prob_class_1:        float
    shap_base_value:     float
    shap_note:           str
    top_contributions:   list[FeatureContribution]
    disclaimer:          str


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Explainable ML — SHAP Prediction API",
        "version": "1.0.0",
        "domains": ["healthcare", "finance"],
        "endpoints": {
            "GET  /health":              "Artifact health check",
            "GET  /models":              "Model metadata",
            "POST /predict/healthcare":  "Breast-cancer prediction",
            "POST /predict/finance":     "Credit-risk prediction",
            "POST /explain/healthcare":  "Prediction + SHAP explanation",
            "POST /explain/finance":     "Prediction + SHAP explanation",
        },
        "disclaimer": "All outputs are for educational purposes only.",
    }


@app.get("/health", tags=["Info"])
def health():
    loaded  = [k for k in _ARTIFACT_PATHS if k in _ARTS]
    missing = [k for k in _ARTIFACT_PATHS if k not in _ARTS]
    ok = len(missing) == 0
    return {
        "status":          "healthy" if ok else "degraded",
        "artifacts_loaded": len(loaded),
        "artifacts_total":  len(_ARTIFACT_PATHS),
        "missing":          missing,
        "load_errors":      _LOAD_ERRORS if _LOAD_ERRORS else None,
    }


@app.get("/models", tags=["Info"])
def models_info():
    import json
    metrics_path = ROOT / "outputs" / "reports" / "model_metrics.json"
    metrics = []
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def _meta(domain: str, model_key: str) -> dict:
        m = next((x for x in metrics
                  if x["domain"] == domain and x["label"] == "XGBoost"), {})
        return {
            "domain":    domain,
            "primary_model": "XGBoost",
            "baseline_model": "Logistic Regression",
            "loaded":    model_key in _ARTS,
            "accuracy":  m.get("accuracy"),
            "roc_auc":   m.get("roc_auc"),
            "recall":    m.get("recall"),
            "f1":        m.get("f1"),
        }

    return {
        "healthcare": _meta("healthcare", "hc_xgb"),
        "finance":    _meta("finance",    "fin_xgb"),
    }


# ── Healthcare predict ─────────────────────────────────────────────────────────
@app.post("/predict/healthcare", response_model=PredictionResponse, tags=["Prediction"])
def predict_healthcare(body: HealthcareInput):
    _guard_loaded("hc_xgb")
    _guard_loaded("hc_pipe")
    try:
        df    = body.to_dataframe()
        X     = _ARTS["hc_pipe"].transform(df)
        pred  = int(_ARTS["hc_xgb"].predict(X)[0])
        probs = _ARTS["hc_xgb"].predict_proba(X)[0]
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Inference failed. Check input values.")
    return PredictionResponse(
        domain="healthcare",
        model_name="XGBoost",
        predicted_class=pred,
        predicted_label="Malignant" if pred == 1 else "Benign",
        prob_class_0=round(float(probs[0]), 6),
        prob_class_1=round(float(probs[1]), 6),
        disclaimer=DISCLAIMER_HC,
    )


# ── Finance predict ────────────────────────────────────────────────────────────
@app.post("/predict/finance", response_model=PredictionResponse, tags=["Prediction"])
def predict_finance(body: FinanceInput):
    _guard_loaded("fin_xgb")
    _guard_loaded("fin_pipe")
    try:
        df    = body.to_dataframe()
        X     = _ARTS["fin_pipe"].transform(df)
        pred  = int(_ARTS["fin_xgb"].predict(X)[0])
        probs = _ARTS["fin_xgb"].predict_proba(X)[0]
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Inference failed. Check input values.")
    return PredictionResponse(
        domain="finance",
        model_name="XGBoost",
        predicted_class=pred,
        predicted_label="Default" if pred == 1 else "No Default",
        prob_class_0=round(float(probs[0]), 6),
        prob_class_1=round(float(probs[1]), 6),
        disclaimer=DISCLAIMER_FIN,
    )


# ── Healthcare explain ─────────────────────────────────────────────────────────
@app.post("/explain/healthcare", response_model=ExplanationResponse, tags=["Explanation"])
def explain_healthcare(body: HealthcareInput):
    _guard_loaded("hc_xgb")
    _guard_loaded("hc_pipe")
    _guard_loaded("hc_bundle")
    try:
        df    = body.to_dataframe()
        X     = _ARTS["hc_pipe"].transform(df)
        pred  = int(_ARTS["hc_xgb"].predict(X)[0])
        probs = _ARTS["hc_xgb"].predict_proba(X)[0]
        sv_vals, base = _shap_for_row(_ARTS["hc_xgb"], X)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Explanation failed. Check input values.")

    raw_names  = _ARTS["hc_bundle"]["feature_names"]
    disp_names = _hc_display(raw_names)
    order      = np.argsort(np.abs(sv_vals))[::-1][:15]
    contribs   = [
        FeatureContribution(
            feature=disp_names[i],
            shap_value=round(float(sv_vals[i]), 6),
            direction="increases malignancy score" if sv_vals[i] > 0
                      else "decreases malignancy score",
        )
        for i in order
    ]
    return ExplanationResponse(
        domain="healthcare",
        model_name="XGBoost",
        predicted_class=pred,
        predicted_label="Malignant" if pred == 1 else "Benign",
        prob_class_0=round(float(probs[0]), 6),
        prob_class_1=round(float(probs[1]), 6),
        shap_base_value=round(base, 6),
        shap_note=("SHAP values are in log-odds (margin) space. They are not probabilities "
                   "and do not establish causation."),
        top_contributions=contribs,
        disclaimer=DISCLAIMER_HC,
    )


# ── Finance explain ────────────────────────────────────────────────────────────
@app.post("/explain/finance", response_model=ExplanationResponse, tags=["Explanation"])
def explain_finance(body: FinanceInput):
    _guard_loaded("fin_xgb")
    _guard_loaded("fin_pipe")
    _guard_loaded("fin_bundle")
    try:
        df    = body.to_dataframe()
        X     = _ARTS["fin_pipe"].transform(df)
        pred  = int(_ARTS["fin_xgb"].predict(X)[0])
        probs = _ARTS["fin_xgb"].predict_proba(X)[0]
        sv_vals, base = _shap_for_row(_ARTS["fin_xgb"], X)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Explanation failed. Check input values.")

    raw_names  = _ARTS["fin_bundle"]["transformed_feature_names"]
    disp_names = _fin_display(raw_names)
    order      = np.argsort(np.abs(sv_vals))[::-1][:15]
    contribs   = [
        FeatureContribution(
            feature=disp_names[i],
            shap_value=round(float(sv_vals[i]), 6),
            direction="increases default risk" if sv_vals[i] > 0
                      else "decreases default risk",
        )
        for i in order
    ]
    return ExplanationResponse(
        domain="finance",
        model_name="XGBoost",
        predicted_class=pred,
        predicted_label="Default" if pred == 1 else "No Default",
        prob_class_0=round(float(probs[0]), 6),
        prob_class_1=round(float(probs[1]), 6),
        shap_base_value=round(base, 6),
        shap_note=("SHAP values are in log-odds (margin) space. They are not probabilities "
                   "and do not establish causation."),
        top_contributions=contribs,
        disclaimer=DISCLAIMER_FIN,
    )
