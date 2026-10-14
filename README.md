# Explainable Machine Learning for Interpreting Black-Box Predictions Using SHAP

## Objective

Train high-accuracy classification models on two real-world domains, then use SHAP (SHapley Additive exPlanations) to make every prediction interpretable — showing which features pushed the model toward or away from a given outcome.

---

## Problem Statement

Modern ML models such as gradient-boosted trees achieve strong predictive performance but are difficult to interpret. In high-stakes domains like healthcare and finance, a prediction without an explanation is insufficient. This project demonstrates how SHAP bridges that gap: it assigns each feature a contribution score for every individual prediction, grounded in cooperative game theory.

---

## Domains

### Healthcare — Breast Cancer Diagnosis
Classify a tumour as **Benign** or **Malignant** from 30 numeric cell-nucleus measurements derived from fine-needle aspirate images.

### Finance — Credit Risk Assessment
Predict whether a loan applicant will **Default** or **Not Default** based on 11 personal and loan-level attributes.

---

## Datasets

| Domain | File | Rows | Features | Target |
|---|---|---|---|---|
| Healthcare | `dataset/brca.csv` | 569 | 30 numeric | Malignant (1) / Benign (0) |
| Finance | `dataset/credit_risk_dataset.csv` | 32 581 (32 414 after cleaning) | 11 raw → 26 processed | Default (1) / No Default (0) |

**Healthcare source:** UCI ML Repository — Breast Cancer Wisconsin (Diagnostic).  
**Finance source:** Kaggle — Credit Risk Dataset (public domain).

---

## Architecture Overview

```
dataset/          Raw CSV files
src/              Preprocessing, training, evaluation, explainability modules
train.py          Orchestrates full pipeline (preprocessing → training → SHAP)
models/           Saved joblib artefacts (preprocessors, models, split bundles)
outputs/figures/  SHAP and evaluation plots (PNG)
outputs/reports/  JSON/CSV/TXT metrics and summaries
app.py            Streamlit dashboard (port 8501)
api.py            FastAPI REST API (port 8000)
```

---

## Technology Stack

| Layer | Library | Version range |
|---|---|---|
| Data | pandas, numpy | ≥2.0, ≥1.24 |
| ML | scikit-learn | ≥1.3 |
| Boosting | xgboost | ≥2.0 |
| Explainability | shap | ≥0.44 |
| Tuning | optuna | ≥3.4 |
| Visualisation | matplotlib, seaborn | ≥3.7, ≥0.12 |
| Dashboard | streamlit | ≥1.32 |
| API | fastapi, uvicorn, pydantic | ≥0.110, ≥0.29, ≥2.6 |

Python 3.12 · Windows 11 development environment.

---

## Data Preprocessing

**Healthcare**
- 569 rows, no missing values.
- StandardScaler applied to all 30 numeric features.
- Stratified 80/20 train/test split → 455 train, 114 test.
- Malignant rate: 37.4% train / 36.8% test.

**Finance**
- 32 581 raw rows.
- Step 1: 5 rows with `person_age > 100` capped at 100.
- Step 2: 2 rows where `person_emp_length > person_age` dropped.
- Step 3: 165 exact duplicates dropped → 32 414 clean rows.
- Numeric features: StandardScaler. Categorical features: OneHotEncoder.
- Stratified 80/20 split → 25 931 train, 6 483 test.
- Default rate: 21.9% train / 21.9% test.
- Processed feature count: 26.

---

## Models and Hyperparameter Tuning

Both domains train two models:
- **Logistic Regression** — baseline, interpretable linear model.
- **XGBoost** — primary model, tuned with Optuna (100 trials, 5-fold CV, ROC-AUC objective).

Key XGBoost hyperparameters found by Optuna:

| Domain | n_estimators | max_depth | learning_rate | scale_pos_weight |
|---|---|---|---|---|
| Healthcare | 377 | 7 | 0.259 | 1.68 |
| Finance | 459 | 4 | 0.120 | 3.57 |

---

## Evaluation Metrics and Results

All metrics are from the held-out test set. No data leakage.

### Healthcare (Breast Cancer) — 114 test samples

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9954 | 0.9954 ± 0.0048 |
| **XGBoost** | **0.9825** | **1.0000** | **0.9524** | **0.9756** | **0.9917** | **0.9944 ± 0.0034** |

XGBoost confusion matrix: TN=72, FP=0, FN=2, TP=40.

### Finance (Credit Risk) — 6 483 test samples

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8095 | 0.5449 | 0.7835 | 0.6428 | 0.8704 | 0.8706 ± 0.0053 |
| **XGBoost** | **0.9136** | **0.7918** | **0.8209** | **0.8061** | **0.9518** | **0.9497 ± 0.0052** |

XGBoost confusion matrix: TN=4759, FP=306, FN=254, TP=1164.

---

## SHAP Methodology

SHAP assigns each feature a contribution value for a specific prediction using Shapley values from cooperative game theory. For tree models, `shap.TreeExplainer` computes exact values efficiently.

**Global explanations** (beeswarm, bar, dependence plots) show which features matter most across the entire test set.

**Local explanations** (waterfall, force plots) show why the model made a specific prediction for one sample.

SHAP values in this project are in **log-odds (margin) space** — they are not probabilities and do not establish causation.

### Limitations
- SHAP explains the model, not the true data-generating process.
- Correlated features can split importance between them.
- TreeExplainer is exact for tree models; other explainers use approximations.
- Feature importance is not the same as causal effect.

---

## Streamlit Dashboard Features

Run on port 8501. Seven pages accessible from the sidebar:

1. **Project Overview** — domain cards, key metrics, quick-start guide.
2. **Dataset Explorer** — raw data preview, class distribution, feature statistics.
3. **Model Performance** — accuracy/AUC tiles, confusion matrices, ROC and PR curves.
4. **Interactive Prediction** — enter feature values, get a live prediction with probability.
5. **SHAP Explainability** — global beeswarm/bar/dependence plots; local waterfall for any test sample.
6. **Model Comparison** — side-by-side XGBoost vs Logistic Regression metrics.
7. **About & Limitations** — methodology, disclaimers, responsible-use guidance.

Domain selector in the sidebar switches between Healthcare and Finance throughout.

---

## FastAPI Endpoints

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info and endpoint map |
| GET | `/health` | Artifact loading status |
| GET | `/models` | Model metadata and metrics |
| POST | `/predict/healthcare` | Breast-cancer prediction (30 numeric fields) |
| POST | `/predict/finance` | Credit-risk prediction (11 fields) |
| POST | `/explain/healthcare` | Prediction + top-15 SHAP contributions |
| POST | `/explain/finance` | Prediction + top-15 SHAP contributions |

Interactive docs: `http://localhost:8000/docs`

---

## Project Folder Structure

```
SHAP-Explainable-ML/
├── dataset/
│   ├── brca.csv
│   └── credit_risk_dataset.csv
├── models/
│   ├── healthcare_xgboost.joblib
│   ├── healthcare_logistic_regression.joblib
│   ├── healthcare_preprocessor.joblib
│   ├── healthcare_train_test_split.joblib
│   ├── finance_xgboost.joblib
│   ├── finance_logistic_regression.joblib
│   ├── finance_preprocessor.joblib
│   └── finance_train_test_split.joblib
├── outputs/
│   ├── figures/          (20 PNG plots)
│   └── reports/          (JSON, CSV, TXT metrics)
├── src/
│   ├── preprocessing.py
│   ├── train_healthcare.py
│   ├── train_finance.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── data_validation.py
│   └── utils.py
├── app.py                Streamlit dashboard
├── api.py                FastAPI API
├── train.py              Full pipeline runner
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── validate_stage5.py
└── validate_stage6.py
```

---

## Windows Virtual Environment Setup

```bat
cd E:\SHAP-Explainable-ML
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running Locally

### Streamlit dashboard
```bat
.venv\Scripts\streamlit.exe run app.py
```
Opens at http://localhost:8501

### FastAPI
```bat
.venv\Scripts\uvicorn.exe api:app --reload --port 8000
```
Opens at http://localhost:8000  
Interactive docs at http://localhost:8000/docs

### Retrain models (optional — pre-trained models are already saved)
```bat
python train.py
```

---

## Docker Build and Run

### Prerequisites
- Docker Desktop for Windows must be running before executing these commands.

### Build the image
```bash
docker build -t shap-explainable-ml:latest .
```

### Start both services with Docker Compose
```bash
docker compose up --build
```

This starts:
- `api` service on http://localhost:8000 (starts first, health-checked)
- `dashboard` service on http://localhost:8501 (starts after API is healthy)

### Stop cleanly
```bash
docker compose down
```

### Run a single service
```bash
# API only
docker compose up api

# Dashboard only (API must already be running)
docker compose up dashboard
```

---

## API Usage Examples

### Health check
```bash
curl http://localhost:8000/health
```

### Healthcare prediction
```bash
curl -X POST http://localhost:8000/predict/healthcare \
  -H "Content-Type: application/json" \
  -d '{
    "x.radius_mean": 17.99, "x.texture_mean": 10.38, "x.perimeter_mean": 122.8,
    "x.area_mean": 1001.0, "x.smoothness_mean": 0.1184, "x.compactness_mean": 0.2776,
    "x.concavity_mean": 0.3001, "x.concave_pts_mean": 0.1471, "x.symmetry_mean": 0.2419,
    "x.fractal_dim_mean": 0.07871, "x.radius_se": 1.095, "x.texture_se": 0.9053,
    "x.perimeter_se": 8.589, "x.area_se": 153.4, "x.smoothness_se": 0.006399,
    "x.compactness_se": 0.04904, "x.concavity_se": 0.05373, "x.concave_pts_se": 0.01587,
    "x.symmetry_se": 0.03003, "x.fractal_dim_se": 0.006193, "x.radius_worst": 25.38,
    "x.texture_worst": 17.33, "x.perimeter_worst": 184.6, "x.area_worst": 2019.0,
    "x.smoothness_worst": 0.1622, "x.compactness_worst": 0.6656, "x.concavity_worst": 0.7119,
    "x.concave_pts_worst": 0.2654, "x.symmetry_worst": 0.4601, "x.fractal_dim_worst": 0.1189
  }'
```

### Finance prediction
```bash
curl -X POST http://localhost:8000/predict/finance \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30, "person_income": 45000, "person_home_ownership": "RENT",
    "person_emp_length": 3.0, "loan_intent": "PERSONAL", "loan_grade": "C",
    "loan_amnt": 10000, "loan_int_rate": 13.5, "loan_percent_income": 0.22,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4
  }'
```

### Healthcare SHAP explanation
```bash
curl -X POST http://localhost:8000/explain/healthcare \
  -H "Content-Type: application/json" \
  -d '{ <same body as predict above> }'
```

---

## Troubleshooting

**`FileNotFoundError` for a model joblib file**  
The pre-trained models are in `models/`. If they are missing, run `python train.py` to regenerate them.

**Streamlit shows blank page or import error**  
Ensure the virtual environment is activated and all packages from `requirements.txt` are installed.

**FastAPI returns 503 on `/health`**  
One or more model artefacts failed to load. Check the `missing` field in the JSON response and verify the `models/` directory is intact.

**Docker build fails with "cannot connect to daemon"**  
Docker Desktop must be running. Open Docker Desktop from the Start menu, wait for the whale icon in the system tray to show "Docker Desktop is running", then retry.

**Port already in use**  
Stop any locally running Streamlit or uvicorn process before starting Docker Compose, or change the host port mapping in `compose.yaml`.

**`httpx2` not found during pip install**  
This package is listed in `requirements.txt` for the validation scripts. If it is unavailable on your platform, install `httpx` instead and update the import in the validation scripts.

---

## Limitations and Responsible-Use Disclaimers

- **Healthcare outputs are not medical diagnoses.** Do not use predictions from this system to make clinical decisions. Consult a qualified clinician.
- **Finance outputs are not lending decisions.** Do not use predictions to approve or deny credit applications.
- Models were trained on publicly available datasets and are intended for educational and research purposes only.
- SHAP explanations describe model behaviour, not causal relationships in the real world.
- Performance on data outside the training distribution is not guaranteed.
- No personally identifiable information is collected or stored by this application.

---

## References

1. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
2. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. UCI ML Repository.
3. Kaggle Credit Risk Dataset — https://www.kaggle.com/datasets/laotse/credit-risk-dataset
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
5. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.
