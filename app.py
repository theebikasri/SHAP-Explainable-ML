"""
Explainable ML Dashboard — Streamlit application (Stage 5).

Run from project root with venv active:
    .venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"
DATASET_DIR = ROOT / "dataset"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable ML — SHAP Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── App background ─────────────────────────────────────────────────────── */
  .stApp { background-color: #F1F5F9; }

  /* ── Main content text — ensure dark on light ───────────────────────────── */
  .stApp, .stApp p, .stApp span, .stApp div,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
  .stApp li, .stApp td, .stApp th {
    color: #1E293B;
  }

  /* ── Widget labels (number_input, selectbox, slider, radio, checkbox) ────── */
  .stApp [data-testid="stWidgetLabel"] p,
  .stApp [data-testid="stWidgetLabel"] label,
  .stApp label,
  .stApp .stRadio label,
  .stApp .stCheckbox label,
  .stApp .stSelectbox label,
  .stApp .stNumberInput label,
  .stApp .stSlider label,
  .stApp .stTextInput label,
  .stApp .stTextArea label {
    color: #1E293B !important;
  }

  /* ── Radio button option text ────────────────────────────────────────────── */
/* Radio options: dark text on main page, light text in sidebar */
.stApp .stRadio div[role="radiogroup"] label,
.stApp .stRadio div[role="radiogroup"] p {
    color: #1E293B !important;
}

/* Sidebar navigation labels */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    color: #F1F5F9 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Sidebar navigation spacing */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}

  /* ── Number / text input fields ─────────────────────────────────────────── */
  .stApp input[type="number"],
  .stApp input[type="text"],
  .stApp textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
  }

  /* ── Selectbox / dropdown ────────────────────────────────────────────────── */
  .stApp [data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
  }

  /* ── Slider value label ──────────────────────────────────────────────────── */
  .stApp [data-testid="stSlider"] p,
  .stApp [data-testid="stSlider"] span {
    color: #1E293B !important;
  }

  /* ── Captions ────────────────────────────────────────────────────────────── */
  .stApp [data-testid="stCaptionContainer"] p,
  .stApp .stCaption {
    color: #64748B !important;
  }

  /* ── Tabs ────────────────────────────────────────────────────────────────── */
  .stApp [data-testid="stTabs"] button p,
  .stApp [data-testid="stTabs"] button {
    color: #1E293B !important;
  }

  /* ── Expander header ─────────────────────────────────────────────────────── */
  .stApp [data-testid="stExpander"] summary p,
  .stApp [data-testid="stExpander"] summary span {
    color: #1E293B !important;
  }

  /* ── Metric widget ───────────────────────────────────────────────────────── */
  .stApp [data-testid="stMetricLabel"] p,
  .stApp [data-testid="stMetricValue"] {
    color: #1E293B !important;
  }

  /* ── Dataframe / table text ──────────────────────────────────────────────── */
  .stApp [data-testid="stDataFrame"] * {
    color: #1E293B;
  }

  /* ── Alert / info / warning / error boxes ───────────────────────────────── */
  .stApp [data-testid="stAlert"] p {
    color: #1E293B !important;
  }

  /* ── Button ──────────────────────────────────────────────────────────────── */
  .stApp .stButton > button {
    color: #FFFFFF !important;
    background-color: #3B4FE4 !important;
    border: none;
  }
  .stApp .stButton > button:hover {
    background-color: #2D3FD3 !important;
  }

  /* ── Cards ───────────────────────────────────────────────────────────────── */
  .card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    color: #1E293B;
  }

  /* ── Metric tiles ────────────────────────────────────────────────────────── */
  .metric-tile {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
  }
  .metric-tile .val { font-size: 1.9rem; font-weight: 700; color: #3B4FE4; }
  .metric-tile .lbl { font-size: 0.78rem; color: #64748B; margin-top: 2px; }

  /* ── Section headers ─────────────────────────────────────────────────────── */
  .section-header {
    font-size: 1.25rem; font-weight: 700;
    color: #1E293B; margin-bottom: 0.6rem;
  }

  /* ── Domain badges ───────────────────────────────────────────────────────── */
  .badge-hc  { background:#EFF6FF; color:#1D4ED8;
               border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:600; }
  .badge-fin { background:#F0FDF4; color:#15803D;
               border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:600; }

  /* ── Prediction result boxes ─────────────────────────────────────────────── */
  .pred-positive { background:#FEF2F2; border-left:4px solid #EF4444;
                   border-radius:8px; padding:1rem 1.2rem; color:#1E293B; }
  .pred-negative { background:#F0FDF4; border-left:4px solid #22C55E;
                   border-radius:8px; padding:1rem 1.2rem; color:#1E293B; }

  /* ── Disclaimer box ──────────────────────────────────────────────────────── */
  .disclaimer { background:#FFFBEB; border-left:4px solid #F59E0B;
                border-radius:8px; padding:0.8rem 1rem;
                font-size:0.82rem; color:#78350F; }

  /* ── Sidebar — scoped, no wildcard bleed ─────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #1E293B !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] li {
    color: #E2E8F0 !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: #334155;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #94A3B8 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stRadio"] label,
  section[data-testid="stSidebar"] [data-testid="stRadio"] p,
  section[data-testid="stSidebar"] div[role="radiogroup"] label,
  section[data-testid="stSidebar"] div[role="radiogroup"] p {
    color: #E2E8F0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: #334155 !important;
    color: #E2E8F0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders — models and data loaded once per session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load all four models and both split bundles once."""
    try:
        hc_bundle  = joblib.load(MODELS_DIR / "healthcare_train_test_split.joblib")
        fin_bundle = joblib.load(MODELS_DIR / "finance_train_test_split.joblib")
        hc_xgb     = joblib.load(MODELS_DIR / "healthcare_xgboost.joblib")
        hc_lr      = joblib.load(MODELS_DIR / "healthcare_logistic_regression.joblib")
        fin_xgb    = joblib.load(MODELS_DIR / "finance_xgboost.joblib")
        fin_lr     = joblib.load(MODELS_DIR / "finance_logistic_regression.joblib")
        hc_pipe    = joblib.load(MODELS_DIR / "healthcare_preprocessor.joblib")
        fin_pipe   = joblib.load(MODELS_DIR / "finance_preprocessor.joblib")
        return {
            "hc_bundle": hc_bundle, "fin_bundle": fin_bundle,
            "hc_xgb": hc_xgb, "hc_lr": hc_lr,
            "fin_xgb": fin_xgb, "fin_lr": fin_lr,
            "hc_pipe": hc_pipe, "fin_pipe": fin_pipe,
        }
    except FileNotFoundError as e:
        st.error(f"Required model artifact not found: {e}. Run train.py first.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_metrics() -> list[dict]:
    path = REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        st.error("model_metrics.json not found. Run train.py first.")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame:
    path = REPORTS_DIR / "shap_global_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Computing SHAP values…")
def get_shap_explainers(arts: dict):
    """Build TreeExplainers for both XGBoost models once."""
    ex_hc  = shap.TreeExplainer(arts["hc_xgb"])
    ex_fin = shap.TreeExplainer(arts["fin_xgb"])
    return ex_hc, ex_fin


@st.cache_data(show_spinner="Computing global SHAP values…")
def compute_global_shap_hc(_arts):
    X = _arts["hc_bundle"]["X_test_processed"]
    ex = shap.TreeExplainer(_arts["hc_xgb"])
    sv = ex(X)
    if sv.values.ndim == 3:
        sv = shap.Explanation(values=sv.values[:,:,1],
                              base_values=sv.base_values[:,1] if sv.base_values.ndim==2 else sv.base_values,
                              data=sv.data, feature_names=sv.feature_names)
    return sv


@st.cache_data(show_spinner="Computing global SHAP values…")
def compute_global_shap_fin(_arts):
    X = _arts["fin_bundle"]["X_test_processed"]
    ex = shap.TreeExplainer(_arts["fin_xgb"])
    sv = ex(X)
    if sv.values.ndim == 3:
        sv = shap.Explanation(values=sv.values[:,:,1],
                              base_values=sv.base_values[:,1] if sv.base_values.ndim==2 else sv.base_values,
                              data=sv.data, feature_names=sv.feature_names)
    return sv


# ── Feature name helpers ───────────────────────────────────────────────────────
def hc_display(raw: list[str]) -> list[str]:
    return [n.replace("x.", "") for n in raw]

def fin_display(raw: list[str]) -> list[str]:
    return [n.replace("num__", "").replace("cat__", "") for n in raw]


# ── Figure helper ──────────────────────────────────────────────────────────────
def show_figure(path: Path, caption: str = "", width: int = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=(width is None))
    else:
        st.warning(f"Figure not found: {path.name}. Re-run src/explainability.py.")


def metric_tile(value, label: str) -> str:
    return (f'<div class="metric-tile">'
            f'<div class="val">{value}</div>'
            f'<div class="lbl">{label}</div></div>')


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 SHAP Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Project Overview", "Dataset Explorer", "Model Performance",
         "Interactive Prediction", "SHAP Explainability",
         "Model Comparison", "About & Limitations"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    domain = st.selectbox("Domain", ["Healthcare — Breast Cancer",
                                      "Finance — Credit Risk"])
    IS_HC = domain.startswith("Healthcare")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#94A3B8;line-height:1.5">'
        'Models: XGBoost + Logistic Regression<br>'
        'SHAP: TreeExplainer (XGBoost)<br>'
        'Python 3.12 · scikit-learn 1.9<br>'
        'Stage 5 — Streamlit Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

# Load everything
arts    = load_artifacts()
metrics = load_metrics()
shap_imp = load_shap_importance()

hc_metrics  = [m for m in metrics if m["domain"] == "healthcare"]
fin_metrics = [m for m in metrics if m["domain"] == "finance"]
dom_metrics = hc_metrics if IS_HC else fin_metrics


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.markdown('<p class="section-header">Explainable Machine Learning for Interpreting Black-Box Predictions Using SHAP</p>', unsafe_allow_html=True)
    st.markdown("This dashboard demonstrates how SHAP (SHapley Additive exPlanations) can make complex machine learning models interpretable across two real-world domains.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
          <span class="badge-hc">Healthcare</span>
          <h4 style="margin:0.6rem 0 0.3rem">Breast Cancer Diagnosis</h4>
          <p style="color:#475569;font-size:0.88rem;margin:0">
            569 samples · 30 numeric features · Binary classification<br>
            <b>Target:</b> Malignant (M=1) vs Benign (B=0)<br>
            <b>Dataset:</b> Wisconsin Breast Cancer (brca.csv)
          </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
          <span class="badge-fin">Finance</span>
          <h4 style="margin:0.6rem 0 0.3rem">Credit Risk / Loan Default</h4>
          <p style="color:#475569;font-size:0.88rem;margin:0">
            32,414 clean rows · 11 features · Binary classification<br>
            <b>Target:</b> Default (1) vs No Default (0)<br>
            <b>Dataset:</b> Credit Risk Dataset (credit_risk_dataset.csv)
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Key Results")
    hc_xgb_m  = next(m for m in hc_metrics  if m["label"] == "XGBoost")
    fin_xgb_m = next(m for m in fin_metrics if m["label"] == "XGBoost")

    cols = st.columns(6)
    tiles = [
        (f"{hc_xgb_m['accuracy']:.1%}",  "HC Accuracy"),
        (f"{hc_xgb_m['roc_auc']:.3f}",   "HC ROC-AUC"),
        (f"{hc_xgb_m['recall']:.1%}",    "HC Malignant Recall"),
        (f"{fin_xgb_m['accuracy']:.1%}", "FIN Accuracy"),
        (f"{fin_xgb_m['roc_auc']:.3f}",  "FIN ROC-AUC"),
        (f"{fin_xgb_m['recall']:.1%}",   "FIN Default Recall"),
    ]
    for col, (val, lbl) in zip(cols, tiles):
        col.markdown(metric_tile(val, lbl), unsafe_allow_html=True)

    st.markdown("### What is SHAP?")
    st.markdown("""
    <div class="card">
    <b>SHAP (SHapley Additive exPlanations)</b> is a game-theory-based method that assigns each
    feature a contribution value for a specific prediction. It answers: <i>"How much did each
    feature push this prediction higher or lower compared to the average prediction?"</i>
    <br><br>
    <b>Key properties:</b>
    <ul style="margin:0.4rem 0 0 1rem;color:#475569">
      <li>Consistent and locally accurate — SHAP values sum to the difference between the prediction and the base value.</li>
      <li>Model-agnostic in principle; TreeExplainer is used here for exact, fast computation on XGBoost.</li>
      <li>SHAP values are in <b>log-odds (margin) space</b> for these models — they are not probabilities.</li>
      <li>SHAP measures model attribution, <b>not causation</b>. A high SHAP value does not mean a feature caused the outcome.</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
          <b>Baseline — Logistic Regression</b><br>
          <span style="color:#64748B;font-size:0.85rem">
          class_weight='balanced' · lbfgs solver · max_iter=2000
          </span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <b>Advanced — XGBoost</b><br>
          <span style="color:#64748B;font-size:0.85rem">
          Optuna-tuned (30 trials) · scale_pos_weight from training labels ·
          TreeExplainer for SHAP
          </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    badge = '<span class="badge-hc">Healthcare</span>' if IS_HC else '<span class="badge-fin">Finance</span>'
    st.markdown(f'<p class="section-header">Dataset Explorer &nbsp;{badge}</p>', unsafe_allow_html=True)

    if IS_HC:
        df = pd.read_csv(DATASET_DIR / "brca.csv")
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        df = df.drop(columns=unnamed)
        target_col = "y"
        pos_label, neg_label = "Malignant (M)", "Benign (B)"
        pos_val, neg_val = "M", "B"
    else:
        df = pd.read_csv(DATASET_DIR / "credit_risk_dataset.csv")
        target_col = "loan_status"
        pos_label, neg_label = "Default (1)", "No Default (0)"
        pos_val, neg_val = 1, 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_tile(f"{len(df):,}", "Total Rows"), unsafe_allow_html=True)
    c2.markdown(metric_tile(str(df.shape[1]-1), "Features"), unsafe_allow_html=True)
    miss = int(df.isnull().sum().sum())
    c3.markdown(metric_tile(f"{miss:,}", "Missing Values"), unsafe_allow_html=True)
    dups = int(df.duplicated().sum())
    c4.markdown(metric_tile(f"{dups:,}", "Duplicates (raw)"), unsafe_allow_html=True)

    st.markdown("#### Class Distribution")
    vc = df[target_col].value_counts()
    col_a, col_b = st.columns([1, 2])
    with col_a:
        dist_df = pd.DataFrame({
            "Class": [pos_label, neg_label],
            "Count": [int(vc.get(pos_val, 0)), int(vc.get(neg_val, 0))],
            "Pct":   [f"{vc.get(pos_val,0)/len(df):.1%}", f"{vc.get(neg_val,0)/len(df):.1%}"],
        })
        st.dataframe(dist_df, hide_index=True, use_container_width=True)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 2.8))
        colors = ["#EF4444", "#22C55E"]
        ax.bar([neg_label, pos_label],
               [int(vc.get(neg_val, 0)), int(vc.get(pos_val, 0))],
               color=colors[::-1])
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Feature Summary")
    feat_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()
    cat_cols = df[feat_cols].select_dtypes(include="object").columns.tolist()

    col_x, col_y = st.columns(2)
    col_x.markdown(f"**Numeric features:** {len(num_cols)}")
    col_y.markdown(f"**Categorical features:** {len(cat_cols)}")

    if cat_cols:
        st.markdown("**Categorical unique values:**")
        for c in cat_cols:
            st.markdown(f"- `{c}`: {sorted(df[c].dropna().unique().tolist())}")

    with st.expander("Numeric feature statistics"):
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

    if miss > 0:
        st.markdown("#### Missing Values")
        miss_s = df.isnull().sum()
        miss_s = miss_s[miss_s > 0].reset_index()
        miss_s.columns = ["Feature", "Missing Count"]
        miss_s["Pct"] = (miss_s["Missing Count"] / len(df) * 100).round(2).astype(str) + "%"
        st.dataframe(miss_s, hide_index=True, use_container_width=True)
        st.info("Missing values are imputed using median imputation fitted on training data only — no leakage.")

    st.markdown("#### Sample Records (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    badge = '<span class="badge-hc">Healthcare</span>' if IS_HC else '<span class="badge-fin">Finance</span>'
    st.markdown(f'<p class="section-header">Model Performance &nbsp;{badge}</p>', unsafe_allow_html=True)

    pos_name = "Malignant" if IS_HC else "Default"
    note_recall = (
        "⚠️ In healthcare, **malignant-class recall** is critical — a false negative "
        "means a cancer case is missed."
        if IS_HC else
        "⚠️ In credit risk, **default-class recall** matters — a false negative means "
        "a defaulting borrower is not flagged."
    )
    st.info(note_recall)

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.markdown("#### Metrics on held-out test set")
    rows = []
    for m in dom_metrics:
        rows.append({
            "Model":     m["label"],
            "Accuracy":  f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "F1":        f"{m['f1']:.4f}",
            "ROC-AUC":   f"{m['roc_auc']:.4f}",
            "CV ROC-AUC": f"{m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Metric tiles for XGBoost ───────────────────────────────────────────────
    xgb_m = next(m for m in dom_metrics if m["label"] == "XGBoost")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, lbl) in zip(
        [c1, c2, c3, c4, c5],
        [(f"{xgb_m['accuracy']:.1%}", "Accuracy"),
         (f"{xgb_m['precision']:.1%}", f"{pos_name} Precision"),
         (f"{xgb_m['recall']:.1%}",   f"{pos_name} Recall"),
         (f"{xgb_m['f1']:.4f}",       "F1-Score"),
         (f"{xgb_m['roc_auc']:.4f}",  "ROC-AUC")],
    ):
        col.markdown(metric_tile(val, lbl), unsafe_allow_html=True)

    st.markdown("")

    # ── Confusion matrices ─────────────────────────────────────────────────────
    st.markdown("#### Confusion Matrices")
    prefix = "healthcare" if IS_HC else "finance"
    col_lr, col_xgb = st.columns(2)
    with col_lr:
        st.markdown("**Logistic Regression**")
        show_figure(FIGURES_DIR / f"{prefix}_logistic_regression_confusion_matrix.png")
    with col_xgb:
        st.markdown("**XGBoost**")
        show_figure(FIGURES_DIR / f"{prefix}_xgboost_confusion_matrix.png")

    # ── FP / FN explanation ────────────────────────────────────────────────────
    lr_m  = next(m for m in dom_metrics if m["label"] == "Logistic Regression")
    st.markdown("#### False Positives and False Negatives")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="card">
        <b>Logistic Regression</b><br>
        <span style="color:#64748B;font-size:0.85rem">
        FP = {lr_m['fp']} &nbsp;|&nbsp; FN = {lr_m['fn']}<br>
        TP = {lr_m['tp']} &nbsp;|&nbsp; TN = {lr_m['tn']}
        </span>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="card">
        <b>XGBoost</b><br>
        <span style="color:#64748B;font-size:0.85rem">
        FP = {xgb_m['fp']} &nbsp;|&nbsp; FN = {xgb_m['fn']}<br>
        TP = {xgb_m['tp']} &nbsp;|&nbsp; TN = {xgb_m['tn']}
        </span>
        </div>""", unsafe_allow_html=True)

    if IS_HC:
        st.markdown(
            f"A **false negative** here means a malignant tumour predicted as benign. "
            f"XGBoost has FN={xgb_m['fn']} vs LR FN={lr_m['fn']}. "
            f"A **false positive** means a benign case flagged as malignant (unnecessary follow-up). "
            f"XGBoost FP={xgb_m['fp']} vs LR FP={lr_m['fp']}."
        )
    else:
        st.markdown(
            f"A **false negative** means a defaulting borrower predicted as safe. "
            f"XGBoost FN={xgb_m['fn']} vs LR FN={lr_m['fn']}. "
            f"A **false positive** means a safe borrower flagged as a default risk. "
            f"XGBoost FP={xgb_m['fp']} vs LR FP={lr_m['fp']}."
        )

    # ── ROC and PR curves ──────────────────────────────────────────────────────
    st.markdown("#### ROC and Precision-Recall Curves")
    col_roc, col_pr = st.columns(2)
    with col_roc:
        show_figure(FIGURES_DIR / f"{prefix}_roc_curves.png", "ROC Curves")
    with col_pr:
        show_figure(FIGURES_DIR / f"{prefix}_pr_curves.png", "Precision-Recall Curves")

    # ── Classification report ──────────────────────────────────────────────────
    with st.expander("Full classification report — XGBoost"):
        st.code(xgb_m["classification_report"])
    with st.expander("Full classification report — Logistic Regression"):
        st.code(lr_m["classification_report"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Interactive Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Interactive Prediction":
    badge = '<span class="badge-hc">Healthcare</span>' if IS_HC else '<span class="badge-fin">Finance</span>'
    st.markdown(f'<p class="section-header">Interactive Prediction &nbsp;{badge}</p>', unsafe_allow_html=True)

    hc_bundle  = arts["hc_bundle"]
    fin_bundle = arts["fin_bundle"]

    # ── Shared helpers ─────────────────────────────────────────────────────────
    def _render_prediction(pred_class: int, prob: float, domain: str):
        if domain == "healthcare":
            label     = "🔴 Malignant" if pred_class == 1 else "🟢 Benign"
            css_class = "pred-positive" if pred_class == 1 else "pred-negative"
            interp    = ("The model output suggests features consistent with malignancy. "
                         if pred_class == 1 else
                         "The model output suggests features consistent with a benign case. ")
            disclaimer = ("⚠️ This result is for educational purposes only and is "
                          "not a medical diagnosis. Consult a qualified clinician.")
        else:
            label     = "🔴 Default" if pred_class == 1 else "🟢 No Default"
            css_class = "pred-positive" if pred_class == 1 else "pred-negative"
            interp    = ("The model output suggests features associated with loan default. "
                         if pred_class == 1 else
                         "The model output suggests features associated with loan repayment. ")
            disclaimer = ("⚠️ This result is for educational purposes only and does not "
                          "represent an official lending decision.")

        st.markdown(f"""
        <div class="{css_class}">
          <b style="font-size:1.1rem">{label}</b><br>
          <span style="color:#475569">Predicted probability (positive class): <b>{prob:.4f}</b></span><br>
          <span style="color:#475569;font-size:0.88rem">{interp}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="disclaimer">{disclaimer}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HEALTHCARE
    # ══════════════════════════════════════════════════════════════════════════
    if IS_HC:
        mode = st.radio("Input mode", ["Demonstration (test-set example)", "Manual input"],
                        horizontal=True)

        feat_names = hc_bundle["feature_names"]   # ['x.radius_mean', ...]
        disp_names = hc_display(feat_names)        # ['radius_mean', ...]
        X_test_raw = hc_bundle["X_test"]           # raw DataFrame

        if mode == "Demonstration (test-set example)":
            n_test = len(X_test_raw)
            idx = st.slider("Test-set sample index", 0, n_test - 1, 1,
                            help="Select a held-out test record.")
            row_raw = X_test_raw.iloc[[idx]]
            actual  = int(hc_bundle["y_test"].iloc[idx])
            st.markdown(f"**Actual label:** {'🔴 Malignant' if actual == 1 else '🟢 Benign'} "
                        f"(test sample {idx})")
            with st.expander("Raw feature values for this sample"):
                st.dataframe(row_raw.T.rename(columns={row_raw.index[0]: "value"})
                             .rename(index=lambda x: x.replace("x.", "")),
                             use_container_width=True)
        else:
            st.markdown("Enter values for all 30 features. "
                        "Default values are the training-set medians.")
            X_train_raw = hc_bundle["X_train"]
            medians = X_train_raw.median()

            groups = {
                "Mean measurements":  [f for f in feat_names if f.endswith("_mean")],
                "SE measurements":    [f for f in feat_names if f.endswith("_se")],
                "Worst measurements": [f for f in feat_names if f.endswith("_worst")],
            }
            input_vals = {}
            for grp_name, grp_feats in groups.items():
                with st.expander(grp_name, expanded=(grp_name == "Mean measurements")):
                    cols = st.columns(2)
                    for i, feat in enumerate(grp_feats):
                        dname = feat.replace("x.", "")
                        val = cols[i % 2].number_input(
                            dname,
                            value=float(round(medians[feat], 5)),
                            format="%.5f",
                            key=f"hc_{feat}",
                        )
                        input_vals[feat] = val
            row_raw = pd.DataFrame([input_vals])

        if st.button("Run Prediction", type="primary", key="hc_predict"):
            try:
                X_proc = arts["hc_pipe"].transform(row_raw)
                pred   = int(arts["hc_xgb"].predict(X_proc)[0])
                prob   = float(arts["hc_xgb"].predict_proba(X_proc)[0, 1])
                _render_prediction(pred, prob, "healthcare")

                # Local SHAP for this exact input
                st.markdown("#### SHAP Contribution for This Prediction")
                ex  = shap.TreeExplainer(arts["hc_xgb"])
                sv  = ex(X_proc)
                if sv.values.ndim == 3:
                    sv_vals = sv.values[0, :, 1]
                    sv_base = float(sv.base_values[0, 1]) if sv.base_values.ndim == 2 else float(sv.base_values[0])
                else:
                    sv_vals = sv.values[0]
                    sv_base = float(sv.base_values[0])

                order = np.argsort(np.abs(sv_vals))[::-1][:12]
                contrib_df = pd.DataFrame({
                    "Feature":       [disp_names[i] for i in order],
                    "Feature Value": [round(float(X_proc[0, i]), 4) for i in order],
                    "SHAP Value":    [round(float(sv_vals[i]), 4) for i in order],
                    "Direction":     ["↑ increases risk" if sv_vals[i] > 0
                                      else "↓ decreases risk" for i in order],
                })
                st.dataframe(contrib_df, hide_index=True, use_container_width=True)
                st.caption("SHAP values are in log-odds space — not probabilities. "
                           "They show how each feature pushed this prediction relative "
                           "to the model's average output.")

                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#EF4444" if v > 0 else "#3B82F6" for v in sv_vals[order][::-1]]
                ax.barh([disp_names[i] for i in order[::-1]],
                        sv_vals[order[::-1]], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (log-odds contribution)")
                ax.set_title("Local Feature Contributions")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # FINANCE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        mode = st.radio("Input mode", ["Demonstration (test-set example)", "Manual input"],
                        horizontal=True)

        num_feats  = fin_bundle["numeric_features"]
        cat_feats  = fin_bundle["categorical_features"]
        disp_names = fin_display(fin_bundle["transformed_feature_names"])
        X_test_raw = fin_bundle["X_test"]

        HOME_OPTS   = ["MORTGAGE", "OWN", "RENT", "OTHER"]
        INTENT_OPTS = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                       "MEDICAL", "PERSONAL", "VENTURE"]
        GRADE_OPTS  = ["A", "B", "C", "D", "E", "F", "G"]
        DEFAULT_OPTS = ["N", "Y"]

        if mode == "Demonstration (test-set example)":
            n_test = len(X_test_raw)
            idx = st.slider("Test-set sample index", 0, n_test - 1, 4,
                            help="Select a held-out test record.")
            row_raw = X_test_raw.iloc[[idx]]
            actual  = int(fin_bundle["y_test"].iloc[idx])
            st.markdown(f"**Actual label:** {'🔴 Default' if actual == 1 else '🟢 No Default'} "
                        f"(test sample {idx})")
            with st.expander("Raw feature values for this sample"):
                st.dataframe(row_raw.T.rename(columns={row_raw.index[0]: "value"}),
                             use_container_width=True)
        else:
            st.markdown("Enter applicant and loan details.")
            X_train_raw = fin_bundle["X_train"]
            num_medians = X_train_raw[num_feats].median()

            c1, c2 = st.columns(2)
            age      = c1.number_input("Age",            min_value=18, max_value=100,
                                       value=int(num_medians["person_age"]))
            income   = c2.number_input("Annual Income ($)", min_value=0,
                                       value=int(num_medians["person_income"]), step=1000)
            emp_len  = c1.number_input("Employment Length (years)", min_value=0.0,
                                       max_value=60.0,
                                       value=float(round(num_medians["person_emp_length"], 1)),
                                       step=0.5)
            loan_amt = c2.number_input("Loan Amount ($)", min_value=500, max_value=35000,
                                       value=int(num_medians["loan_amnt"]), step=500)
            int_rate = c1.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0,
                                       value=float(round(num_medians["loan_int_rate"], 2)),
                                       step=0.1)
            pct_inc  = c2.number_input("Loan % of Income", min_value=0.0, max_value=1.0,
                                       value=float(round(num_medians["loan_percent_income"], 3)),
                                       step=0.01)
            cred_len = c1.number_input("Credit History Length (years)", min_value=0,
                                       max_value=30,
                                       value=int(num_medians["cb_person_cred_hist_length"]))
            home_own = c2.selectbox("Home Ownership", HOME_OPTS)
            intent   = c1.selectbox("Loan Intent",    INTENT_OPTS)
            grade    = c2.selectbox("Loan Grade",     GRADE_OPTS)
            default_file = c1.selectbox("Previous Default on File", DEFAULT_OPTS)

            row_raw = pd.DataFrame([{
                "person_age":                age,
                "person_income":             income,
                "person_emp_length":         emp_len,
                "loan_amnt":                 loan_amt,
                "loan_int_rate":             int_rate,
                "loan_percent_income":       pct_inc,
                "cb_person_cred_hist_length": cred_len,
                "person_home_ownership":     home_own,
                "loan_intent":               intent,
                "loan_grade":                grade,
                "cb_person_default_on_file": default_file,
            }])

        if st.button("Run Prediction", type="primary", key="fin_predict"):
            try:
                X_proc = arts["fin_pipe"].transform(row_raw)
                pred   = int(arts["fin_xgb"].predict(X_proc)[0])
                prob   = float(arts["fin_xgb"].predict_proba(X_proc)[0, 1])
                _render_prediction(pred, prob, "finance")

                st.markdown("#### SHAP Contribution for This Prediction")
                ex  = shap.TreeExplainer(arts["fin_xgb"])
                sv  = ex(X_proc)
                if sv.values.ndim == 3:
                    sv_vals = sv.values[0, :, 1]
                    sv_base = float(sv.base_values[0, 1]) if sv.base_values.ndim == 2 else float(sv.base_values[0])
                else:
                    sv_vals = sv.values[0]
                    sv_base = float(sv.base_values[0])

                order = np.argsort(np.abs(sv_vals))[::-1][:12]
                contrib_df = pd.DataFrame({
                    "Feature":       [disp_names[i] for i in order],
                    "Feature Value": [round(float(X_proc[0, i]), 4) for i in order],
                    "SHAP Value":    [round(float(sv_vals[i]), 4) for i in order],
                    "Direction":     ["↑ increases default risk" if sv_vals[i] > 0
                                      else "↓ decreases default risk" for i in order],
                })
                st.dataframe(contrib_df, hide_index=True, use_container_width=True)
                st.caption("SHAP values are in log-odds space — not probabilities.")

                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#EF4444" if v > 0 else "#3B82F6" for v in sv_vals[order][::-1]]
                ax.barh([disp_names[i] for i in order[::-1]],
                        sv_vals[order[::-1]], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (log-odds contribution)")
                ax.set_title("Local Feature Contributions")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    badge = '<span class="badge-hc">Healthcare</span>' if IS_HC else '<span class="badge-fin">Finance</span>'
    st.markdown(f'<p class="section-header">SHAP Explainability &nbsp;{badge}</p>', unsafe_allow_html=True)

    st.info(
        "SHAP values are in **log-odds (margin) space** — they are not probabilities. "
        "A positive SHAP value means the feature pushed the model output toward the "
        "positive class. A negative value means it pushed toward the negative class. "
        "SHAP measures model attribution, not causation."
    )

    prefix = "healthcare" if IS_HC else "finance"
    bundle = arts["hc_bundle"] if IS_HC else arts["fin_bundle"]
    model  = arts["hc_xgb"]   if IS_HC else arts["fin_xgb"]
    pipe   = arts["hc_pipe"]  if IS_HC else arts["fin_pipe"]

    raw_names = bundle["feature_names"] if IS_HC else bundle["transformed_feature_names"]
    disp      = hc_display(raw_names) if IS_HC else fin_display(raw_names)
    pos_label = "Malignant" if IS_HC else "Default"
    neg_label = "Benign"    if IS_HC else "No Default"

    # ── Global explanations ────────────────────────────────────────────────────
    st.markdown("### Global Explanations (full test set)")
    st.markdown(
        "Global importance shows which features the model relied on most across all "
        "test predictions. It does not mean every feature always increases or decreases "
        "the prediction — direction varies per sample."
    )

    tab_bee, tab_bar, tab_dep = st.tabs(
        ["Beeswarm Plot", "Feature Importance Bar", "Dependence Plot"]
    )
    with tab_bee:
        show_figure(FIGURES_DIR / f"{prefix}_shap_beeswarm.png",
                    f"Each dot is one test sample. Colour = feature value (red=high, blue=low). "
                    f"X-axis = SHAP value (impact on {pos_label} prediction).")
    with tab_bar:
        show_figure(FIGURES_DIR / f"{prefix}_shap_importance.png",
                    "Mean absolute SHAP value per feature — higher = more influential globally.")
        if not shap_imp.empty:
            dom_key = "healthcare" if IS_HC else "finance"
            top_df = (shap_imp[(shap_imp["domain"] == dom_key) &
                               (shap_imp["model"] == "XGBoost")]
                      .head(10)[["rank", "feature", "mean_abs_shap"]])
            st.dataframe(top_df, hide_index=True, use_container_width=True)
    with tab_dep:
        show_figure(FIGURES_DIR / f"{prefix}_shap_dependence.png",
                    "SHAP value vs raw feature value for the top feature. "
                    "Shows how the model's attribution changes across the feature's range.")

    # ── Local explanations ─────────────────────────────────────────────────────
    st.markdown("### Local Explanation — Individual Prediction")
    st.markdown(
        "Select a test-set sample to see exactly which features drove that specific prediction."
    )

    X_test_proc = bundle["X_test_processed"]
    X_test_raw  = bundle["X_test"]
    y_test      = bundle["y_test"].to_numpy()
    n_test      = len(y_test)

    idx = st.slider("Test-set sample index", 0, n_test - 1, 1, key="shap_local_idx")

    y_pred_all = model.predict(X_test_proc)
    y_prob_all = model.predict_proba(X_test_proc)[:, 1]
    actual     = int(y_test[idx])
    predicted  = int(y_pred_all[idx])
    prob       = float(y_prob_all[idx])

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_tile(pos_label if actual == 1 else neg_label, "Actual Class"),
                unsafe_allow_html=True)
    c2.markdown(metric_tile(pos_label if predicted == 1 else neg_label, "Predicted Class"),
                unsafe_allow_html=True)
    c3.markdown(metric_tile(f"{prob:.4f}", f"P({pos_label})"), unsafe_allow_html=True)
    st.markdown("")

    correct = actual == predicted
    st.markdown(
        f"{'✅ Correctly classified' if correct else '❌ Misclassified'} — "
        f"Actual: **{pos_label if actual==1 else neg_label}** | "
        f"Predicted: **{pos_label if predicted==1 else neg_label}**"
    )

    # Compute SHAP for this single sample
    with st.spinner("Computing SHAP values for selected sample…"):
        ex = shap.TreeExplainer(model)
        sv_single = ex(X_test_proc[[idx]])
        if sv_single.values.ndim == 3:
            sv_vals = sv_single.values[0, :, 1]
            sv_base = float(sv_single.base_values[0, 1]) if sv_single.base_values.ndim == 2 else float(sv_single.base_values[0])
        else:
            sv_vals = sv_single.values[0]
            sv_base = float(sv_single.base_values[0])

    # Waterfall plot
    st.markdown("#### Waterfall Plot")
    single_exp = shap.Explanation(
        values=sv_vals,
        base_values=sv_base,
        data=X_test_proc[idx],
        feature_names=disp,
    )
    fig_wf, _ = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(single_exp, max_display=15, show=False)
    plt.title(f"Sample {idx} — {pos_label if predicted==1 else neg_label} (prob={prob:.4f})",
              pad=8)
    fig_wf = plt.gcf()
    st.pyplot(fig_wf, use_container_width=True)
    plt.close(fig_wf)

    # Contribution table
    st.markdown("#### Top Feature Contributions")
    order = np.argsort(np.abs(sv_vals))[::-1][:12]
    contrib_df = pd.DataFrame({
        "Feature":       [disp[i] for i in order],
        "SHAP Value":    [round(float(sv_vals[i]), 4) for i in order],
        "Direction":     [f"↑ toward {pos_label}" if sv_vals[i] > 0
                          else f"↓ toward {neg_label}" for i in order],
    })
    st.dataframe(contrib_df, hide_index=True, use_container_width=True)

    # Plain-language interpretation
    top_pos = [(disp[i], sv_vals[i]) for i in order if sv_vals[i] > 0][:3]
    top_neg = [(disp[i], sv_vals[i]) for i in order if sv_vals[i] < 0][:3]
    interp_lines = []
    if top_pos:
        interp_lines.append(
            "Features that **pushed the model output toward " + pos_label + "**: " +
            ", ".join(f"`{f}` (SHAP={v:+.3f})" for f, v in top_pos) + "."
        )
    if top_neg:
        interp_lines.append(
            "Features that **pushed the model output toward " + neg_label + "**: " +
            ", ".join(f"`{f}` (SHAP={v:+.3f})" for f, v in top_neg) + "."
        )
    interp_lines.append(
        "These attributions reflect the model's internal weighting for this specific "
        "sample. They do not establish that any feature caused the outcome."
    )
    st.markdown("\n\n".join(interp_lines))

    # Saved global waterfall for reference
    with st.expander("Reference: saved global waterfall (Stage 4 example)"):
        show_figure(FIGURES_DIR / f"{prefix}_shap_waterfall.png",
                    "Saved from Stage 4 — may differ from the sample selected above.")

    # LR comparison
    with st.expander("Logistic Regression SHAP importance (for comparison)"):
        show_figure(FIGURES_DIR / f"{prefix}_lr_shap_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<p class="section-header">Model Comparison — Baseline vs Advanced</p>',
                unsafe_allow_html=True)

    for domain_label, domain_key, dm in [
        ("Healthcare — Breast Cancer", "healthcare", hc_metrics),
        ("Finance — Credit Risk",      "finance",    fin_metrics),
    ]:
        badge_cls = "badge-hc" if domain_key == "healthcare" else "badge-fin"
        st.markdown(f'<span class="{badge_cls}">{domain_label}</span>', unsafe_allow_html=True)

        lr_m  = next(m for m in dm if m["label"] == "Logistic Regression")
        xgb_m = next(m for m in dm if m["label"] == "XGBoost")

        comp_df = pd.DataFrame({
            "Metric":              ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                                    "CV ROC-AUC", "FP", "FN"],
            "Logistic Regression": [lr_m["accuracy"],  lr_m["precision"],  lr_m["recall"],
                                    lr_m["f1"],         lr_m["roc_auc"],
                                    f"{lr_m['cv_roc_auc_mean']:.4f}±{lr_m['cv_roc_auc_std']:.4f}",
                                    lr_m["fp"],         lr_m["fn"]],
            "XGBoost":             [xgb_m["accuracy"], xgb_m["precision"], xgb_m["recall"],
                                    xgb_m["f1"],        xgb_m["roc_auc"],
                                    f"{xgb_m['cv_roc_auc_mean']:.4f}±{xgb_m['cv_roc_auc_std']:.4f}",
                                    xgb_m["fp"],        xgb_m["fn"]],
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        prefix = "healthcare" if domain_key == "healthcare" else "finance"
        col_roc, col_pr = st.columns(2)
        with col_roc:
            show_figure(FIGURES_DIR / f"{prefix}_roc_curves.png", "ROC Curves")
        with col_pr:
            show_figure(FIGURES_DIR / f"{prefix}_pr_curves.png", "PR Curves")

        if domain_key == "healthcare":
            st.markdown(
                "Both models perform strongly on this small, clean dataset. "
                f"XGBoost achieves **zero false positives** (FP={xgb_m['fp']}) vs "
                f"LR FP={lr_m['fp']}, and a higher F1 ({xgb_m['f1']} vs {lr_m['f1']}). "
                "Recall is identical (0.9524) — both miss 2 malignant cases. "
                "LR's ROC-AUC (0.9954) is marginally higher than XGBoost's (0.9917), "
                "which is notable: linear models can be competitive on well-scaled numeric data."
            )
        else:
            st.markdown(
                f"XGBoost is the clear winner in finance. It reduces false positives from "
                f"{lr_m['fp']} to {xgb_m['fp']} while improving default recall "
                f"({lr_m['recall']} → {xgb_m['recall']}). "
                f"LR's 80.9% accuracy hides poor default-class precision (0.545) — "
                f"it generates {lr_m['fp']} false alarms. "
                f"XGBoost's ROC-AUC of {xgb_m['roc_auc']} vs {lr_m['roc_auc']} "
                "reflects its ability to capture non-linear interactions between "
                "loan_grade, loan_int_rate and loan_percent_income."
            )
        st.markdown("---")

    # SHAP comparison
    st.markdown("### SHAP Feature Importance Comparison")
    if not shap_imp.empty:
        for domain_key in ["healthcare", "finance"]:
            st.markdown(f"**{domain_key.capitalize()} — Top 10 features**")
            sub = shap_imp[shap_imp["domain"] == domain_key].pivot_table(
                index=["rank", "feature"], columns="model",
                values="mean_abs_shap", aggfunc="first"
            ).reset_index().head(10)
            st.dataframe(sub, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — About & Limitations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About & Limitations":
    st.markdown('<p class="section-header">About & Limitations</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Project</b><br>
    Explainable Machine Learning for Interpreting Black-Box Predictions Using SHAP<br>
    Advanced Machine Learning Lab — Academic Project<br><br>
    <b>Stack</b><br>
    Python 3.12 · scikit-learn 1.9 · XGBoost 3.4 · SHAP 0.53 · Streamlit 1.65 ·
    Optuna 5.0 · pandas · numpy · matplotlib
    </div>""", unsafe_allow_html=True)

    st.markdown("### Responsible Use")
    st.markdown("""
    <div class="disclaimer">
    <b>Healthcare disclaimer:</b> The breast cancer predictions and SHAP explanations
    produced by this dashboard are for <b>educational and research purposes only</b>.
    They are not a medical diagnosis, clinical recommendation or substitute for
    professional medical advice. Do not use these outputs to make health decisions.
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="disclaimer">
    <b>Finance disclaimer:</b> The credit risk predictions and SHAP explanations
    produced by this dashboard are for <b>educational and research purposes only</b>.
    They do not represent official lending decisions, credit assessments or financial
    advice. Do not use these outputs to approve or deny credit applications.
    </div>""", unsafe_allow_html=True)

    st.markdown("### Technical Limitations")
    limitations = [
        ("Small healthcare dataset", "569 samples across 30 numeric cell-nucleus measurements. Results may not generalise to other populations or clinical settings. The dataset is a well-studied benchmark (UCI Wisconsin) and is not representative of a production diagnostic pipeline."),
        ("SHAP is attribution, not causation", "A high SHAP value means the model weighted that feature heavily for this prediction. It does not mean the feature caused the outcome. Do not recommend changing feature values to influence a prediction."),
        ("SHAP values are in log-odds space", "For XGBoost with TreeExplainer, SHAP values are in raw margin (log-odds) space. They are not probabilities and should not be interpreted as percentage contributions. The base value shown (~0.33 for healthcare, ~0.02 for finance) is also in log-odds space, not a probability."),
        ("Feature leakage risk in finance", "loan_grade and loan_int_rate are assigned by lenders and may partially encode the lender's own risk assessment. This could inflate model performance metrics, since these features may not be available at the time a lending decision is made."),
        ("Models are not calibrated", "Predicted probabilities have not been post-hoc calibrated (e.g. via CalibratedClassifierCV with isotonic regression). The finance XGBoost model overestimates the mean predicted probability (~0.31) relative to the actual default rate (~0.22), confirmed by a Brier score of 0.069. Probabilities should not be treated as precise risk estimates."),
        ("No temporal validation", "Neither dataset contains a timestamp column, so no out-of-time validation was performed. Model performance may degrade on data from different time periods due to distribution shift."),
        ("Class imbalance", "The finance dataset has a 78/22 class split (no-default/default). Both models address this during training — XGBoost uses scale_pos_weight=3.57 and Logistic Regression uses class_weight='balanced'. However, threshold-dependent metrics such as accuracy still reflect the majority class and can be misleading; recall and F1 for the default class are more informative."),
        ("No retraining on interaction", "Models are loaded once at startup and never retrained during dashboard use. Predictions are deterministic given the same inputs. This is expected behaviour for a deployed model, not an error."),
    ]
    for title, body in limitations:
        with st.expander(f"⚠️ {title}"):
            st.markdown(body)

    st.markdown("### Methodology Summary")
    st.markdown("""
    | Step | Detail |
    |---|---|
    | Data cleaning | Duplicates dropped, age capped, impossible employment rows removed — all before split |
    | Train/test split | Stratified 80/20, shuffle=True, random_state=42 |
    | Preprocessing | Fit on training data only — no test-set leakage |
    | Tuning | Optuna TPE sampler, 30 trials, 5-fold CV on training set |
    | Evaluation | All metrics computed on held-out test set only |
    | SHAP | TreeExplainer (exact) for XGBoost; LinearExplainer for LR |
    | Reproducibility | random_state=42 throughout |
    """)

    st.markdown("### References")
    st.markdown("""
    - Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
    - Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
    - Wolberg, W. H., et al. (1995). Breast cancer Wisconsin dataset. UCI ML Repository.
    """)
