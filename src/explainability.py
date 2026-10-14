"""
Stage 4 — SHAP explainability module.

Supports both domains (healthcare / finance) with:
  - TreeExplainer  for XGBoost models
  - LinearExplainer for Logistic Regression models

All SHAP values are for the POSITIVE class:
  Healthcare : class 1 = Malignant
  Finance    : class 1 = Default (loan_status = 1)

SHAP values represent additive contributions to the model's raw log-odds
output (margin space). They are NOT probabilities and do NOT establish
causation. Changing a feature value will not necessarily change a real
outcome.

DISCLAIMER — Healthcare: explanations are educational only and are not a
medical diagnosis. Finance: explanations are educational only and do not
represent official lending decisions.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import csv
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import (
    FIGURES_DIR,
    HC_LR_PATH,
    HC_XGB_PATH,
    FIN_LR_PATH,
    FIN_XGB_PATH,
    HEALTHCARE_SPLIT_PATH,
    FINANCE_SPLIT_PATH,
    REPORTS_DIR,
    ensure_project_dirs,
)

# ── Feature-name helpers ───────────────────────────────────────────────────────

def _clean_fin_names(raw: list[str]) -> list[str]:
    """Strip sklearn ColumnTransformer prefixes for readable plot labels."""
    return [n.replace("num__", "").replace("cat__", "") for n in raw]


def _hc_display_names(raw: list[str]) -> list[str]:
    """Strip 'x.' prefix from breast-cancer feature names."""
    return [n.replace("x.", "") for n in raw]


# ── Explainer factories ────────────────────────────────────────────────────────

def make_tree_explainer(model, X_background=None):
    """Return a SHAP TreeExplainer; background data is optional for XGBoost."""
    return shap.TreeExplainer(model)


def make_linear_explainer(model, X_background):
    """Return a SHAP LinearExplainer with a capped background sample."""
    masker = shap.maskers.Independent(X_background, max_samples=200)
    return shap.LinearExplainer(model, masker)


# ── Core computation ───────────────────────────────────────────────────────────

def compute_shap_values(explainer, X: np.ndarray) -> shap.Explanation:
    """
    Compute SHAP Explanation object.
    Returns an Explanation with .values shape (n_samples, n_features).
    """
    sv = explainer(X)
    # Guard: some explainers return 3-D arrays for multi-class; take class-1 slice
    if sv.values.ndim == 3:
        sv = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=sv.data,
            feature_names=sv.feature_names,
        )
    return sv


def global_importance(sv: shap.Explanation, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of features ranked by mean |SHAP value|."""
    mean_abs = np.abs(sv.values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int = 130) -> Path:
    ensure_project_dirs()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_beeswarm(sv: shap.Explanation, display_names: list[str],
                  title: str, out_path: Path) -> Path:
    sv_named = shap.Explanation(
        values=sv.values,
        base_values=sv.base_values,
        data=sv.data,
        feature_names=display_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(sv_named, max_display=15, show=False)
    plt.title(title, pad=10)
    fig = plt.gcf()
    return _savefig(fig, out_path)


def plot_bar_importance(importance_df: pd.DataFrame, title: str,
                        out_path: Path, top_n: int = 15) -> Path:
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP value|  (log-odds contribution)")
    ax.set_title(title)
    fig.tight_layout()
    return _savefig(fig, out_path)


def plot_waterfall(sv: shap.Explanation, idx: int, display_names: list[str],
                   title: str, out_path: Path) -> Path:
    single = shap.Explanation(
        values=sv.values[idx],
        base_values=float(sv.base_values[idx]),
        data=sv.data[idx],
        feature_names=display_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(single, max_display=15, show=False)
    plt.title(title, pad=10)
    fig = plt.gcf()
    return _savefig(fig, out_path)


def plot_dependence(sv: shap.Explanation, feature_name: str,
                    display_names: list[str], title: str, out_path: Path) -> Path:
    """Scatter SHAP value vs raw feature value for one feature."""
    if feature_name not in display_names:
        raise ValueError(f"Feature '{feature_name}' not in display_names.")
    idx = display_names.index(feature_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sv.data[:, idx], sv.values[:, idx],
               alpha=0.3, s=8, color="#E91E63")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"SHAP value for {feature_name}")
    ax.set_title(title)
    fig.tight_layout()
    return _savefig(fig, out_path)


def plot_local_bar(sv: shap.Explanation, idx: int, display_names: list[str],
                   title: str, out_path: Path, top_n: int = 12) -> Path:
    """Horizontal bar chart of signed SHAP contributions for one sample."""
    vals = sv.values[idx]
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    feats = [display_names[i] for i in order]
    contributions = vals[order]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in contributions]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feats[::-1], contributions[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (log-odds contribution to positive class)")
    ax.set_title(title)
    fig.tight_layout()
    return _savefig(fig, out_path)


# ── Example selector ──────────────────────────────────────────────────────────

def pick_examples(y_true, y_pred, y_prob) -> dict[str, int]:
    """
    Return indices for:
      - correct_positive : TP (actual=1, predicted=1)
      - correct_negative : TN (actual=0, predicted=0)
      - misclassified    : FN or FP (first found)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]

    examples = {}
    if len(tp): examples["correct_positive"] = int(tp[0])
    if len(tn): examples["correct_negative"] = int(tn[0])
    if len(fn): examples["misclassified"]    = int(fn[0])
    elif len(fp): examples["misclassified"]  = int(fp[0])
    return examples


# ── Local explanation dict (for Streamlit / FastAPI) ─────────────────────────

def local_explanation_dict(sv: shap.Explanation, idx: int,
                            display_names: list[str],
                            y_true, y_pred, y_prob,
                            top_n: int = 10) -> dict:
    """Return a structured dict describing one prediction's SHAP breakdown."""
    vals = sv.values[idx]
    data = sv.data[idx]
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    contributions = [
        {
            "feature":      display_names[i],
            "feature_value": round(float(data[i]), 4),
            "shap_value":   round(float(vals[i]), 4),
            "direction":    "increases risk" if vals[i] > 0 else "decreases risk",
        }
        for i in order
    ]
    return {
        "sample_index":   idx,
        "actual_class":   int(y_true[idx]),
        "predicted_class": int(y_pred[idx]),
        "predicted_prob": round(float(y_prob[idx]), 4),
        "base_value":     round(float(sv.base_values[idx]), 4),
        "top_contributions": contributions,
    }


# ── Domain runners ────────────────────────────────────────────────────────────

def run_healthcare() -> dict:
    ensure_project_dirs()
    bundle   = joblib.load(HEALTHCARE_SPLIT_PATH)
    xgb_model = joblib.load(HC_XGB_PATH)
    lr_model  = joblib.load(HC_LR_PATH)

    X_te      = bundle["X_test_processed"]
    y_te      = bundle["y_test"].to_numpy()
    raw_names = bundle["feature_names"]          # ['x.radius_mean', ...]
    disp      = _hc_display_names(raw_names)     # ['radius_mean', ...]

    # ── XGBoost SHAP ──────────────────────────────────────────────────────────
    ex_xgb = make_tree_explainer(xgb_model)
    sv_xgb = compute_shap_values(ex_xgb, X_te)

    y_pred_xgb = xgb_model.predict(X_te)
    y_prob_xgb = xgb_model.predict_proba(X_te)[:, 1]

    imp_xgb = global_importance(sv_xgb, disp)

    # Beeswarm
    plot_beeswarm(sv_xgb, disp,
                  "Healthcare XGBoost — SHAP Beeswarm (Malignant class)",
                  FIGURES_DIR / "healthcare_shap_beeswarm.png")

    # Bar importance
    plot_bar_importance(imp_xgb,
                        "Healthcare XGBoost — Global Feature Importance (SHAP)",
                        FIGURES_DIR / "healthcare_shap_importance.png")

    # Dependence — top feature
    top_feat = imp_xgb.iloc[0]["feature"]
    plot_dependence(sv_xgb, top_feat, disp,
                    f"Healthcare XGBoost — SHAP Dependence: {top_feat}",
                    FIGURES_DIR / "healthcare_shap_dependence.png")

    # Individual examples
    examples = pick_examples(y_te, y_pred_xgb, y_prob_xgb)
    local_records = []
    waterfall_idx = examples.get("correct_positive",
                    examples.get("correct_negative", 0))

    plot_waterfall(sv_xgb, waterfall_idx, disp,
                   f"Healthcare XGBoost — Waterfall (sample {waterfall_idx})",
                   FIGURES_DIR / "healthcare_shap_waterfall.png")

    plot_local_bar(sv_xgb, waterfall_idx, disp,
                   f"Healthcare XGBoost — Local Explanation (sample {waterfall_idx})",
                   FIGURES_DIR / "healthcare_local_explanation.png")

    for role, idx in examples.items():
        rec = local_explanation_dict(sv_xgb, idx, disp, y_te, y_pred_xgb, y_prob_xgb)
        rec["role"] = role
        rec["domain"] = "healthcare"
        rec["model"] = "XGBoost"
        local_records.append(rec)

    # ── LR SHAP ───────────────────────────────────────────────────────────────
    ex_lr  = make_linear_explainer(lr_model, bundle["X_train_processed"])
    sv_lr  = compute_shap_values(ex_lr, X_te)
    imp_lr = global_importance(sv_lr, disp)

    y_pred_lr = lr_model.predict(X_te)
    y_prob_lr = lr_model.predict_proba(X_te)[:, 1]

    plot_bar_importance(imp_lr,
                        "Healthcare LR — Global Feature Importance (SHAP)",
                        FIGURES_DIR / "healthcare_lr_shap_importance.png")

    print(f"[Healthcare] Top-5 XGBoost SHAP features:")
    for _, row in imp_xgb.head(5).iterrows():
        print(f"  {row['feature']:<30} mean|SHAP|={row['mean_abs_shap']:.4f}")

    return {
        "domain": "healthcare",
        "xgb_importance": imp_xgb,
        "lr_importance":  imp_lr,
        "local_records":  local_records,
        "sv_xgb":         sv_xgb,
        "sv_lr":          sv_lr,
        "display_names":  disp,
        "y_te": y_te, "y_pred_xgb": y_pred_xgb, "y_prob_xgb": y_prob_xgb,
    }


def run_finance() -> dict:
    ensure_project_dirs()
    bundle    = joblib.load(FINANCE_SPLIT_PATH)
    xgb_model = joblib.load(FIN_XGB_PATH)
    lr_model  = joblib.load(FIN_LR_PATH)

    X_te      = bundle["X_test_processed"]
    y_te      = bundle["y_test"].to_numpy()
    raw_names = bundle["transformed_feature_names"]
    disp      = _clean_fin_names(raw_names)

    # ── XGBoost SHAP ──────────────────────────────────────────────────────────
    ex_xgb = make_tree_explainer(xgb_model)
    sv_xgb = compute_shap_values(ex_xgb, X_te)

    y_pred_xgb = xgb_model.predict(X_te)
    y_prob_xgb = xgb_model.predict_proba(X_te)[:, 1]

    imp_xgb = global_importance(sv_xgb, disp)

    # Beeswarm
    plot_beeswarm(sv_xgb, disp,
                  "Finance XGBoost — SHAP Beeswarm (Default class)",
                  FIGURES_DIR / "finance_shap_beeswarm.png")

    # Bar importance
    plot_bar_importance(imp_xgb,
                        "Finance XGBoost — Global Feature Importance (SHAP)",
                        FIGURES_DIR / "finance_shap_importance.png")

    # Dependence — top feature
    top_feat = imp_xgb.iloc[0]["feature"]
    plot_dependence(sv_xgb, top_feat, disp,
                    f"Finance XGBoost — SHAP Dependence: {top_feat}",
                    FIGURES_DIR / "finance_shap_dependence.png")

    # Individual examples
    examples = pick_examples(y_te, y_pred_xgb, y_prob_xgb)
    local_records = []
    waterfall_idx = examples.get("correct_positive",
                    examples.get("correct_negative", 0))

    plot_waterfall(sv_xgb, waterfall_idx, disp,
                   f"Finance XGBoost — Waterfall (sample {waterfall_idx})",
                   FIGURES_DIR / "finance_shap_waterfall.png")

    plot_local_bar(sv_xgb, waterfall_idx, disp,
                   f"Finance XGBoost — Local Explanation (sample {waterfall_idx})",
                   FIGURES_DIR / "finance_local_explanation.png")

    for role, idx in examples.items():
        rec = local_explanation_dict(sv_xgb, idx, disp, y_te, y_pred_xgb, y_prob_xgb)
        rec["role"] = role
        rec["domain"] = "finance"
        rec["model"] = "XGBoost"
        local_records.append(rec)

    # ── LR SHAP ───────────────────────────────────────────────────────────────
    ex_lr  = make_linear_explainer(lr_model, bundle["X_train_processed"])
    sv_lr  = compute_shap_values(ex_lr, X_te)
    imp_lr = global_importance(sv_lr, disp)

    y_pred_lr = lr_model.predict(X_te)
    y_prob_lr = lr_model.predict_proba(X_te)[:, 1]

    plot_bar_importance(imp_lr,
                        "Finance LR — Global Feature Importance (SHAP)",
                        FIGURES_DIR / "finance_lr_shap_importance.png")

    print(f"[Finance] Top-5 XGBoost SHAP features:")
    for _, row in imp_xgb.head(5).iterrows():
        print(f"  {row['feature']:<35} mean|SHAP|={row['mean_abs_shap']:.4f}")

    return {
        "domain": "finance",
        "xgb_importance": imp_xgb,
        "lr_importance":  imp_lr,
        "local_records":  local_records,
        "sv_xgb":         sv_xgb,
        "sv_lr":          sv_lr,
        "display_names":  disp,
        "y_te": y_te, "y_pred_xgb": y_pred_xgb, "y_prob_xgb": y_prob_xgb,
    }


# ── Report writers ────────────────────────────────────────────────────────────

def save_global_importance_csv(hc_result: dict, fin_result: dict) -> Path:
    path = REPORTS_DIR / "shap_global_importance.csv"
    rows = []
    for res in (hc_result, fin_result):
        for model_key, imp_df in [("XGBoost", res["xgb_importance"]),
                                   ("Logistic Regression", res["lr_importance"])]:
            for rank, (_, row) in enumerate(imp_df.iterrows(), 1):
                rows.append({
                    "domain": res["domain"],
                    "model": model_key,
                    "rank": rank,
                    "feature": row["feature"],
                    "mean_abs_shap": round(row["mean_abs_shap"], 6),
                })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain","model","rank","feature","mean_abs_shap"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_local_examples_csv(hc_result: dict, fin_result: dict) -> Path:
    path = REPORTS_DIR / "shap_local_examples.csv"
    rows = []
    for res in (hc_result, fin_result):
        for rec in res["local_records"]:
            for contrib in rec["top_contributions"]:
                rows.append({
                    "domain":          rec["domain"],
                    "model":           rec["model"],
                    "role":            rec["role"],
                    "sample_index":    rec["sample_index"],
                    "actual_class":    rec["actual_class"],
                    "predicted_class": rec["predicted_class"],
                    "predicted_prob":  rec["predicted_prob"],
                    "feature":         contrib["feature"],
                    "feature_value":   contrib["feature_value"],
                    "shap_value":      contrib["shap_value"],
                    "direction":       contrib["direction"],
                })
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["domain","model","role","sample_index","actual_class",
                  "predicted_class","predicted_prob","feature",
                  "feature_value","shap_value","direction"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_interpretation_summary(hc_result: dict, fin_result: dict) -> Path:
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    path = REPORTS_DIR / "shap_interpretation_summary.txt"
    lines = [
        "Stage 4 — SHAP Interpretation Summary",
        "======================================",
        f"Generated: {ts}",
        "",
        "SHAP values represent additive contributions to the model's raw log-odds",
        "output (margin space). They are NOT probabilities and do NOT establish",
        "causation. A high SHAP value for a feature means the model attributed",
        "more weight to that feature for this prediction — it does not mean",
        "changing the feature will change the real-world outcome.",
        "",
        "DISCLAIMER — Healthcare: educational only, not a medical diagnosis.",
        "DISCLAIMER — Finance: educational only, not an official lending decision.",
        "",
    ]

    for res in (hc_result, fin_result):
        domain = res["domain"].capitalize()
        lines += [
            f"{domain} — XGBoost Top-10 Global Features (mean |SHAP|)",
            "-" * 55,
        ]
        for _, row in res["xgb_importance"].head(10).iterrows():
            lines.append(f"  {row['feature']:<35} {row['mean_abs_shap']:.6f}")
        lines.append("")

        lines += [f"{domain} — Local Examples", "-" * 55]
        for rec in res["local_records"]:
            pos_label = "Malignant" if res["domain"] == "healthcare" else "Default"
            neg_label = "Benign"    if res["domain"] == "healthcare" else "No Default"
            actual_str    = pos_label if rec["actual_class"]    == 1 else neg_label
            predicted_str = pos_label if rec["predicted_class"] == 1 else neg_label
            lines += [
                f"  Role: {rec['role']}  |  Sample: {rec['sample_index']}",
                f"  Actual: {actual_str}  |  Predicted: {predicted_str}"
                f"  |  Prob: {rec['predicted_prob']:.4f}",
                "  Top contributions:",
            ]
            for c in rec["top_contributions"][:5]:
                lines.append(
                    f"    {c['feature']:<30} SHAP={c['shap_value']:+.4f}"
                    f"  val={c['feature_value']:.3f}  ({c['direction']})"
                )
            lines.append("")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Top-level runner ──────────────────────────────────────────────────────────

def run_all() -> tuple[dict, dict]:
    ensure_project_dirs()
    print("=" * 60)
    print("Stage 4 — SHAP Explainability")
    print("=" * 60)
    hc_result  = run_healthcare()
    fin_result = run_finance()

    csv1 = save_global_importance_csv(hc_result, fin_result)
    csv2 = save_local_examples_csv(hc_result, fin_result)
    txt  = save_interpretation_summary(hc_result, fin_result)

    print(f"\nReports saved:")
    print(f"  {csv1}")
    print(f"  {csv2}")
    print(f"  {txt}")
    return hc_result, fin_result


if __name__ == "__main__":
    run_all()
