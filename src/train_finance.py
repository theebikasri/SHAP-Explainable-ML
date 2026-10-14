"""
Stage 3 — Finance (credit risk) model training.

Loads the verified Stage 2 split bundle (32,414 clean rows, 80/20 split).
Trains:
  1. Logistic Regression baseline  (class_weight='balanced')
  2. XGBoost  (scale_pos_weight from train labels, Optuna tuning)

Fits on X_train_processed / y_train only.
Evaluates on X_test_processed / y_test only.
loan_status=1 is the positive class (default).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
)
from src.utils import (
    FIN_LR_PATH,
    FIN_XGB_PATH,
    FINANCE_SPLIT_PATH,
    RANDOM_STATE,
    ensure_project_dirs,
)

DOMAIN = "Finance"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def _load_split() -> dict:
    bundle = joblib.load(FINANCE_SPLIT_PATH)
    assert bundle["domain"] == "finance"
    cl = bundle["cleaning"]
    assert cl["n_rows_after_cleaning"] == 32414, (
        f"Expected 32414 clean rows, got {cl['n_rows_after_cleaning']}. "
        "Re-run src.preprocessing first."
    )
    return bundle


def train_logistic_regression(bundle: dict) -> tuple[LogisticRegression, dict, object, object]:
    X_tr = bundle["X_train_processed"]
    y_tr = bundle["y_train"]
    X_te = bundle["X_test_processed"]
    y_te = bundle["y_test"]

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_tr, y_tr)

    cv_scores = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc")

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, y_pred, y_prob, "Logistic Regression")
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()),  4)
    metrics["best_params"] = {
        "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"
    }

    ensure_project_dirs()
    joblib.dump(model, FIN_LR_PATH)
    save_confusion_matrix(y_te, y_pred, "Logistic Regression", DOMAIN)
    return model, metrics, y_te, y_prob


def _xgb_objective(trial, X_tr, y_tr, scale_pos_weight: float) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight":  scale_pos_weight,
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      RANDOM_STATE,
        "verbosity":         0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc")
    return scores.mean()


def train_xgboost(bundle: dict) -> tuple[XGBClassifier, dict, object, object]:
    X_tr = bundle["X_train_processed"]
    y_tr = bundle["y_train"]
    X_te = bundle["X_test_processed"]
    y_te = bundle["y_test"]

    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    scale_pos_weight = neg / pos

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_tr, y_tr, scale_pos_weight),
        n_trials=30,
        show_progress_bar=False,
    )

    best = study.best_params
    best["scale_pos_weight"]  = scale_pos_weight
    best["eval_metric"]       = "auc"
    best["use_label_encoder"] = False
    best["random_state"]      = RANDOM_STATE
    best["verbosity"]         = 0

    model = XGBClassifier(**best)
    model.fit(X_tr, y_tr)

    cv_scores = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc")

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, y_pred, y_prob, "XGBoost")
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()),  4)
    metrics["best_params"]     = {k: v for k, v in best.items()
                                  if k not in ("eval_metric", "use_label_encoder", "verbosity")}

    ensure_project_dirs()
    joblib.dump(model, FIN_XGB_PATH)
    save_confusion_matrix(y_te, y_pred, "XGBoost", DOMAIN)
    return model, metrics, y_te, y_prob


def run() -> list[dict]:
    bundle = _load_split()
    print(f"[Finance] train={bundle['X_train_processed'].shape}  "
          f"test={bundle['X_test_processed'].shape}  "
          f"default_rate_train={bundle['y_train'].mean():.4f}")

    print("[Finance] Training Logistic Regression...")
    _, lr_metrics, y_te_lr, prob_lr = train_logistic_regression(bundle)
    print(f"  LR  acc={lr_metrics['accuracy']}  roc_auc={lr_metrics['roc_auc']}  "
          f"recall={lr_metrics['recall']}  f1={lr_metrics['f1']}")

    print("[Finance] Tuning XGBoost (30 Optuna trials)...")
    _, xgb_metrics, y_te_xgb, prob_xgb = train_xgboost(bundle)
    print(f"  XGB acc={xgb_metrics['accuracy']}  roc_auc={xgb_metrics['roc_auc']}  "
          f"recall={xgb_metrics['recall']}  f1={xgb_metrics['f1']}")

    results = [lr_metrics, xgb_metrics]
    y_tests = [y_te_lr,    y_te_xgb]
    y_probs = [prob_lr,    prob_xgb]
    save_roc_curve(results, DOMAIN, y_tests, y_probs)
    save_pr_curve(results,  DOMAIN, y_tests, y_probs)

    for m in results:
        m["domain"] = "finance"
    return results


if __name__ == "__main__":
    run()
