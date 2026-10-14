"""
Stage 3 validation script.
Run: .venv/Scripts/python.exe validate_stage3.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent
errors = []

def chk(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {label}")
    if not condition:
        errors.append(label)


# ── Load split bundles (must not be altered) ──────────────────────────────────
hc  = joblib.load(ROOT / "models/healthcare_train_test_split.joblib")
fin = joblib.load(ROOT / "models/finance_train_test_split.joblib")

# Confirm splits are unchanged from Stage 2
chk(hc["X_train"].shape  == (455, 30),  "HC train split unchanged (455,30)")
chk(hc["X_test"].shape   == (114, 30),  "HC test split unchanged (114,30)")
chk(fin["X_train"].shape[0] == 25931,   "FIN train split unchanged (25931)")
chk(fin["X_test"].shape[0]  == 6483,    "FIN test split unchanged (6483)")

# ── Model artifact existence ──────────────────────────────────────────────────
model_files = {
    "HC LR":   ROOT / "models/healthcare_logistic_regression.joblib",
    "HC XGB":  ROOT / "models/healthcare_xgboost.joblib",
    "FIN LR":  ROOT / "models/finance_logistic_regression.joblib",
    "FIN XGB": ROOT / "models/finance_xgboost.joblib",
}
for name, path in model_files.items():
    chk(path.exists(), f"{name} artifact exists")

# ── Load models and run predictions ──────────────────────────────────────────
for name, path in model_files.items():
    if not path.exists():
        print(f"  SKIP  {name} — file missing, skipping prediction checks")
        continue

    model = joblib.load(path)
    domain = "hc" if name.startswith("HC") else "fin"
    bundle = hc if domain == "hc" else fin
    X_te   = bundle["X_test_processed"]
    y_te   = bundle["y_test"]

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    chk(y_pred.shape == (len(y_te),),          f"{name} pred shape correct")
    chk(set(np.unique(y_pred)).issubset({0,1}), f"{name} pred labels are 0/1")
    chk(np.isfinite(y_prob).all(),              f"{name} probs are finite")
    chk(float(y_prob.min()) >= 0.0,             f"{name} probs >= 0")
    chk(float(y_prob.max()) <= 1.0,             f"{name} probs <= 1")

# ── Report files ──────────────────────────────────────────────────────────────
report_files = [
    ROOT / "outputs/reports/model_comparison.csv",
    ROOT / "outputs/reports/model_metrics.json",
    ROOT / "outputs/reports/training_summary.txt",
]
for p in report_files:
    chk(p.exists(), f"Report exists: {p.name}")

# ── Figure files ──────────────────────────────────────────────────────────────
figure_files = [
    "healthcare_logistic_regression_confusion_matrix.png",
    "healthcare_xgboost_confusion_matrix.png",
    "finance_logistic_regression_confusion_matrix.png",
    "finance_xgboost_confusion_matrix.png",
    "healthcare_roc_curves.png",
    "finance_roc_curves.png",
    "healthcare_pr_curves.png",
    "finance_pr_curves.png",
]
for fname in figure_files:
    p = ROOT / "outputs/figures" / fname
    chk(p.exists() and p.stat().st_size > 1000, f"Figure exists & non-empty: {fname}")

# ── Metrics JSON sanity ───────────────────────────────────────────────────────
json_path = ROOT / "outputs/reports/model_metrics.json"
if json_path.exists():
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    chk(len(metrics) == 4, f"model_metrics.json has 4 entries (got {len(metrics)})")
    for m in metrics:
        chk(0.0 <= m["accuracy"]  <= 1.0, f"{m['domain']}/{m['label']} accuracy in [0,1]")
        chk(0.0 <= m["roc_auc"]   <= 1.0, f"{m['domain']}/{m['label']} roc_auc in [0,1]")
        chk(0.0 <= m["recall"]    <= 1.0, f"{m['domain']}/{m['label']} recall in [0,1]")
        chk(0.0 <= m["precision"] <= 1.0, f"{m['domain']}/{m['label']} precision in [0,1]")
        chk(m["fn"] >= 0 and m["fp"] >= 0, f"{m['domain']}/{m['label']} CM counts non-negative")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    total = 4 + len(model_files)*5 + len(report_files) + len(figure_files) + 1 + 4*5
    print(f"ALL CHECKS PASSED ({total} assertions)")
