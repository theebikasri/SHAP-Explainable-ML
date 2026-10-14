"""
Stage 2 validation script — verifies both saved joblib artifacts.
Run: .venv/Scripts/python.exe validate_stage2.py
"""
import sys
import joblib
import numpy as np

errors = []

# ── Healthcare ────────────────────────────────────────────────────────────────
hc   = joblib.load("models/healthcare_train_test_split.joblib")
pipe = joblib.load("models/healthcare_preprocessor.joblib")

checks = [
    (hc["domain"] == "healthcare",                          "HC domain key"),
    (hc["X_train"].shape == (455, 30),                      f"HC train shape {hc['X_train'].shape}"),
    (hc["X_test"].shape  == (114, 30),                      f"HC test shape {hc['X_test'].shape}"),
    (hc["X_train_processed"].shape == (455, 30),            "HC proc train shape"),
    (hc["X_test_processed"].shape  == (114, 30),            "HC proc test shape"),
    (set(hc["y_train"].unique()) == {0, 1},                 "HC y_train classes"),
    (set(hc["y_test"].unique())  == {0, 1},                 "HC y_test classes"),
    (abs(hc["y_train"].mean() - hc["y_test"].mean()) < 0.01, "HC stratification"),
    (abs(hc["X_train_processed"].mean()) < 0.01,            "HC scaler mean ~0"),
    (not np.isnan(hc["X_train_processed"]).any(),           "HC no NaN proc train"),
    (not np.isnan(hc["X_test_processed"]).any(),            "HC no NaN proc test"),
    (len(hc["feature_names"]) == 30,                        "HC feature_names length"),
]
for ok, label in checks:
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        errors.append(label)

print()
print(f"Healthcare — train {hc['X_train'].shape}  test {hc['X_test'].shape}")
print(f"  Malignant rate  train={hc['y_train'].mean():.4f}  test={hc['y_test'].mean():.4f}")
print(f"  Processed train mean={hc['X_train_processed'].mean():.6f}  std={hc['X_train_processed'].std():.4f}")

# ── Finance ───────────────────────────────────────────────────────────────────
fin  = joblib.load("models/finance_train_test_split.joblib")
pipe = joblib.load("models/finance_preprocessor.joblib")
cl   = fin["cleaning"]

fin_checks = [
    (fin["domain"] == "finance",                                    "FIN domain key"),
    (fin["X_train"].shape[0] == 25931,                              f"FIN train rows {fin['X_train'].shape[0]}"),
    (fin["X_test"].shape[0]  == 6483,                               f"FIN test rows {fin['X_test'].shape[0]}"),
    (fin["X_train_processed"].shape == (25931, 26),                 f"FIN proc train shape {fin['X_train_processed'].shape}"),
    (fin["X_test_processed"].shape  == (6483,  26),                 f"FIN proc test shape {fin['X_test_processed'].shape}"),
    (set(fin["y_train"].unique()) == {0, 1},                        "FIN y_train classes"),
    (set(fin["y_test"].unique())  == {0, 1},                        "FIN y_test classes"),
    (abs(fin["y_train"].mean() - fin["y_test"].mean()) < 0.01,      "FIN stratification"),
    (not np.isnan(fin["X_train_processed"]).any(),                  "FIN no NaN proc train"),
    (not np.isnan(fin["X_test_processed"]).any(),                   "FIN no NaN proc test"),
    (cl["n_initial_rows"]                        == 32581,          "FIN n_initial_rows"),
    (cl["n_age_capped_at_100"]                   == 5,              "FIN age_capped"),
    (cl["n_after_age_cap"]                       == 32581,          "FIN n_after_age_cap"),
    (cl["n_rows_dropped_emp_length_gt_age"]      == 2,              "FIN emp_dropped"),
    (cl["n_after_emp_drop"]                      == 32579,          "FIN n_after_emp_drop"),
    (cl["n_duplicates_dropped"]                  == 165,            "FIN duplicates_dropped"),
    (cl["n_after_dedup"]                         == 32414,          "FIN n_after_dedup"),
    (cl["n_rows_after_cleaning"]                 == 32414,          "FIN n_rows_after_cleaning"),
    (fin["transformed_feature_names"] is not None,                  "FIN feature names present"),
    (len(fin["transformed_feature_names"])       == 26,             f"FIN feature names count {len(fin['transformed_feature_names'])}"),
]
for ok, label in fin_checks:
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        errors.append(label)

print()
print(f"Finance — train {fin['X_train'].shape}  test {fin['X_test'].shape}")
print(f"  Default rate  train={fin['y_train'].mean():.4f}  test={fin['y_test'].mean():.4f}")
print(f"  Transformed features: {fin['X_train_processed'].shape[1]}")
print(f"  Cleaning chain: {cl['n_initial_rows']} -> cap {cl['n_age_capped_at_100']} "
      f"-> drop emp {cl['n_rows_dropped_emp_length_gt_age']} "
      f"-> dedup {cl['n_duplicates_dropped']} -> {cl['n_rows_after_cleaning']}")
print(f"  Feature names sample: {fin['transformed_feature_names'][:6]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"ALL {len(checks) + len(fin_checks)} CHECKS PASSED")
