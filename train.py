"""
train.py  -  HeartVigil AI  |  Model Training Pipeline v3
==========================================================
CLEAN APPROACH: 13 UCI features only.
- XGBoost + RandomForest stacking (no feature engineering overhead)
- Three-way split: train(60%) / calibration(20%) / test(20%)
- SMOTE on training only, sigmoid calibration on held-out cal set
- Cross-validates on full dataset for unbiased estimate
- Clinical sanity checks on AHA-literature cases
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    brier_score_loss
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── Clinical sanity cases ─────────────────────────────────────────────────────
# NOTE: In the UCI Cleveland dataset encoding:
#   cp=3 (asymptomatic) is PARADOXICALLY associated with disease presence
#   (silent ischaemia — patients found incidentally)
#   cp=0 (typical angina) — mixed, many turn out non-cardiac
# Sanity is against the MODEL's learned distribution, not intuition.
SANITY_CASES = [
    # These use UCI semantics: ca, exang, oldpeak are the strongest predictors
    ("HIGH: 64yo male, exang, 3vessels, oldpeak=4.0, reversed-thal",
     {"age":64,"sex":1,"cp":3,"trestbps":160,"chol":254,"fbs":0,
      "restecg":0,"thalach":112,"exang":1,"oldpeak":4.0,"slope":1,"ca":3,"thal":3},
     0.60, 1.00),
    ("HIGH: 62yo male, exang, 2vessels, oldpeak=3.5",
     {"age":62,"sex":1,"cp":3,"trestbps":150,"chol":268,"fbs":0,
      "restecg":0,"thalach":134,"exang":1,"oldpeak":3.5,"slope":1,"ca":2,"thal":3},
     0.60, 1.00),
    ("LOW: 29yo female, no vessels, no exang, oldpeak=0",
     {"age":29,"sex":0,"cp":1,"trestbps":112,"chol":165,"fbs":0,
      "restecg":0,"thalach":188,"exang":0,"oldpeak":0.0,"slope":2,"ca":0,"thal":2},
     0.00, 0.35),
    ("LOW: 35yo female, no risk factors",
     {"age":35,"sex":0,"cp":2,"trestbps":108,"chol":172,"fbs":0,
      "restecg":0,"thalach":180,"exang":0,"oldpeak":0.0,"slope":2,"ca":0,"thal":2},
     0.00, 0.40),
]


def load_data():
    csv = BASE_DIR / "heart.csv"
    if not csv.exists():
        print("ERROR: heart.csv not found.")
        sys.exit(1)
    df = pd.read_csv(csv)
    target_col = "target" if "target" in df.columns else "num"
    # Binarise (UCI raw has 0=None, 1-4=Disease)
    # BUT this heart.csv has 0=Disease, 1=No Disease (Ronit/Kaggle inversion)
    # We want Class 1 to be Disease Presence.
    df[target_col] = (df[target_col] == 0).astype(int)
    # Remove rows with physiologically impossible values
    df = df[(df["trestbps"] > 0) & (df["chol"] > 0) & (df["thalach"] > 0)]
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    print("Label distribution: %s" % dict(y.value_counts()))
    return X, y


def main():
    print("=" * 56)
    print("HeartVigil AI  --  Model Training Pipeline v3")
    print("=" * 56)

    X, y = load_data()
    n = len(X)
    print("Dataset: %d patients | Positive rate: %.1f%%" % (n, y.mean()*100))
    print("Features: %d (13 UCI, no engineering)" % len(FEATURE_COLS))

    # ── Three-way split ────────────────────────────────────────────────────────
    # 60% train, 20% calibration (held-out), 20% test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print("Split: train=%d  cal=%d  test=%d" % (len(X_tr), len(X_cal), len(X_te)))

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc  = pd.DataFrame(scaler.fit_transform(X_tr),  columns=FEATURE_COLS)
    X_cal_sc = pd.DataFrame(scaler.transform(X_cal),     columns=FEATURE_COLS)
    X_te_sc  = pd.DataFrame(scaler.transform(X_te),      columns=FEATURE_COLS)

    # ── SMOTE on training folds only ───────────────────────────────────────────
    # Ensure k_neighbors doesn't exceed class size
    min_class_samples = y_tr.value_counts().min()
    k_neighbors = min(5, min_class_samples - 1) if min_class_samples > 1 else 1
    if k_neighbors >= 1:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X_tr_sc, y_tr)
        print("After SMOTE: %d training samples (%.1f%% positive)" % (
            len(X_res), y_res.mean()*100))
    else:
        X_res, y_res = X_tr_sc, y_tr
        print("SMOTE skipped (class too small)")

    # ── Build estimators ────────────────────────────────────────────────────────
    estimators = []
    
    # XGBoost
    try:
        import xgboost as xgb
        estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.15, reg_alpha=0.1, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, use_label_encoder=False,
        )))
        print("  XGBoost: OK")
    except ImportError:
        print("  XGBoost not installed, using GradientBoosting instead")
        estimators.append(("gb", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )))
    
    # RandomForest
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=500, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )))
    print("  RandomForest: OK")

    # LightGBM (optional)
    try:
        import lightgbm as lgb
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=20, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5,
            random_state=42, verbose=-1,
        )))
        print("  LightGBM: OK")
    except ImportError:
        print("  LightGBM not installed, skipping")

    # ── Stacking (no passthrough, LR meta) ─────────────────────────────────────
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    stacked = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_lr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    print("\nFitting stacking ensemble...")
    stacked.fit(X_res, y_res)

    # ── Calibrate on held-out cal set ──────────────────────────────────────────
    # Note: CalibratedClassifierCV with cv='prefit' expects the base model
    # to be already fitted. We'll fit a fresh calibrator on the calibration set.
    # Alternative: use CalibratedClassifierCV with cv=5 on the calibration set
    calibrated = CalibratedClassifierCV(stacked, method="sigmoid", cv=5)
    calibrated.fit(X_cal_sc, y_cal)
    print("Calibration done (sigmoid, cv=5, n_cal=%d)" % len(X_cal))

    # ── Evaluate ────────────────────────────────────────────────────────────────
    y_pred = calibrated.predict(X_te_sc)
    y_prob = calibrated.predict_proba(X_te_sc)[:, 1]
    acc   = accuracy_score(y_te, y_pred)
    auc   = roc_auc_score(y_te, y_prob)
    brier = brier_score_loss(y_te, y_prob)

    print("\n" + "-" * 48)
    print("  Test  n=%d  Accuracy=%.2f%%  AUC=%.4f  Brier=%.4f" % (
        len(y_te), acc*100, auc, brier))
    print("-" * 48)
    print(classification_report(y_te, y_pred, target_names=["No Disease","Disease"]))

    # ── 5-fold CV on full data (without calibration for honest estimate) ────────
    print("\n5-fold CV on full dataset (uncalibrated stacking)...")
    X_full_sc = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=FEATURE_COLS
    )
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(stacked, X_full_sc, y, cv=cv5,
                             scoring="accuracy", n_jobs=-1)
    cv_auc = cross_val_score(stacked, X_full_sc, y, cv=cv5,
                             scoring="roc_auc", n_jobs=-1)
    print("5-fold CV Accuracy: %.2f%% +/- %.2f%%" % (cv_acc.mean()*100, cv_acc.std()*100))
    print("5-fold CV ROC-AUC : %.4f +/- %.4f" % (cv_auc.mean(), cv_auc.std()))

    # ── Sample probability distribution check ─────────────────────────────────
    print("\nProbability distribution on test set:")
    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for i in range(len(buckets)-1):
        mask = (y_prob >= buckets[i]) & (y_prob < buckets[i+1])
        n_in = mask.sum()
        pos  = y_te[mask].sum() if n_in else 0
        print("  [%.1f-%.1f]: %d samples, %d positive" % (
            buckets[i], buckets[i+1], n_in, pos))

    # ── Clinical sanity checks ─────────────────────────────────────────────────
    print("\n" + "-" * 48)
    print("Clinical Sanity Checks:")
    passed = 0
    for desc, vals, lo, hi in SANITY_CASES:
        row_df = pd.DataFrame([vals], columns=FEATURE_COLS)
        row_sc = pd.DataFrame(scaler.transform(row_df), columns=FEATURE_COLS)
        prob   = calibrated.predict_proba(row_sc)[0][1]
        ok     = lo <= prob <= hi
        passed += ok
        tag = "PASS" if ok else "FAIL"
        print("  [%s] %s" % (tag, desc[:58]))
        print("       P(disease)=%.3f  expected=[%.2f,%.2f]" % (prob, lo, hi))

    print("\nSanity: %d/%d passed" % (passed, len(SANITY_CASES)))

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(FEATURE_COLS, BASE_DIR / "feature_names.joblib")
    joblib.dump(FEATURE_COLS, BASE_DIR / "feature_names_full.joblib")
    joblib.dump(scaler,       BASE_DIR / "scaler.joblib")
    joblib.dump(calibrated,   BASE_DIR / "model.joblib")

    print("\nSaved: model.joblib, scaler.joblib, feature_names.joblib, feature_names_full.joblib")
    print("Final: Accuracy=%.2f%%  AUC=%.4f" % (acc*100, auc))
    print("Done.")
    return acc, auc


if __name__ == "__main__":
    main()