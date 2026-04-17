"""
train.py  -  HeartVigil AI  |  Model Training Pipeline v4
==========================================================
CLEAN APPROACH: 13 UCI features only.
- One-Hot Encoder for categorical features.
- Soft Voting Classifier (XGBoost + RandomForest) (no stacking meta-overfit).
- 80% Train / 20% Test split.
- SMOTE removed (dataset is sufficiently balanced).
- CalibratedClassifierCV fitted with cv=5 on training dataset.
- Clinical sanity checks on AHA-literature cases.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    brier_score_loss
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# Define which features are categorical vs continuous
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

SANITY_CASES = [
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
    
    # Correct label mapping:
    # Kaggle dataset: natively 1=healthy, 0=disease. We invert it so 1=disease, 0=healthy.
    # Original UCI dataset: 0=healthy, 1-4=disease presence.
    if target_col == "target":
        df[target_col] = 1 - df[target_col]
    else:
        df[target_col] = (df[target_col] > 0).astype(int)
        
    df = df[(df["trestbps"] > 0) & (df["chol"] > 0) & (df["thalach"] > 0)]
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    print("Label distribution: %s" % dict(y.value_counts()))
    return X, y

def main():
    print("=" * 56)
    print("HeartVigil AI  --  Model Training Pipeline v4")
    print("=" * 56)

    X, y = load_data()
    n = len(X)
    print("Dataset: %d patients | Positive rate: %.1f%%" % (n, y.mean()*100))

    # ── Split ──────────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Split: train=%d  test=%d" % (len(X_tr), len(X_te)))

    # ── Preprocessing Pipeline ─────────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ]
    )

    # ── Build Estimators ───────────────────────────────────────────────────────
    estimators = []
    try:
        from sklearn.linear_model import LogisticRegression
        estimators.append(("lr", LogisticRegression(
            C=1.0, max_iter=2000, random_state=42, class_weight="balanced"
        )))
        print("  LogisticRegression: OK")
    except ImportError:
        estimators.append(("gb", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )))

    # ── Voting Classifier (Soft Voting) ────────────────────────────────────────
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )

    # Compile into full pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", voting_clf)
    ])

    print("\nFitting integrated pipeline (preprocessor + ensemble)...")
    
    # ── Calibrate (with cross validation) ──────────────────────────────────────
    # Wrap the full pipeline in a CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(full_pipeline, method="sigmoid", cv=5)
    calibrated.fit(X_tr, y_tr)
    
    print("Calibration done (sigmoid, cv=5)")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = calibrated.predict(X_te)
    y_prob = calibrated.predict_proba(X_te)[:, 1]
    acc   = accuracy_score(y_te, y_pred)
    auc   = roc_auc_score(y_te, y_prob)
    brier = brier_score_loss(y_te, y_prob)

    print("\n" + "-" * 48)
    print("  Test  n=%d  Accuracy=%.2f%%  AUC=%.4f  Brier=%.4f" % (
        len(y_te), acc*100, auc, brier))
    print("-" * 48)
    print(classification_report(y_te, y_pred, target_names=["No Disease","Disease"]))

    # ── Clinical sanity checks ─────────────────────────────────────────────────
    print("\n" + "-" * 48)
    print("Clinical Sanity Checks:")
    passed = 0
    for desc, vals, lo, hi in SANITY_CASES:
        row_df = pd.DataFrame([vals], columns=FEATURE_COLS)
        prob   = calibrated.predict_proba(row_df)[0][1]
        ok     = lo <= prob <= hi
        passed += ok
        tag = "PASS" if ok else "FAIL"
        print("  [%s] %s" % (tag, desc[:58]))
        print("       P(disease)=%.3f  expected=[%.2f,%.2f]" % (prob, lo, hi))

    print("\nSanity: %d/%d passed" % (passed, len(SANITY_CASES)))

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(FEATURE_COLS, BASE_DIR / "feature_names.joblib")
    joblib.dump(FEATURE_COLS, BASE_DIR / "feature_names_full.joblib")
    
    # We dump dummy/legacy scaler so older frontend processes don't crash before hot-reload
    joblib.dump("dummy", BASE_DIR / "scaler.joblib")
    
    joblib.dump(calibrated, BASE_DIR / "model.joblib")

    print("\nSaved: model.joblib, scaler.joblib, feature_names.joblib, feature_names_full.joblib")
    print("Final: Accuracy=%.2f%%  AUC=%.4f" % (acc*100, auc))
    print("Done.")
    return acc, auc


if __name__ == "__main__":
    main()