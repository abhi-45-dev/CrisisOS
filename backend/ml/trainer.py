from __future__ import annotations

"""FloodNow TN Model Trainer.

Trains, tunes with TimeSeriesSplit, calibrates probabilities, evaluates on held-out test data,
and serializes artifacts with comprehensive provenance and model card.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "flood_now_tn" / "training_dataset.parquet"
MANIFEST_PATH = BASE_DIR / "data" / "flood_now_tn" / "training_manifest.json"
MODEL_DIR = BASE_DIR / "models" / "flood_now_tn" / "v1"

FEATURE_NAMES = [
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "rain_24h_mm",
    "rain_72h_mm",
    "soil_moisture",
    "elevation_m",
    "slope_deg",
    "distance_to_water_km",
]
TARGET_COL = "flood_occurrence"


def train_flood_now_tn(random_seed: int = 42) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        from ml.dataset_builder import build_training_dataset
        build_training_dataset()

    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values(by=["year", "month"]).reset_index(drop=True)

    # 1. Temporal Partitions
    # Train: <= 2012, Val: 2013-2018, Test: >= 2019 (Strictly held out)
    train_mask = df["year"] <= 2012
    val_mask = (df["year"] >= 2013) & (df["year"] <= 2018)
    test_mask = df["year"] >= 2019

    X_train = df.loc[train_mask, FEATURE_NAMES]
    y_train = df.loc[train_mask, TARGET_COL].values

    X_val = df.loc[val_mask, FEATURE_NAMES]
    y_val = df.loc[val_mask, TARGET_COL].values

    X_test = df.loc[test_mask, FEATURE_NAMES]
    y_test = df.loc[test_mask, TARGET_COL].values

    # 2. Candidate Models
    candidates: dict[str, Any] = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=random_seed, max_iter=1000)),
        ]),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    # 3. Time-Aware Tuning on Train Partition using TimeSeriesSplit
    param_dist = {
        "n_estimators": [80, 120, 160],
        "max_depth": [5, 7, 10, None],
        "min_samples_split": [2, 4, 8],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=random_seed, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=6,
        cv=tscv,
        scoring="average_precision",
        random_state=random_seed,
    )
    rf_search.fit(X_train, y_train)
    candidates["TunedRandomForest"] = rf_search.best_estimator_

    # 4. Evaluate candidates on Validation partition
    val_scores = {}
    best_candidate_name = None
    best_pr_auc = -1.0
    best_estimator = None

    for name, model in candidates.items():
        if name != "TunedRandomForest":
            model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
        pr_auc = average_precision_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        val_scores[name] = {"pr_auc": round(pr_auc, 4), "brier_score": round(brier, 4)}

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_candidate_name = name
            best_estimator = model

    # 5. Calibrate probabilities using TimeSeriesSplit on Train + Validation
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    calibrated_model = CalibratedClassifierCV(
        estimator=best_estimator,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    calibrated_model.fit(X_train_val, y_train_val)

    # 6. Final Evaluation on Held-Out Test Partition (2019-2023)
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)

    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds).tolist()

    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=5)
    calib_curve_data = [
        {"predicted_probability": round(float(p), 4), "true_frequency": round(float(t), 4)}
        for p, t in zip(prob_pred, prob_true)
    ]

    # 7. Global Feature Importances
    base_rf = best_estimator if isinstance(best_estimator, RandomForestClassifier) else candidates["RandomForestClassifier"]
    importances = base_rf.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda p: p[1], reverse=True)
    ]

    # Save feature baseline means for local perturbation explainability
    feature_means = {col: round(float(df[col].mean()), 4) for col in FEATURE_NAMES}

    # 8. Persist Artifacts
    # Save Model Pipeline
    joblib.dump(calibrated_model, MODEL_DIR / "model.joblib")

    # Metrics
    metrics_payload = {
        "model_name": "FloodNow TN",
        "version": "1.0.0",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_comparison": val_scores,
        "selected_algorithm": best_candidate_name,
        "held_out_test_metrics": {
            "pr_auc": round(float(test_pr_auc), 4),
            "roc_auc": round(float(test_roc_auc), 4),
            "brier_score": round(float(test_brier), 4),
            "precision": round(float(test_precision), 4),
            "recall": round(float(test_recall), 4),
            "f1": round(float(test_f1), 4),
            "accuracy": round(float(test_acc), 4),
            "confusion_matrix": cm,
            "test_sample_count": len(y_test),
            "test_positive_count": int(np.sum(y_test)),
        },
        "calibration": {
            "method": "sigmoid",
            "evaluated_brier_score": round(float(test_brier), 4),
            "calibration_curve": calib_curve_data,
        },
    }
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Feature Schema & Baseline
    schema_payload = {
        "features": FEATURE_NAMES,
        "feature_means": feature_means,
        "risk_bands": {
            "low": {"min": 0.0, "max": 0.35, "description": "Minimal flood likelihood"},
            "moderate": {"min": 0.35, "max": 0.60, "description": "Elevated soil moisture or moderate runoff"},
            "high": {"min": 0.60, "max": 0.80, "description": "High likelihood of surface inundation"},
            "critical": {"min": 0.80, "max": 1.0, "description": "Severe recorded flood hazard signature"},
        },
    }
    with (MODEL_DIR / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)

    # Feature Importance
    with (MODEL_DIR / "feature_importance.json").open("w", encoding="utf-8") as f:
        json.dump({"global_mdi_importance": feat_imp}, f, indent=2)

    # Model Card
    model_card = {
        "model_name": "FloodNow TN",
        "version": "1.0.0",
        "purpose": "Supervised binary flood occurrence probability prediction across Tamil Nadu space-time units.",
        "algorithm": best_candidate_name,
        "training_data_source": "India Flood Inventory v3 (Zenodo record 16994648) with weak-negative non-event sampling.",
        "prospective_guarantee": "Only prospective features constructible at inference time are used. Post-event impact features are strictly excluded.",
        "temporal_validation": "Train partition: <=2012, Val partition: 2013-2018, Test partition: 2019-2023 (unseen). Hyperparameter search with TimeSeriesSplit.",
        "calibration": "Calibrated with Sigmoid scaling on validation partition.",
        "limitations": "Absence of an event in IFI does not guarantee zero local hyper-local pooling. Predictions serve decision support, not certified emergency dispatch.",
    }
    with (MODEL_DIR / "model_card.json").open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    # Metadata
    metadata_payload = {
        "model_id": "flood_now_tn_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": best_candidate_name,
        "feature_count": len(FEATURE_NAMES),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "calibrated": True,
        "status": "READY",
    }
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return metrics_payload


if __name__ == "__main__":
    metrics = train_flood_now_tn()
    print("Training complete! Held-out test PR-AUC:", metrics["held_out_test_metrics"]["pr_auc"])
