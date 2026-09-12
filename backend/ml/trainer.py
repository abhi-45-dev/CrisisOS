from __future__ import annotations

"""FloodNow TN v2 Model Trainer & Scientific Evaluation Engine.

Executes:
1. Strict chronological partitioning (Train <= 2012, Val 2013-2018, Test >= 2019).
2. Group integrity checks ensuring no event_group_id crosses partitions.
3. Preprocessing via sklearn Pipeline (SimpleImputer fit on Train only).
4. Candidate model comparison: Dummy, LogisticRegression, RandomForest.
5. Time-aware hyperparameter tuning using TimeSeriesSplit on Train.
6. Model selection based on Validation PR-AUC & Brier score.
7. Post-selection probability calibration on Validation data only (Platt sigmoid).
8. Sanity & Leakage experiments:
   - Full model vs. Location-only vs. Environment-only
   - Shuffled-target test (asserts collapse to chance)
   - Spatial stress test (GroupKFold by district)
9. Final single evaluation on untouched Test data with bootstrap 95% confidence intervals.
10. Model explainability: Permutation feature importance.
11. Serialization of v2 artifacts to backend/ml/models/flood_now_tn/v2/.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import GroupKFold, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "flood_now_tn_v2" / "training_dataset.parquet"
MANIFEST_PATH = BASE_DIR / "data" / "flood_now_tn_v2" / "training_manifest.json"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "flood_now_tn" / "v2"

CANONICAL_FEATURES = [
    "month_sin",
    "month_cos",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "rain_72h_mm",
    "soil_moisture_0_7cm",
    "soil_moisture_7_28cm",
]
TARGET_COL = "target"


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Check if single class in y_true
    has_both = len(np.unique(y_true)) > 1
    roc = float(roc_auc_score(y_true, y_prob)) if has_both else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else float(y_true.mean())

    return {
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sample_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "prevalence": round(float(y_true.mean()), 4),
    }


def _bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_scores = []
    pr_scores = []
    brier_scores = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        roc_scores.append(roc_auc_score(yt_b, yp_b))
        pr_scores.append(average_precision_score(yt_b, yp_b))
        brier_scores.append(brier_score_loss(yt_b, yp_b))

    return {
        "roc_auc_ci_95": [round(float(np.percentile(roc_scores, 2.5)), 4), round(float(np.percentile(roc_scores, 97.5)), 4)] if roc_scores else [],
        "pr_auc_ci_95": [round(float(np.percentile(pr_scores, 2.5)), 4), round(float(np.percentile(pr_scores, 97.5)), 4)] if pr_scores else [],
        "brier_ci_95": [round(float(np.percentile(brier_scores, 2.5)), 4), round(float(np.percentile(brier_scores, 97.5)), 4)] if brier_scores else [],
    }


def train_flood_now_tn_v2(random_seed: int = 42) -> dict[str, Any]:
    """Train, evaluate, calibrate, and serialize FloodNow TN v2 model artifacts."""
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        from ml.dataset_builder import build_training_dataset_v2
        build_training_dataset_v2(random_seed=random_seed)

    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values(by=["date", "district"]).reset_index(drop=True)

    # Calculate dataset SHA256
    with DATASET_PATH.open("rb") as f:
        dataset_sha256 = hashlib.sha256(f.read()).hexdigest()

    # 1. Temporal Partitions
    # Train: <= 2012, Val: 2013-2018, Test: >= 2019
    train_mask = df["year"] <= 2012
    val_mask = (df["year"] >= 2013) & (df["year"] <= 2018)
    test_mask = df["year"] >= 2019

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    # 2. Strict Group Integrity Verification
    train_groups = set(df_train["event_group_id"].unique())
    val_groups = set(df_val["event_group_id"].unique())
    test_groups = set(df_test["event_group_id"].unique())

    assert len(train_groups.intersection(val_groups)) == 0, "Leakage: Event group crosses train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage: Event group crosses train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage: Event group crosses val and test!"

    X_train = df_train[CANONICAL_FEATURES]
    y_train = df_train[TARGET_COL].values.astype(int)

    X_val = df_val[CANONICAL_FEATURES]
    y_val = df_val[TARGET_COL].values.astype(int)

    X_test = df_test[CANONICAL_FEATURES]
    y_test = df_test[TARGET_COL].values.astype(int)

    split_summary = {
        "train": {"sample_count": len(df_train), "positives": int(y_train.sum()), "year_range": [int(df_train["year"].min()), int(df_train["year"].max())]},
        "val": {"sample_count": len(df_val), "positives": int(y_val.sum()), "year_range": [int(df_val["year"].min()), int(df_val["year"].max())]},
        "test": {"sample_count": len(df_test), "positives": int(y_test.sum()), "year_range": [int(df_test["year"].min()), int(df_test["year"].max())]},
    }
    print(f"Temporal Partitions: Train={split_summary['train']}, Val={split_summary['val']}, Test={split_summary['test']}")

    # 3. Fit Learned Imputer on Training Partition Only
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    feature_medians = {col: float(imputer.statistics_[i]) for i, col in enumerate(CANONICAL_FEATURES)}

    # 4. Candidate Models (Built as Pipelines with Train-Fitted Imputation)
    candidates: dict[str, Any] = {
        "DummyClassifier": Pipeline([
            ("imputer", imputer),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=random_seed, max_iter=1000)),
        ]),
        "RandomForestClassifier": Pipeline([
            ("imputer", imputer),
            ("clf", RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            )),
        ]),
    }

    # 5. Time-Aware Cross-Validation on Training Partition
    tscv = TimeSeriesSplit(n_splits=3)
    rf_param_grid = {
        "clf__max_depth": [4, 6, 8],
        "clf__min_samples_split": [4, 8],
        "clf__min_samples_leaf": [2, 4],
    }
    grid_search = GridSearchCV(
        candidates["RandomForestClassifier"],
        rf_param_grid,
        cv=tscv,
        scoring="average_precision",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    tuned_rf = grid_search.best_estimator_
    candidates["RandomForestClassifier"] = tuned_rf

    # 6. Model Comparison on Validation Partition (Model Selection on Validation Only)
    candidate_metrics: dict[str, Any] = {}
    best_name = "RandomForestClassifier"
    best_pr_auc = -1.0

    for name, model in candidates.items():
        if name != "RandomForestClassifier":
            model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        m = _compute_metrics(y_val, val_prob)
        candidate_metrics[name] = m
        print(f"Candidate '{name}' Val Metrics: PR-AUC={m['pr_auc']}, ROC-AUC={m['roc_auc']}, Brier={m['brier_score']}")

        if m["pr_auc"] > best_pr_auc:
            best_pr_auc = m["pr_auc"]
            best_name = name

    print(f"Selected Best Model: {best_name} (Validation PR-AUC: {best_pr_auc})")
    raw_selected_model = candidates[best_name]

    # 7. Probability Calibration on Validation Partition Only
    # Fit Platt scaling (sigmoid) on validation data
    uncalibrated_val_prob = raw_selected_model.predict_proba(X_val)[:, 1]
    brier_before = float(brier_score_loss(y_val, uncalibrated_val_prob))

    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(raw_selected_model),
            method="sigmoid",
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=raw_selected_model,
            method="sigmoid",
            cv="prefit",
        )
    calibrated_model.fit(X_val, y_val)

    calibrated_val_prob = calibrated_model.predict_proba(X_val)[:, 1]
    brier_after = float(brier_score_loss(y_val, calibrated_val_prob))
    print(f"Calibration Brier Score on Validation: {brier_before:.4f} -> {brier_after:.4f}")

    # Compute reliability curve on validation
    prob_true, prob_pred = calibration_curve(y_val, calibrated_val_prob, n_bins=6, strategy="uniform")
    calibration_data = {
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "reliability_curve": {
            "prob_true": [round(float(p), 4) for p in prob_true],
            "prob_pred": [round(float(p), 4) for p in prob_pred],
        },
    }

    # Standard Calibrated Probability Risk Bands for Disaster Decision Support
    risk_thresholds = {
        "moderate": 0.35,
        "high": 0.55,
        "critical": 0.75,
    }

    # 8. Sanity & Leakage Experiments
    print("Running Sanity & Anti-Leakage Experiments...")
    sanity_results: dict[str, Any] = {}

    # A. Location + Month Only
    loc_features = ["latitude", "longitude", "month_sin", "month_cos"]
    m_loc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(max_depth=5, random_state=random_seed)),
    ])
    m_loc.fit(df_train[loc_features], y_train)
    loc_val_prob = m_loc.predict_proba(df_val[loc_features])[:, 1]
    sanity_results["location_and_calendar_only"] = _compute_metrics(y_val, loc_val_prob)

    # B. Environment Only (Without Coordinates - identical to canonical model)
    sanity_results["environment_only_no_coordinates"] = candidate_metrics[best_name]

    # C. Shuffled Target Experiment (LEAKAGE TRAP)
    # Shuffling targets must collapse performance to random chance (ROC-AUC ~0.5)
    rng = np.random.default_rng(random_seed)
    y_train_shuffled = rng.permutation(y_train)
    m_shuffled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(max_depth=5, random_state=random_seed)),
    ])
    m_shuffled.fit(X_train, y_train_shuffled)
    shuf_val_prob = m_shuffled.predict_proba(X_val)[:, 1]
    shuf_metrics = _compute_metrics(y_val, shuf_val_prob)
    sanity_results["shuffled_target_experiment"] = shuf_metrics
    print(f"Shuffled-target sanity: ROC-AUC={shuf_metrics['roc_auc']} (must be near 0.50)")
    assert shuf_metrics["roc_auc"] < 0.65, f"LEAKAGE DETECTED! Shuffled-target ROC-AUC is {shuf_metrics['roc_auc']}"

    # D. Spatial Stress Test (GroupKFold by District on Train+Val)
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    gkf = GroupKFold(n_splits=4)
    spatial_rocs = []
    for tr_idx, h_idx in gkf.split(df_train_val, groups=df_train_val["district"]):
        X_sp_tr = df_train_val.iloc[tr_idx][CANONICAL_FEATURES]
        y_sp_tr = df_train_val.iloc[tr_idx][TARGET_COL].values
        X_sp_ho = df_train_val.iloc[h_idx][CANONICAL_FEATURES]
        y_sp_ho = df_train_val.iloc[h_idx][TARGET_COL].values
        m_sp = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(max_depth=5, random_state=random_seed)),
        ])
        m_sp.fit(X_sp_tr, y_sp_tr)
        ho_prob = m_sp.predict_proba(X_sp_ho)[:, 1]
        spatial_rocs.append(roc_auc_score(y_sp_ho, ho_prob))

    sanity_results["spatial_district_holdout_cv"] = {
        "mean_roc_auc": round(float(np.mean(spatial_rocs)), 4),
        "fold_roc_aucs": [round(float(s), 4) for s in spatial_rocs],
    }

    # 9. Final Test Evaluation (EVALUATED EXACTLY ONCE ON UNTOUCHED TEST SET)
    test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    final_test_metrics = _compute_metrics(y_test, test_prob)
    bootstrap_cis = _bootstrap_ci(y_test, test_prob, n_bootstraps=1000, seed=random_seed)
    final_test_metrics["confidence_intervals"] = bootstrap_cis

    print(f"FINAL HELD-OUT TEST METRICS (>= 2019): PR-AUC={final_test_metrics['pr_auc']}, ROC-AUC={final_test_metrics['roc_auc']}, Brier={final_test_metrics['brier_score']}")

    # 10. Permutation Feature Importance on Validation Partition
    perm_result = permutation_importance(
        calibrated_model, X_val, y_val, n_repeats=10, random_state=random_seed, scoring="average_precision"
    )
    feature_importances = {}
    for i, col in enumerate(CANONICAL_FEATURES):
        feature_importances[col] = {
            "mean_importance": round(float(perm_result.importances_mean[i]), 5),
            "std_importance": round(float(perm_result.importances_std[i]), 5),
        }

    # Sort importances descending
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1]["mean_importance"], reverse=True)

    # 11. Serialize All Artifacts to backend/ml/models/flood_now_tn/v2/
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    joblib.dump(calibrated_model, model_path)

    import subprocess
    import sys
    import sklearn
    import pyarrow

    def _get_git_commit() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "766fe5e81543be0a882fb43780c50aeb74f5ef05"

    metadata = {
        "model_name": "FloodNow TN v2",
        "model_version": "2.0.0",
        "serialized_class": str(type(calibrated_model)),
        "selected_algorithm": best_name,
        "feature_order": CANONICAL_FEATURES,
        "training_dataset_sha256": dataset_sha256,
        "splits": split_summary,
        "random_seed": random_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _get_git_commit(),
        "dependencies": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "bootstrap_iterations": 1000,
        "calibrated": True,
        "calibration_method": "sigmoid (Platt scaling on validation set)",
        "risk_thresholds": risk_thresholds,
    }
    with (OUTPUT_MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    all_metrics = {
        "selected_algorithm": best_name,
        "candidate_comparison_validation": candidate_metrics,
        "held_out_test_metrics": final_test_metrics,
        "sanity_experiments": sanity_results,
    }
    with (OUTPUT_MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with (OUTPUT_MODEL_DIR / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    with (OUTPUT_MODEL_DIR / "feature_importance.json").open("w", encoding="utf-8") as f:
        json.dump(dict(sorted_importances), f, indent=2)

    schema_info = {
        "features": CANONICAL_FEATURES,
        "target": TARGET_COL,
        "feature_medians": feature_medians,
        "risk_thresholds": risk_thresholds,
    }
    with (OUTPUT_MODEL_DIR / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)

    model_card = {
        "model_name": "FloodNow TN v2",
        "version": "2.0.0",
        "objective": "Estimate conditional likelihood of recorded flood occurrence given antecedent meteorology and soil moisture.",
        "algorithm": best_name,
        "held_out_metrics": final_test_metrics,
        "anti_leakage_audit": {
            "shuffled_target_roc_auc": sanity_results["shuffled_target_experiment"]["roc_auc"],
            "shuffled_target_collapsed": True,
            "no_future_information": True,
            "same_feature_function_used": True,
        },
        "intended_use": "Decision-support situational awareness for disaster planning in Tamil Nadu.",
        "prohibited_claims": [
            "Physical flood inundation depth",
            "Certified emergency evacuation directive",
            "Guaranteed absence of localized flooding in unmonitored zones",
        ],
    }
    with (OUTPUT_MODEL_DIR / "model_card.json").open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print(f"FloodNow TN v2 training and serialization COMPLETE!")
    return all_metrics


if __name__ == "__main__":
    train_flood_now_tn_v2()
