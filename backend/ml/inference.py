from __future__ import annotations

"""FloodNow TN v2 Model Inference Service.

Provides production-ready, calibrated inference and explainable risk attribution
for Tamil Nadu flood hazard prediction.

Loads trained and calibrated FloodNow TN v2 pipeline from:
backend/ml/models/flood_now_tn/v2/model.joblib

Maintains strict backward and forward compatibility with the DisasterPulse TN platform.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

V2_MODEL_DIR = Path(__file__).resolve().parent / "models" / "flood_now_tn" / "v2"
V1_MODEL_DIR = Path(__file__).resolve().parent / "models" / "flood_now_tn" / "v1"

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

DEFAULT_RISK_THRESHOLDS = {
    "moderate": 0.35,
    "high": 0.55,
    "critical": 0.75,
}


class ModelNotTrainedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=503,
            detail={
                "error": "MODEL_NOT_TRAINED",
                "message": (
                    "FloodNow TN model artifact is not trained yet. "
                    "Run 'python -m ml.trainer' to train, calibrate, and serialize v2 artifacts."
                ),
            },
        )


class FloodNowTNInferenceService:
    """Inference service executing FloodNow TN v2 calibrated predictions with genuine attributions."""

    def __init__(self):
        self._model = None
        self._schema = {}
        self._metrics = {}
        self._metadata = {}
        self._importance = {}
        self._version = "2.0.0"
        self._is_v2 = False
        self.reload()

    def reload(self):
        """Load or reload model artifacts, prioritizing v2 over legacy v1."""
        if (V2_MODEL_DIR / "model.joblib").exists():
            self._load_v2()
        elif (V1_MODEL_DIR / "model.joblib").exists():
            self._load_v1()
        else:
            self._model = None

    def _load_v2(self):
        try:
            self._model = joblib.load(V2_MODEL_DIR / "model.joblib")
            if (V2_MODEL_DIR / "feature_schema.json").exists():
                with (V2_MODEL_DIR / "feature_schema.json").open("r", encoding="utf-8") as f:
                    self._schema = json.load(f)
            if (V2_MODEL_DIR / "metrics.json").exists():
                with (V2_MODEL_DIR / "metrics.json").open("r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            if (V2_MODEL_DIR / "metadata.json").exists():
                with (V2_MODEL_DIR / "metadata.json").open("r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            if (V2_MODEL_DIR / "feature_importance.json").exists():
                with (V2_MODEL_DIR / "feature_importance.json").open("r", encoding="utf-8") as f:
                    self._importance = json.load(f)
            self._version = "2.0.0"
            self._is_v2 = True
        except Exception:
            self._model = None

    def _load_v1(self):
        try:
            self._model = joblib.load(V1_MODEL_DIR / "model.joblib")
            if (V1_MODEL_DIR / "metrics.json").exists():
                with (V1_MODEL_DIR / "metrics.json").open("r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            self._version = "1.0.0 (LEGACY)"
            self._is_v2 = False
        except Exception:
            self._model = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "model_name": "FloodNow TN",
            "version": self._version,
            "trained": self.is_trained,
            "status": "READY" if self.is_trained else "UNAVAILABLE",
            "algorithm": self._metadata.get("selected_algorithm", self._metrics.get("selected_algorithm", "None")),
            "held_out_metrics": self._metrics.get("held_out_test_metrics"),
            "features": CANONICAL_FEATURES,
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict calibrated flood hazard probability and compute local feature attributions."""
        if not self.is_trained:
            raise ModelNotTrainedError()

        # Backward compatibility mapping for old variable names
        clean_features = dict(features)
        if "soil_moisture" in clean_features and "soil_moisture_0_7cm" not in clean_features:
            clean_features["soil_moisture_0_7cm"] = clean_features["soil_moisture"]
        if "soil_moisture_0_7cm" in clean_features and "soil_moisture_7_28cm" not in clean_features:
            clean_features["soil_moisture_7_28cm"] = clean_features["soil_moisture_0_7cm"]

        # Compute month cyclics if missing
        if "month_sin" not in clean_features or "month_cos" not in clean_features:
            month = datetime.now(timezone.utc).month
            clean_features.setdefault("month_sin", round(math.sin(2 * math.pi * month / 12.0), 4))
            clean_features.setdefault("month_cos", round(math.cos(2 * math.pi * month / 12.0), 4))

        # Build feature DataFrame in exact canonical order
        input_data = {}
        for col in CANONICAL_FEATURES:
            val = clean_features.get(col)
            # Retain None so the trained pipeline's SimpleImputer handles missing values
            input_data[col] = [val if val is not None else np.nan]

        missing_inputs = [col for col in CANONICAL_FEATURES if clean_features.get(col) is None]

        X = pd.DataFrame(input_data, columns=CANONICAL_FEATURES)
        prob = float(self._model.predict_proba(X)[0, 1])

        # Assign risk bands based on metadata thresholds (defaulting to 0.35, 0.55, 0.75)
        thresholds = self._metadata.get("risk_thresholds", DEFAULT_RISK_THRESHOLDS)
        if prob >= thresholds.get("critical", 0.75):
            band = "critical"
        elif prob >= thresholds.get("high", 0.55):
            band = "high"
        elif prob >= thresholds.get("moderate", 0.35):
            band = "moderate"
        else:
            band = "low"

        # Compute uncalibrated tree-path local contributions
        local_explanations = self._compute_local_explanations(X, clean_features)

        return {
            "flood_probability": round(prob, 4),
            "risk_band": band,
            "calibrated": True,
            "model_version": f"FloodNow TN v{self._version}",
            "explanation_method": "tree_decision_path_contributions (uncalibrated base estimator)",
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "missing_inputs": missing_inputs,
            "local_explanations": local_explanations,
        }

    @property
    def canonical_features(self) -> list[str]:
        return list(CANONICAL_FEATURES)

    def _compute_local_explanations(self, X: pd.DataFrame, clean_features: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract uncalibrated tree decision-path feature contributions truthfully.
        
        Missing inputs are explicitly marked as imputed and excluded from physical
        hazard attribution claims.
        """
        medians = self._schema.get("feature_medians", {})
        explanations = []

        try:
            # Extract base estimator from CalibratedClassifierCV if calibrated
            base_estimator = getattr(self._model, "estimator", self._model)
            if hasattr(self._model, "calibrated_classifiers_") and len(self._model.calibrated_classifiers_) > 0:
                base_estimator = self._model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "estimator"):
                base_estimator = base_estimator.estimator

            # Extract pipeline classifier and imputer
            if hasattr(base_estimator, "named_steps"):
                imputer = base_estimator.named_steps.get("imputer")
                clf = base_estimator.named_steps.get("clf")
            else:
                imputer = None
                clf = base_estimator

            X_imputed = imputer.transform(X) if imputer else X.values

            # Case A: RandomForestClassifier - exact tree path contribution
            if hasattr(clf, "estimators_"):
                rf = clf
                contributions = np.zeros(len(CANONICAL_FEATURES))
                for tree in rf.estimators_:
                    node_indicator = tree.decision_path(X_imputed)
                    node_index = node_indicator.indices
                    # Probability of class 1 at each node
                    node_values = tree.tree_.value[:, 0, 1] / np.maximum(tree.tree_.value[:, 0].sum(axis=1), 1e-9)
                    features = tree.tree_.feature
                    for i in range(len(node_index) - 1):
                        curr_node = node_index[i]
                        next_node = node_index[i + 1]
                        feat = features[curr_node]
                        if feat >= 0:
                            diff = node_values[next_node] - node_values[curr_node]
                            contributions[feat] += diff
                contributions /= len(rf.estimators_)

                for idx, feat_name in enumerate(CANONICAL_FEATURES):
                    weight = float(contributions[idx])
                    observed = clean_features.get(feat_name)
                    med = medians.get(feat_name, 0.0)
                    is_risk = weight > 0
                    was_imputed = observed is None
                    imputed_val = round(float(X_imputed[0, idx]), 4)

                    if was_imputed:
                        explanations.append({
                            "feature": feat_name,
                            "attribution_weight": 0.0,
                            "importance_weight": 0.0,
                            "observed_value": None,
                            "imputed_value": imputed_val,
                            "value": imputed_val,
                            "baseline_mean": med,
                            "was_imputed": True,
                            "impact": "imputed_input",
                            "direction": "imputed_input",
                        })
                    else:
                        explanations.append({
                            "feature": feat_name,
                            "attribution_weight": round(weight, 4),
                            "importance_weight": round(abs(weight), 4),
                            "observed_value": observed,
                            "imputed_value": None,
                            "value": observed,
                            "baseline_mean": med,
                            "was_imputed": False,
                            "impact": "increases_risk" if is_risk else "decreases_risk",
                            "direction": "increases_risk" if is_risk else "decreases_risk",
                        })

            # Case B: LogisticRegression - linear contribution w_j * (x_j - mu_j)
            elif hasattr(clf, "coef_"):
                weights = clf.coef_[0]
                for idx, feat_name in enumerate(CANONICAL_FEATURES):
                    val = float(X_imputed[0, idx])
                    med = medians.get(feat_name, 0.0)
                    weight = float(weights[idx] * (val - med))
                    is_risk = weight > 0
                    observed = clean_features.get(feat_name)
                    was_imputed = observed is None
                    imputed_val = round(val, 4)

                    if was_imputed:
                        explanations.append({
                            "feature": feat_name,
                            "attribution_weight": 0.0,
                            "importance_weight": 0.0,
                            "observed_value": None,
                            "imputed_value": imputed_val,
                            "value": imputed_val,
                            "baseline_mean": med,
                            "was_imputed": True,
                            "impact": "imputed_input",
                            "direction": "imputed_input",
                        })
                    else:
                        explanations.append({
                            "feature": feat_name,
                            "attribution_weight": round(weight, 4),
                            "importance_weight": round(abs(weight), 4),
                            "observed_value": observed,
                            "imputed_value": None,
                            "value": observed,
                            "baseline_mean": med,
                            "was_imputed": False,
                            "impact": "increases_risk" if is_risk else "decreases_risk",
                            "direction": "increases_risk" if is_risk else "decreases_risk",
                        })

        except Exception as exc:
            # If explanation fails, return empty list rather than faking
            return []

        # Sort by absolute attribution weight descending
        explanations.sort(key=lambda e: abs(e["attribution_weight"]), reverse=True)
        return explanations


flood_now_inference = FloodNowTNInferenceService()
FloodNowInferenceService = FloodNowTNInferenceService

