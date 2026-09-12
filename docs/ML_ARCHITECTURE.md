# Disaster Pulse — Flood v2 ML Architecture

## What is learned

The flood hazard layer is a supervised machine-learning classifier trained from the real India Flood Inventory v3 (1967–2023), sourced from the Indian Meteorological Department and maintained by HydroSense Lab/IIT Delhi.

The primary target is the dataset's `Severity` field. If the source is ordinal/integer with a small number of distinct classes, those observed classes are learned directly. If the source is continuous, class bands are learned from the training partition only; no fixed severity cutoffs are hard-coded.

## Features

The training feature vector is built from the event inventory:

- State (categorical)
- Main Cause (categorical)
- Month represented by cyclic sine/cosine
- Flood duration
- Number of affected districts
- Event latitude/longitude when present
- Rainfall explicitly observed in the IMD event description, or an event-level external rainfall table if configured
- Missing-rainfall indicator

No synthetic rainfall is generated.

## Training

The pipeline uses chronological year-based train/validation/test partitions to reduce temporal leakage. The test partition remains untouched until final evaluation.

Models:

1. DummyClassifier baseline
2. RandomForestClassifier baseline
3. XGBoostClassifier final candidate

The XGBoost model is tuned with RandomizedSearchCV and TimeSeriesSplit using only the training partition. The best validation model is refit on train + validation, and then evaluated once on the held-out test set.

## Explainability

- permutation importance is computed on the held-out test set
- SHAP TreeExplainer is computed when SHAP is installed

The frontend reads these artifacts rather than hard-coding feature importance.

## Inference

`POST /ml/flood/predict` sends a feature vector through the saved trained model artifact. The response includes class probabilities and a `severe_probability` derived from the model's predicted probability distribution.

The flood risk score in CrisisOS is `100 * severe_probability`. This is not a hand-weighted flood formula.

## What-if

A rainfall scenario changes the rainfall feature on a cloned crisis world. The baseline and modified feature vectors are each passed through the same saved model. The frontend therefore shows a genuine model inference delta rather than multiplying an old score by a scenario factor.

## Data sources

Primary event source: IFI-Impacts/India Flood Inventory v3, DOI 10.5281/zenodo.11275211.

For richer precipitation features, an event-level precipitation table can be provided with `FLOOD_RAINFALL_SOURCE=csv` and `FLOOD_RAINFALL_PATH`. This adapter is the integration point for an IPED-derived event table. The full IPED gridded dataset is intentionally not bundled because the published archive is approximately 1.9 GB.

## Deterministic system logic

CrisisOS still contains deterministic operational simulation for roads, shelters, hospitals, resources, evacuation and routing. Those are explicitly not described as learned ML. The predictive flood hazard component and the operational response layer are kept separate.

## Re-training

```bash
cd backend
source .venv/bin/activate
python -m ml.training.train_flood_model
```

Artifacts are written to:

`backend/ml/models/flood/v1/`

- `model.joblib`
- `metrics.json`
- `metadata.json`
- `feature_importance.json`
- `shap_importance.json`


## IFI date parsing robustness

IFI v3 uses event dates such as `02-07-1967 00:00`. The ingestion layer parses mixed
date formats and falls back to the 4-digit year encoded in the stable `UEI` identifier.
This prevents valid official records from being rejected because of pandas date-format
inference differences across Python environments.


## Important target definition for IFI v3
The IFI v3 event CSV has a `Severity` column but many event rows leave it blank. The training pipeline therefore uses the published `Severity` field when populated and otherwise extracts only explicit observed severity phrases (moderate, severe, very severe) from the historical `Main Cause` text. `Main Cause` is excluded from the model features to prevent target leakage. No synthetic labels are generated.
