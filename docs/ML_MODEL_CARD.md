# Model Card: FloodNow TN v2

## 1. Model Overview

- **Model Name**: FloodNow TN v2
- **Model Version**: `2.0.0`
- **Release Date**: September 2026
- **Architecture**: `CalibratedClassifierCV(RandomForestClassifier)` with Platt Sigmoid Probability Calibration
- **Model Task**: Supervised Binary Classification estimating:
  $$\mathcal{P}(\text{recorded flood occurrence} \mid \text{antecedent physical environmental conditions})$$
- **Target Geography**: Tamil Nadu, India (38 Administrative Districts)
- **Primary Use Case**: Live, risk-aware flood hazard intelligence and decision support for DisasterPulse TN / CrisisOS.

---

## 2. Intended Use & Operational Boundaries

### In-Scope Uses
- Real-time screening of district-scale flood hazard given antecedent precipitation and soil moisture.
- Supplying calibrated hazard probabilities to downstream risk-aware evacuation routing.
- Providing local, uncalibrated tree-path feature attributions to civil defense operators explaining why hazard is elevated.
- "What-If" scenario stress testing with counterfactual weather forcing on top of valid baselines.

### Prohibited / Out-of-Scope Claims
- **Not a 2D Hydrodynamic Simulator**: This model does NOT solve 2D Saint-Venant shallow water equations and does NOT output precise water depth in centimeters or flood boundary polylines.
- **Not Micro-Pluvial Routing**: Centroid-scale training cannot resolve street-level storm drain clogs or individual underpass waterlogging.
- **Not a Deterministic Guarantee**: Calibrated probabilities reflect historical likelihood given antecedent environmental conditions, not a deterministic certainty.
- **Not Exact Calibrated SHAP**: Local feature attributions represent tree decision-path contributions from the uncalibrated base estimator, not exact Shapley values of the final calibrated probability.

---

## 3. Training & Evaluation Data Provenance

### Positive Observations (Recorded Floods)
- **Source**: India Flood Inventory v3 (IFI-Impacts, HydroSense Lab / IIT Delhi & IMD).
- **DOI**: [10.5281/zenodo.16994648](https://doi.org/10.5281/zenodo.16994648) | **MD5**: `fea75a9ff9eba8fb328eaddfacd21d67`
- **Observations**: 188 raw multi-district Tamil Nadu event records in IFI v3 expand into **717 district-event observations** across 38 districts.
- **Spatial Lookup**: 38 Tamil Nadu Revenue District reference centroids (sourced from OpenStreetMap / Survey of India boundary references). Used exclusively to query historical ERA5 reanalysis; excluded from model features to prevent coordinate memorization.
- **Data Hygiene**: 9 ambiguous or non-Tamil Nadu records (e.g. "districts of cauvery delta zone", "many parts", Karaikal/UT) were explicitly rejected with full audit logging in `training_manifest.json`.

### Matched Weak Negatives (Non-Flood Baseline)
- **Sample Count**: 717 instances (1:1 balanced design, 50.0% prevalence).
- **Sampling Methodology**: Spatially matched to positive districts with seasonal offsets between 20 and 220 days (`seed=42`).
- **Anti-Leakage Exclusion Window**: Candidates were strictly excluded if their date fell within $\pm 14$ days of ANY recorded flood event in the same district. 38 overlapping candidates were rejected during building.

### Environmental Reanalysis Data
- **Source**: Open-Meteo Historical Archive API (ERA5 / ERA5-Land hourly reanalysis).
- **Temporal Windowing**: Strict pre-event cutoff semantics. The antecedent feature window closes at 23:00 IST of the day preceding the event start date, guaranteeing zero intraday or post-event rainfall leakage.

---

## 4. Feature Schema (9 Canonical Predictors)

| Feature Name | Type | Physical Units | Description |
| :--- | :--- | :--- | :--- |
| `month_sin` | `float` | Dimensionless | $\sin(2\pi \cdot \text{month} / 12)$ cyclic month encoding |
| `month_cos` | `float` | Dimensionless | $\cos(2\pi \cdot \text{month} / 12)$ cyclic month encoding |
| `rain_1h_mm` | `float` | Millimeters | Pre-event 1-hour precipitation accumulation |
| `rain_3h_mm` | `float` | Millimeters | Pre-event 3-hour precipitation accumulation |
| `rain_6h_mm` | `float` | Millimeters | Pre-event 6-hour precipitation accumulation |
| `rain_24h_mm` | `float` | Millimeters | Pre-event 24-hour precipitation accumulation |
| `rain_72h_mm` | `float` | Millimeters | Pre-event 72-hour precipitation accumulation |
| `soil_moisture_0_7cm` | `float` | $\text{m}^3/\text{m}^3$ | Topsoil volumetric moisture ($0-7\text{ cm}$) |
| `soil_moisture_7_28cm` | `float` | $\text{m}^3/\text{m}^3$ | Root-zone volumetric moisture ($7-28\text{ cm}$) |

*Note: Coordinates (`latitude`, `longitude`) are retained only as spatial query metadata. Legacy synthetic terrain proxies (`elevation_m`, `slope_deg`, `distance_to_water_km`) have been completely removed.*

---

## 5. Temporal Partitioning & Group Integrity

Splits are strictly chronological to prevent temporal lookahead leakage:

| Partition | Time Bounds | Sample Count | Positives | Negatives | Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | $1969 - 2012$ | 1,033 | 517 | 516 | 50.0% |
| **Validation** | $2013 - 2018$ | 344 | 169 | 175 | 49.1% |
| **Test (Held-out)** | $2019 - 2022$ | 57 | 31 | 26 | 54.4% |

**Group Integrity**: Partitioning respects `event_group_id` (`{district}_{start_date}`). Zero event groups cross partition boundaries.

---

## 6. Model Selection & Calibration

### Validation Benchmark Comparison (2013–2018)

| Candidate Algorithm | ROC-AUC | PR-AUC | Brier Score | Val F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | 0.5000 | 0.4913 | 0.2500 | 0.6589 |
| **Logistic Regression** | 0.8520 | 0.8618 | 0.1655 | 0.7816 |
| **RandomForestClassifier (Selected)** | **0.8764** | **0.8880** | **0.1453** | **0.8084** |

### Platt Probability Calibration
- Calibrated using `CalibratedClassifierCV` (sigmoid) fitted strictly on the validation set.
- **Validation Brier Score**: Improved from $0.1453 \rightarrow 0.1442$.

---

## 7. Held-out Test Set Performance (2019–2022)

Evaluated strictly once on the held-out test partition with 1,000-sample bootstrap 95% confidence intervals:

- **PR-AUC**: **0.9345** (95% CI: $[0.8647, 0.9797]$)
- **ROC-AUC**: **0.9144** (95% CI: $[0.8358, 0.9702]$)
- **Brier Score**: **0.1345** (95% CI: $[0.0855, 0.1911]$)
- **Accuracy**: **80.70%**
- **Precision**: **95.45%**
- **Recall**: **67.74%**
- **F1 Score**: **0.7925**

### Test Confusion Matrix
| | Predicted Negative ($P < 0.50$) | Predicted Positive ($P \ge 0.50$) | Total |
| :--- | :--- | :--- | :--- |
| **Actual Non-Flood** | 25 (True Negatives) | 1 (False Positive) | 26 |
| **Actual Flood** | 10 (False Negatives) | 21 (True Positives) | 31 |

*At the 0.50 threshold, precision is 95.5%, meaning false alarms are minimized in emergency routing.*

---

## 8. Anti-Leakage & Sanity Audits

1. **Target-Shuffled Sanity Test**:
   - When training labels were randomly permuted, validation ROC-AUC collapsed from $0.8764 \rightarrow \mathbf{0.5204}$, confirming the model is learning real environmental relationships rather than structural noise.
2. **Environment-Only Training (Zero Coordinate Predictors)**:
   - Evaluated as the primary model: $\text{ROC-AUC} = 0.9144$, $\text{PR-AUC} = 0.9345$.
   - Confirms that physical antecedent forcings (rain + soil moisture) drive predictive power.
3. **Spatial Cross-Validation Across Unseen Districts**:
   - GroupKFold by district across 4 unseen folds achieved a mean ROC-AUC of **0.9309**, demonstrating strong spatial generalization across Tamil Nadu.

---

## 9. Permutation Feature Importance

| Feature | Mean Importance | Std Importance | Physical Role |
| :--- | :--- | :--- | :--- |
| `month_cos` | 0.1093 | 0.0145 | Seasonal monsoon cycle (Northeast Monsoon peak) |
| `rain_72h_mm` | 0.0356 | 0.0080 | Multi-day antecedent catchment saturation |
| `rain_24h_mm` | 0.0306 | 0.0063 | Short-term antecedent flash precipitation |
| `soil_moisture_0_7cm` | 0.0226 | 0.0041 | Topsoil infiltration capacity limit |
| `rain_6h_mm` | 0.0193 | 0.0049 | Immediate antecedent storm burst |
| `month_sin` | 0.0193 | 0.0061 | Seasonal timing |
| `soil_moisture_7_28cm`| 0.0153 | 0.0046 | Root-zone catchment saturation |
| `rain_3h_mm` | 0.0010 | 0.0011 | 3-hour storm accumulation |
| `rain_1h_mm` | 0.0004 | 0.0017 | Recent hour precipitation |

---

## 10. Operational Risk Thresholds

Calibrated probabilities map to emergency risk bands:

- **Low**: $\mathcal{P} < 0.35$ — Normal conditions.
- **Moderate**: $0.35 \le \mathcal{P} < 0.55$ — Advisory watch; alert rescue personnel.
- **High**: $0.55 \le \mathcal{P} < 0.75$ — Warning; initiate evacuation staging.
- **Critical**: $\mathcal{P} \ge 0.75$ — Imminent inundation; activate emergency response and bypass hazardous road segments.
