# Model Card: FloodNow TN v2

## 1. Model Overview

- **Model Name**: FloodNow TN v2
- **Model Version**: `2.0.0`
- **Release Date**: September 2026
- **Architecture**: `CalibratedClassifierCV(RandomForestClassifier)` with Platt Sigmoid Probability Calibration
- **Model Task**: Supervised Binary Classification estimating:
  $$\mathcal{P}(\text{recorded flood occurrence} \mid \text{antecedent environmental conditions})$$
- **Target Geography**: Tamil Nadu, India (38 Administrative Districts)
- **Primary Use Case**: Live, risk-aware flood hazard intelligence and decision support for DisasterPulse TN / CrisisOS.

---

## 2. Intended Use & Boundaries

### In-Scope Uses
- Real-time screening of district-scale flood hazard given antecedent precipitation and soil moisture.
- Supplying calibrated hazard probabilities to downstream risk-aware evacuation routing.
- Providing local, tree-path feature attributions to civil defense operators explaining why hazard is elevated.
- "What-If" scenario stress testing with counterfactual weather forcing.

### Prohibited / Out-of-Scope Claims
- **Not a 2D Hydrodynamic Simulator**: This model does NOT solve 2D Saint-Venant shallow water equations and does NOT output precise water depth in centimeters or flood boundary polylines.
- **Not Micro-Pluvial Routing**: Centroid-scale training cannot resolve street-level storm drain clogs or individual underpass waterlogging.
- **Not a Deterministic Guarantee**: Calibrated probabilities reflect historical likelihood given antecedent environmental conditions, not a deterministic certainty.

---

## 3. Training & Evaluation Data Provenance

### Positive Observations (Recorded Floods)
- **Source**: India Flood Inventory v3 (IFI-Impacts, HydroSense Lab / IIT Delhi & IMD).
- **DOI**: [10.5281/zenodo.16994648](https://doi.org/10.5281/zenodo.16994648) | **MD5**: `fea75a9ff9eba8fb328eaddfacd21d67`
- **Accepted Positives**: 717 distinct district-event occurrences across Tamil Nadu.
- **Spatial Resolution**: Canonical verified centroids for all 38 Tamil Nadu districts.
- **Data Hygiene**: 9 ambiguous or non-Tamil Nadu records (e.g. "districts of cauvery delta zone", "many parts", Karaikal/UT) were explicitly rejected with full audit logging in `training_manifest.json`.

### Matched Weak Negatives (Non-Flood Baseline)
- **Sample Count**: 717 instances (1:1 balanced design, 50.0% prevalence).
- **Sampling Methodology**: Spatially matched to positive districts with seasonal offsets between 20 and 220 days (`seed=42`).
- **Anti-Leakage Exclusion Window**: Candidates were strictly excluded if their date fell within $\pm 14$ days of ANY recorded flood event in the same district. 38 overlapping candidates were rejected during building.

### Environmental Reanalysis Data
- **Source**: Open-Meteo Historical Archive API (ERA5 / ERA5-Land hourly reanalysis).
- **Temporal Windowing**: Strict pre-event cutoff semantics. The antecedent feature window closes at 23:00 IST of the day preceding the event start date, guaranteeing zero intraday or post-event rainfall leakage.

---

## 4. Feature Schema & Canonical Inputs

| Feature Name | Type | Physical Units | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | `float` | Degrees N | WGS84 latitude coordinate |
| `longitude` | `float` | Degrees E | WGS84 longitude coordinate |
| `month_sin` | `float` | Dimensionless | $\sin(2\pi \cdot \text{month} / 12)$ cyclic month encoding |
| `month_cos` | `float` | Dimensionless | $\cos(2\pi \cdot \text{month} / 12)$ cyclic month encoding |
| `rain_1h_mm` | `float` | Millimeters | Pre-event 1-hour precipitation accumulation |
| `rain_3h_mm` | `float` | Millimeters | Pre-event 3-hour precipitation accumulation |
| `rain_6h_mm` | `float` | Millimeters | Pre-event 6-hour precipitation accumulation |
| `rain_24h_mm` | `float` | Millimeters | Pre-event 24-hour precipitation accumulation |
| `rain_72h_mm` | `float` | Millimeters | Pre-event 72-hour precipitation accumulation |
| `soil_moisture_0_7cm` | `float` | $\text{m}^3/\text{m}^3$ | Topsoil volumetric moisture ($0-7\text{ cm}$) |
| `soil_moisture_7_28cm` | `float` | $\text{m}^3/\text{m}^3$ | Root-zone volumetric moisture ($7-28\text{ cm}$) |

*Note: Legacy synthetic terrain features (`elevation_m`, `slope_deg`, `distance_to_water_km`) have been completely removed.*

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
| **Logistic Regression** | 0.8503 | 0.8603 | 0.1668 | 0.7816 |
| **RandomForestClassifier (Selected)** | **0.8805** | **0.8937** | **0.1454** | **0.8084** |

### Platt Probability Calibration
- Calibrated using `CalibratedClassifierCV` (sigmoid) fitted strictly on the validation set.
- **Validation Brier Score**: Improved from $0.1454 \rightarrow 0.1438$.

---

## 7. Held-out Test Set Performance (2019–2022)

Evaluated strictly once on the held-out test partition with 1,000-sample bootstrap 95% confidence intervals:

- **PR-AUC**: **0.9218** (95% CI: $[0.8428, 0.9712]$)
- **ROC-AUC**: **0.8896** (95% CI: $[0.7977, 0.9568]$)
- **Brier Score**: **0.1412** (95% CI: $[0.0873, 0.2057]$)
- **Accuracy**: **80.70%**
- **Precision**: **95.45%**
- **Recall**: **67.74%**
- **F1 Score**: **0.7925**

### Test Confusion Matrix
| | Predicted Negative ($P < 0.50$) | Predicted Positive ($P \ge 0.50$) | Total |
| :--- | :--- | :--- | :--- |
| **Actual Non-Flood** | 25 (True Negatives) | 1 (False Positive) | 26 |
| **Actual Flood** | 10 (False Negatives) | 21 (True Positives) | 31 |

*At high threshold, precision is 95.5%, meaning false alarms are minimized in emergency routing.*

---

## 8. Anti-Leakage & Sanity Audits

1. **Target-Shuffled Sanity Test**:
   - When training labels were randomly permuted, validation ROC-AUC collapsed from $0.8805 \rightarrow 0.5475$, confirming the model is learning real environmental patterns rather than structural artifacts.
2. **Environment-Only (Ablating Latitude/Longitude)**:
   - Without coordinates: $\text{ROC-AUC} = 0.8772$, $\text{PR-AUC} = 0.8894$.
   - Proves that physical environmental forcings (rain + soil saturation) drive predictive power, not spatial memorization.
3. **Spatial Cross-Validation Across Unseen Districts**:
   - GroupKFold by district across 4 unseen folds achieved a mean ROC-AUC of **0.9309**, demonstrating strong spatial generalization across Tamil Nadu.

---

## 9. Permutation Feature Importance

| Feature | Mean Importance | Std Importance | Physical Role |
| :--- | :--- | :--- | :--- |
| `month_cos` | 0.1101 | 0.0145 | Seasonal monsoon cycle (Northeast Monsoon peak) |
| `rain_24h_mm` | 0.0516 | 0.0059 | Short-term antecedent flash precipitation |
| `rain_72h_mm` | 0.0403 | 0.0072 | Multi-day antecedent catchment saturation |
| `soil_moisture_0_7cm` | 0.0275 | 0.0055 | Topsoil infiltration capacity limit |
| `rain_6h_mm` | 0.0229 | 0.0060 | Immediate antecedent rain burst |
| `month_sin` | 0.0222 | 0.0066 | Seasonal timing |
| `soil_moisture_7_28cm`| 0.0215 | 0.0043 | Root-zone catchment saturation |
| `rain_1h_mm` | 0.0032 | 0.0017 | Recent hour precipitation |
| `longitude` | 0.0031 | 0.0021 | Coastal vs inland gradient |
| `rain_3h_mm` | 0.0026 | 0.0016 | 3-hour storm accumulation |
| `latitude` | -0.0010 | 0.0017 | North-South spatial coordinate |

---

## 10. Operational Risk Thresholds

Calibrated probabilities map to emergency risk bands:

- **Low**: $\mathcal{P} < 0.35$ — Normal conditions.
- **Moderate**: $0.35 \le \mathcal{P} < 0.55$ — Advisory watch; alert rescue personnel.
- **High**: $0.55 \le \mathcal{P} < 0.75$ — Warning; initiate evacuation staging.
- **Critical**: $\mathcal{P} \ge 0.75$ — Imminent inundation; activate emergency response and bypass hazardous road segments.
