# FloodNow TN v2 — Feature Provenance & Production Alignment

This document details the exact scientific lineage, historical training definitions, live operational providers, and missing-data policies for every feature in **FloodNow TN v2**.

---

## 🔬 Model Predictor Schema (9 Physical Features)

To eliminate coordinate memorization and ensure that FloodNow TN v2 learns purely transferable physical and meteorological relationships, `latitude` and `longitude` are **excluded from direct model predictors**. They are retained strictly as transparent spatial metadata used to query localized meteorological reanalysis and live forecasts.

| Feature | Historical Training Source | Training Definition | Live / Production Source | Live Definition | Same Semantics? | Missing-Data Handling |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `month_sin` | Event antecedent window calendar timestamp | $\sin(2\pi \cdot \text{month} / 12)$ cyclic encoding | Current inference timestamp | $\sin(2\pi \cdot \text{month} / 12)$ cyclic encoding | **YES** | Imputed from current UTC month |
| `month_cos` | Event antecedent window calendar timestamp | $\cos(2\pi \cdot \text{month} / 12)$ cyclic encoding | Current inference timestamp | $\cos(2\pi \cdot \text{month} / 12)$ cyclic encoding | **YES** | Imputed from current UTC month |
| `rain_1h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Hourly precipitation accumulation in the 1 hour prior to prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Hourly precipitation in the most recent completed past hour (mm) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `rain_3h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Cumulative precipitation in the 3 hours prior to prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Cumulative precipitation across the 3 trailing hourly slots (mm) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `rain_6h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Cumulative precipitation in the 6 hours prior to prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Cumulative precipitation across the 6 trailing hourly slots (mm) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `rain_24h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Antecedent 24-hour precipitation accumulation ending before prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Trailing 24-hour observed precipitation accumulation (mm) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `rain_72h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Antecedent 72-hour precipitation accumulation ending before prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Trailing 72-hour observed precipitation accumulation (mm) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `soil_moisture_0_7cm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Volumetric soil water content in topsoil layer (0–7 cm, $\text{m}^3/\text{m}^3$) | Open-Meteo Forecast API (soil_moisture_0_to_7cm) | Latest topsoil volumetric soil moisture nowcast ($\text{m}^3/\text{m}^3$) | **YES** | Median-imputed via train-fitted SimpleImputer |
| `soil_moisture_7_28cm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Volumetric soil water content in root-zone layer (7–28 cm, $\text{m}^3/\text{m}^3$) | Open-Meteo Forecast API (soil_moisture_7_to_28cm) | Latest root-zone volumetric soil moisture nowcast ($\text{m}^3/\text{m}^3$) | **YES** | Median-imputed via train-fitted SimpleImputer |

---

## 📍 Spatial Metadata & Centroid Lineage

- **Coordinates (`latitude`, `longitude`)**:
  - Sourced from OpenStreetMap and Survey of India administrative district boundary references for the 38 Tamil Nadu Revenue Districts.
  - Role: Used exclusively as coordinate query arguments to fetch reanalysis data from Open-Meteo ERA5 / ERA5-Land and to clip predictions to Tamil Nadu geometry.
  - Dropped from model feature matrix $X$ to prevent geographic overfitting.

---

## 🚫 Explicitly Rejected / Dropped Features

The following features present in legacy FloodNow TN v1 were **permanently removed**:

1. `elevation_m` (Synthetic): Fabricated based on crude longitude thresholds (`25.0 if lon > 79.5 else 120.0`). Dropped globally.
2. `slope_deg` (Synthetic): Fabricated based on longitude thresholds (`1.2 if lon > 79.5 else 2.5`). Dropped globally.
3. `distance_to_water_km` (Synthetic): Fabricated based on longitude thresholds (`2.5 if lon > 79.8 else 8.0`). Dropped globally.

---

## 🛡️ Anti-Leakage & Missing-Data Invariants

1. **Single Feature Function**:
   - `get_environmental_features(latitude, longitude, prediction_timestamp)` accepts zero target labels or positive/negative flags.
   - Positives and weak negatives query identical variables through identical code paths.

2. **Pre-Event Time Cutoff**:
   - Antecedent evaluation windows end strictly at 23:00 IST on the day preceding the event date.
   - Zero precipitation occurring on or after the event date is ever included in predictors.

3. **Honest Missingness & Imputation Reporting**:
   - Missing meteorological values are passed as `None` / `np.nan` and handled by the pipeline's train-fitted `SimpleImputer`.
   - Missing inputs are flagged in responses with `was_imputed: true`, `observed_value: null`, `impact: "imputed_input"`, and `attribution_weight: 0.0`.
   - Unobserved features are never described as physically increasing or decreasing flood risk.

4. **Scenario Overrides Require Baseline**:
   - Scenario rainfall increments (+$\Delta$ mm) require observed baseline precipitation. If baseline weather is unavailable, the service degrades or rejects rather than inventing fabricated fallback numbers.
