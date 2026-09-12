# FloodNow TN v2 — Feature Provenance & Production Alignment

This document details the exact scientific lineage, historical training definitions, live operational providers, and missing-data policies for every feature in **FloodNow TN v2**.

---

## 🔬 Feature Provenance Matrix

| Feature | Historical Training Source | Training Definition | Live / Production Source | Live Definition | Same Semantics? | Missing-Data Handling | Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| `latitude` | India Flood Inventory v3 + TN District Centroid Registry | WGS84 latitude coordinate (degrees) | User GPS coordinate / Grid cell centroid | WGS84 latitude coordinate (degrees) | **YES** | Mandatory. Value must be within [-90.0, 90.0] | Verified within Tamil Nadu state boundary geometry. |
| `longitude` | India Flood Inventory v3 + TN District Centroid Registry | WGS84 longitude coordinate (degrees) | User GPS coordinate / Grid cell centroid | WGS84 longitude coordinate (degrees) | **YES** | Mandatory. Value must be within [-180.0, 180.0] | Verified within Tamil Nadu state boundary geometry. |
| `month_sin` | Calendar timestamp of event antecedent window | sin(2*pi*month/12) cyclic month representation | Current UTC/IST inference timestamp | sin(2*pi*month/12) cyclic month representation | **YES** | Imputed with current month cyclic transform | Captures cyclical seasonality (SW Monsoon vs NE Monsoon). |
| `month_cos` | Calendar timestamp of event antecedent window | cos(2*pi*month/12) cyclic month representation | Current UTC/IST inference timestamp | cos(2*pi*month/12) cyclic month representation | **YES** | Imputed with current month cyclic transform | Paired with month_sin for smooth periodic encoding. |
| `rain_1h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Hourly precipitation accumulation in the 1 hour prior to prediction timestamp (mm) | Open-Meteo Forecast API (precipitation) | Hourly precipitation in the most recent completed past hour (mm) | **YES** | Median-imputed via train-fitted SimpleImputer | Captures immediate cloudburst / flash-flood intensity. |
| `rain_3h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Cumulative precipitation in the 3 hours prior to prediction timestamp (mm) | Open-Meteo Forecast API (precipitation) | Cumulative precipitation across the 3 trailing hourly slots (mm) | **YES** | Median-imputed via train-fitted SimpleImputer | Captures short-window storm cell accumulation. |
| `rain_6h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Cumulative precipitation in the 6 hours prior to prediction timestamp (mm) | Open-Meteo Forecast API (precipitation) | Cumulative precipitation across the 6 trailing hourly slots (mm) | **YES** | Median-imputed via train-fitted SimpleImputer | Captures mesoscale convective rainfall accumulation. |
| `rain_24h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Antecedent 24-hour precipitation accumulation ending before prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Trailing 24-hour observed precipitation accumulation (mm) | **YES** | Median-imputed via train-fitted SimpleImputer | Primary monsoon daily rainfall volume indicator. |
| `rain_72h_mm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Antecedent 72-hour precipitation accumulation ending before prediction cutoff (mm) | Open-Meteo Forecast API (precipitation) | Trailing 72-hour observed precipitation accumulation (mm) | **YES** | Median-imputed via train-fitted SimpleImputer | Multi-day depression/cyclone catchment saturation driver. |
| `soil_moisture_0_7cm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Volumetric soil water content in topsoil layer (0-7 cm, m3/m3) | Open-Meteo Forecast API (soil_moisture_0_to_7cm) | Latest topsoil volumetric soil moisture nowcast (m3/m3) | **YES** | Median-imputed via train-fitted SimpleImputer | High topsoil moisture dramatically reduces infiltration capacity. |
| `soil_moisture_7_28cm` | Open-Meteo Historical Archive (ERA5 / ERA5-Land reanalysis) | Volumetric soil water content in root-zone layer (7-28 cm, m3/m3) | Open-Meteo Forecast API (soil_moisture_7_to_28cm) | Latest root-zone volumetric soil moisture nowcast (m3/m3) | **YES** | Median-imputed via train-fitted SimpleImputer | Root-zone saturation indicates sub-surface catchment saturation. |

---

## Explicitly Rejected / Dropped Features

The following features present in legacy FloodNow TN v1 were **permanently removed** for scientific validity:

1. `elevation_m` (Synthetic):
   - Reason for removal: In v1, elevation was hard-coded based on arbitrary longitude cutoffs (25.0 if lon > 79.5 else 120.0). This fabricated physical variables and had no genuine empirical basis.
   - Status: DROPPED globally.

2. `slope_deg` (Synthetic):
   - Reason for removal: In v1, slope was hardcoded based on longitude (1.2 if lon > 79.5 else 2.5).
   - Status: DROPPED globally.

3. `distance_to_water_km` (Synthetic):
   - Reason for removal: In v1, distance to water was approximated with crude longitude rules (2.5 if lon > 79.8 else 8.0).
   - Status: DROPPED globally.

---

## Anti-Leakage Invariants

1. Single Feature Function:
   - get_environmental_features(latitude, longitude, prediction_timestamp) does not take any target label or positive/negative flag.
   - Positives and weak negatives query identical variables through identical code paths.

2. Pre-Event Time Semantics:
   - For historical flood events with date-level resolution (ending in 00:00), antecedent evaluation windows end strictly at 23:00 IST on the preceding day.
   - No precipitation occurring on or after the event date is ever included in predictors.

3. Missingness Transparency:
   - Missing meteorological values are preserved as NaN / None and handled by the pipeline fitted SimpleImputer. They are never coerced to 0.0 or synthetic values.
