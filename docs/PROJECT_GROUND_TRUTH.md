# DisasterPulse TN — Project Ground Truth

Canonical system reference, operating principles, allowable claims, source hierarchy, and definition of done for DisasterPulse TN.

---

## 1. Product Definition

**DisasterPulse TN** is an evidence-backed flood intelligence and risk-aware evacuation decision-support system for the state of **Tamil Nadu, India**.

### Central Workflow
$$\text{REAL DATA} \longrightarrow \text{TRAINED FLOOD MODEL} \longrightarrow \text{TAMIL NADU HAZARD SURFACE} \longrightarrow \text{REAL OSM ROAD NETWORK} \longrightarrow \text{ROAD FLOOD EXPOSURE} \longrightarrow \text{SURVIVOR GPS} \longrightarrow \text{REAL FACILITIES} \longrightarrow \text{FASTEST / BALANCED / SAFEST ROUTES} \longrightarrow \text{EXPLAINABLE RECOMMENDATION}$$

A user can supply **any valid geographic coordinate inside Tamil Nadu** and receive risk-evaluated routing over the genuine road network. The system never depends on fictional zones or synthetic graph topologies.

---

## 2. Product Philosophy

1. **LIVE**: Environmental inputs (rainfall, forecast, soil moisture) are retrieved from authentic external meteorological providers (e.g., Open-Meteo) for specific coordinates.
2. **LEARNED**: Flood hazard probability is derived from a machine learning model (FloodNow TN) trained reproducibly on historical observations (India Flood Inventory v3) with defensible weak-negative sampling and temporal cross-validation.
3. **VERIFIABLE**: Every data layer, facility record, and model output exposes provenance, retrieval timestamp, observation window, freshness, and status.

### Scientific Principle
- **ML** is used for uncertain flood occurrence prediction.
- **Geospatial algorithms** handle geometric intersections and clipping.
- **Graph routing** (OSM / GraphHopper) performs deterministic pathfinding.
- **Optimization** performs facility matching and resource allocation.
- **LLMs** are restricted to summarization, situational synthesis, and Tamil translation.
- **Core Motto**: *"AI predicts uncertainty, deterministic algorithms perform routing and optimization, and humans remain in control."*

---

## 3. Allowed vs. Forbidden Claims

### Allowed Claims (Verified by Implementation)
- Real Tamil Nadu state boundary and coordinate validation.
- Real OpenStreetMap road network geometry and routing.
- Real geocoded public hospitals and candidate evacuation facilities in Tamil Nadu.
- Coordinate-specific live and forecast rainfall from verified meteorological APIs.
- Real historical flood event training data (India Flood Inventory v3, Zenodo 16994648).
- Reproducibly trained, calibrated ML flood hazard model.
- Real graph routing from arbitrary coordinates within Tamil Nadu.
- Hazard-influenced route alternatives: FASTEST, BALANCED, SAFEST.
- Transparent provenance metadata for every provider.
- Model-based "what-if" rainfall scenario simulation.

### Strictly Forbidden Claims (Without Live Sourced Feeds)
- Live hospital bed counts or ICU occupancy (must be displayed as `null` / "Live bed availability unavailable").
- Live official shelter status (candidate facilities must be labeled `CANDIDATE EVACUATION FACILITY`).
- Live road closures or live traffic congestion (unless backed by verified live feeds).
- Guaranteed evacuation safety (always framed as decision support).
- Exact physical inundation depth in meters (model predicts occurrence/hazard probability).
- Emergency-authority or government certification.
- Labeling predicted road flood exposure as "confirmed road closure" (must be labeled `PREDICTED HIGH FLOOD EXPOSURE`).

---

## 4. Source Hierarchy

| Domain | Primary Source | Fallback / Policy | Freshness Policy |
|---|---|---|---|
| **Boundary** | Survey of India / Tamil Nadu State GeoJSON | Cached GeoJSON Polygon | Static verified boundary |
| **Weather** | Open-Meteo Hourly Forecast API (Coordinate-specific) | Cached reading (TTL 30m); never default to 0 | Max 60m; marked DEGRADED if stale |
| **Hydrology** | GloFAS / Open-Meteo River Discharge | Clearly labeled as modelled/reanalysis | Model run frequency |
| **Road Network** | OpenStreetMap (Geofabrik Southern Zone extract) | Embedded OSM Road Network Graph | Static extract with provenance |
| **Hospitals** | Open Government Data (OGD) / National Health Portal TN | Verified OSM health amenities | Sourced attributes only; beds = null |
| **Shelters** | OSM Emergency Shelters & Public Facilities | Labeled `CANDIDATE EVACUATION FACILITY` | Sourced attributes only |
| **Flood Training** | India Flood Inventory v3 (Zenodo 16994648) | Strict MD5 verification (`fea75a9ff9eba8fb328eaddfacd21d67`) | Static training baseline |

---

## 5. Machine Learning Semantics (FloodNow TN)

- **Target**: Flood occurrence likelihood at a location-time unit.
- **Positives**: Validated IFI v3 flood event records in Tamil Nadu / Southern India.
- **Negatives**: Spatio-temporally matched samples outside flood event windows in Tamil Nadu ("weak negatives", transparently documented).
- **Features**: `latitude`, `longitude`, `month_sin`, `month_cos`, `rainfall_24h_mm`, `rainfall_72h_mm`, `soil_moisture`, `elevation_m`, `slope_deg`, `distance_to_water_km`.
- **Excluded Features**: No post-event leakage features (`duration_days`, `affected_districts`, post-event damage text).
- **Validation**: Strict temporal split (`TimeSeriesSplit`). Outer test set held out.
- **Calibration**: Probability calibration via Brier score optimization.
- **Missing Model Behavior**: Returns HTTP 503 `MODEL_NOT_TRAINED` with training instructions. No synthetic mathematical formulas.

---

## 6. Routing Semantics

- **FASTEST**: Standard road network shortest travel time using posted road class speeds.
- **BALANCED**: Incorporates moderate detour penalties through flood hazard areas.
- **SAFEST**: Heavily penalizes or avoids high/critical hazard polygons.
- **Exposure Metric**: Independently evaluates geometry against hazard surface:
  $$\text{Exposure} = (\text{Distance}, \text{Max Hazard}, \text{Mean Hazard}, \text{High-Risk km}, \text{Critical-Risk km})$$
- **No Safe Route**: When all candidate paths cross critical flood hazards, the system returns `NO ROUTE CONFIDENTLY RECOMMENDED`.

---

## 7. Core Demonstration Flow

1. Open Tamil Nadu map -> Display live DATA HEALTH panel.
2. Enable model-predicted flood hazard layer.
3. Click any cell -> Display flood occurrence probability, weather inputs, and local feature contributions.
4. Enter Survivor Mode -> Click or provide survivor GPS inside Tamil Nadu.
5. System queries real candidate facilities within reach.
6. System calculates FASTEST and SAFEST routes, demonstrating hazard exposure differences.
7. System provides human-understandable destination recommendation explaining safety trade-offs.
8. Modify rainfall in Scenario Lab -> Rerun same trained model -> Hazard and routes adapt.
9. Open System Evidence page -> Audit that handcrafted records = 0 and all metrics are dynamic.

---

## 8. Definition of Done

- [ ] Zero synthetic Harbor City JSONs in production flood path (`zones.json`, `roads.json`, `hospitals.json`, etc.).
- [ ] No `z01`/`z02` identifiers in active frontend or backend API.
- [ ] Arbitrary coordinate support inside Tamil Nadu; out-of-state coordinates rejected.
- [ ] Chennai and Coimbatore use genuine coordinate-specific weather feeds.
- [ ] Missing weather never defaults to 0.0 rainfall.
- [ ] Real Tamil Nadu hospitals and shelters loaded with provenance and no fake bed counts.
- [ ] Real OSM road geometry and routing returned as GeoJSON.
- [ ] FloodNow TN model trained reproducibly with no post-event leakage features.
- [ ] Probability calibration evaluated and persisted.
- [ ] Route exposure analysis independently measures flood risk along paths.
- [ ] Hazard polygons demonstrably change SAFEST routing in automated tests.
- [ ] Survivor Mode recommends optimal destination with explainable rationale.
- [ ] System Evidence page displays dynamic metrics and provenance.
- [ ] Automated anti-slop test passes.
- [ ] Complete backend test suite and frontend build succeed.
