# DisasterPulse TN — Master Implementation Plan

Engineering roadmap for the end-to-end transformation into DisasterPulse TN.

---

## Phases & Execution Order

### Phase 0: Ground Truth & Audit Documentation
- [x] Create `docs/PROJECT_GROUND_TRUTH.md`
- [x] Create `docs/AUDIT_V3.md`
- [x] Create `docs/IMPLEMENTATION_PLAN.md`
- [x] Create `docs/DATA_PROVENANCE.md`
- [x] Create `docs/BUILD_STATUS.md`

### Phase 1: Decouple Fictional Operational World & Anti-Slop
- Quarantine synthetic entities (`zones.json`, `roads.json`, `hospitals.json`, `shelters.json`, `resources.json`, `disaster_overlays.json`) into `app/legacy/`.
- Remove cyclone/earthquake from active production API routes.
- Implement `tests/test_anti_slop.py` asserting no Harbor City or toy identifiers appear in active production endpoints.

### Phase 2: Provenance Foundation & Data Health
- Implement `SourceMetadata` schema (Pydantic v2) and `BaseProvider[T]`.
- Implement `GET /api/v1/health` and `GET /api/v1/data-sources`.
- Implement `GET /api/v1/evidence` exposing verified dynamic facts.
- Implement Data Health UI status badges.

### Phase 3: Tamil Nadu Geographic Foundation
- Integrate official Tamil Nadu boundary GeoJSON.
- Implement `is_inside_tamil_nadu(lat, lon) -> bool` with fast Shapely spatial index.
- Implement statewide grid generator clipped to state boundary.
- Implement automated tests for in-boundary and out-of-boundary rejection.

### Phase 4: Real Infrastructure Ingestion
- Ingest real Tamil Nadu geocoded public hospitals (OGD / OSM verified).
- Ingest real evacuation facilities (OSM emergency shelters + candidate centres tagged `CANDIDATE EVACUATION FACILITY`).
- Explicitly enforce `available_beds: null` (no fabricated live bed counts).
- Implement spatial query `GET /api/v1/infrastructure/nearby`.

### Phase 5: Real Road Network Routing
- Build `RoutingProvider` abstraction with two engines:
  1. `EmbeddedOsmRoutingProvider`: In-process Tamil Nadu OSM road graph with KDTree spatial indexing and A* search.
  2. `GraphHopperProvider`: Dockerized GraphHopper with Southern Zone OSM PBF.
- Support coordinate-to-coordinate routing returning GeoJSON LineStrings.
- Centralize and document routing policy profiles (`FASTEST`, `BALANCED`, `SAFEST`).

### Phase 6: Environmental Data & Hydrology
- Implement coordinate-specific `WeatherProvider` using Open-Meteo.
- Support `rain_1h`, `rain_3h`, `rain_6h`, `rain_24h`, `rain_72h`, `soil_moisture_0_7cm`, `soil_moisture_7_28cm`.
- Guarantee that missing weather never silently defaults to 0.0 rainfall.
- Implement `HydrologyProvider` for modelled GloFAS river discharge.
- Implement caching with TTL and manifest metadata.

### Phase 7: Training Data Pipeline V3 (FloodNow TN)
- Ingest IFI v3 positives with Zenodo MD5 validation.
- Construct defensible weak-negative samples across Tamil Nadu districts.
- Extract prospective meteorological and terrain features.
- Eliminate post-event leakage variables (`duration_days`, `affected_districts`).
- Output reproducible `training_dataset.parquet` and `training_manifest.json`.

### Phase 8: FloodNow TN Machine Learning Model
- Train model candidates: DummyClassifier, LogisticRegression, RandomForestClassifier.
- Time-aware validation using chronological splits and `TimeSeriesSplit` tuning.
- Calibrate model probabilities and evaluate Brier score.
- Compute local per-prediction feature contributions.
- Persist artifacts (`model.joblib`, `metrics.json`, `metadata.json`, `model_card.json`, etc.).
- Ensure missing model returns HTTP 503 `MODEL_NOT_TRAINED` with no heuristic fallback.

### Phase 9: Statewide Hazard Map
- Generate batch hazard surface for Tamil Nadu.
- Expose GeoJSON endpoint `GET /api/v1/hazard/grid`.
- Leaflet polygon layer with risk band color coding and cell-click inspection (probability + local explanations).

### Phase 10: Risk-Aware Routing & Exposure Analysis
- Implement independent route hazard exposure calculation:
  `distance_km`, `duration_min`, `max_flood_probability`, `mean_flood_probability`, `high_risk_distance_km`, `critical_risk_distance_km`, `hazard_segments`.
- Demonstrate SAFEST route detours around flood hazard in automated tests.

### Phase 11: Survivor Mode
- UI flow: "I NEED HELP" -> GPS / map-click input.
- Discover candidate facilities and evaluate route safety.
- Recommend optimal reachable destination with human-understandable explanation.
- Handle `NO ROUTE CONFIDENTLY RECOMMENDED` when risk exceeds thresholds.

### Phase 12: System Evidence & UI Polish
- Render DATA HEALTH panel and SYSTEM EVIDENCE page.
- Map layer toggles for Hazard, Hospitals, Shelters, Incidents, Routes.
- Sanitize all Leaflet popup HTML to eliminate XSS.

### Phase 13: Scenario Lab
- Re-run the SAME trained FloodNow TN model with modified rainfall parameters.
- Provide side-by-side Baseline vs. Scenario delta comparisons.

### Phase 14: Citizen Incidents & Tamil Localization
- Implement incident reporting (`POST /api/v1/incidents`) with default `UNVERIFIED` status.
- Provide Tamil translation for survivor guidance and emergency alerts.

### Phase 15: Grounded AI Agents
- LLM agents restricted to structured backend state. No invented facts.

### Phase 16: Verification, Testing & Hardening
- Complete pytest suite.
- Frontend production build.
- Documentation and README rewrite.
