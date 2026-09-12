# DisasterPulse TN — Build Status

Tracking ledger updated after each implementation phase.

---

## Current Status Overview

- **Overall Progress**: All Phases Complete (Phases 0 through 16)
- **Active Model Version**: FloodNow TN v1 (Calibrated RandomForestClassifier)
- **Active Environment**: macOS / Python 3.12 / React 19 / Vite
- **Test Suite Status**: 53 tests passing across 10 test modules (100% pass rate)
- **Frontend Build**: `pnpm build` passing with zero errors

---

## Phase Execution Ledger

### Phase 0: Truth + Audit
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Researched entire codebase (backend simulation, toy data, ML models, frontend).
  - Authored canonical docs: `PROJECT_GROUND_TRUTH.md`, `AUDIT_V3.md`, `IMPLEMENTATION_PLAN.md`, `DATA_PROVENANCE.md`, `BUILD_STATUS.md`.
  - Defined explicit allowed/forbidden claims and definition of done.

---

### Phase 1: Decouple Synthetic Prototype
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Decoupled production runtime from `zones.json`, `roads.json`, `hospitals.json`, `shelters.json`, `resources.json`, `disaster_overlays.json`.
  - Quarantined legacy synthetic services, simulation, agents, and router into `backend/app/legacy/`.
  - Removed cyclone/earthquake from active production API paths.
  - Implemented automated anti-slop check in `backend/tests/test_anti_slop.py`.

---

### Phase 2: Provenance Foundation & Data Health
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `SourceMetadata`, `ProviderStatus`, and `BaseProvider` in `app.providers.base`.
  - Built `ProviderManager` in `app.providers.manager` tracking all system providers.
  - Created endpoints `GET /api/v1/health`, `GET /api/v1/data-sources`, `GET /api/v1/evidence`.
  - Verified zero handcrafted operational records dynamically.

---

### Phase 3: Tamil Nadu Geography
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Integrated official Tamil Nadu boundary GeoJSON (`backend/app/data/tamil_nadu_boundary.geojson`).
  - Implemented `BoundaryProvider` with fast Shapely prepared polygon `is_inside_tamil_nadu()`.
  - Implemented coordinate validation with HTTP 422 rejection for out-of-bounds queries.
  - Implemented statewide prediction grid generator.

---

### Phase 4: Real Infrastructure Ingestion
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Ingested geocoded Tamil Nadu public hospitals dataset (`backend/app/data/tamil_nadu_hospitals.json`) with strict enforcement of `available_beds = None` (no fabricated bed counts).
  - Ingested Tamil Nadu evacuation shelters dataset (`backend/app/data/tamil_nadu_shelters.json`) with classification (`CANDIDATE EVACUATION FACILITY` vs `OFFICIAL EMERGENCY SHELTER`).
  - Built `HospitalProvider` and `EvacuationFacilityProvider` in `app.providers.infrastructure_provider`.
  - Created endpoints `GET /api/v1/infrastructure/nearby`, `/hospitals`, `/shelters`.

---

### Phase 5 & 10: Real Road Network Routing & Hazard Exposure
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Extracted and assembled genuine Tamil Nadu OpenStreetMap arterial highway network (`backend/app/data/tamil_nadu_roads.json`).
  - Implemented `EmbeddedOsmRoutingProvider` in `app.routing.provider` with KD-tree node snapping and Dijkstra/A* routing.
  - Provided `docker-compose.graphhopper.yml` for self-hosted container deployments.
  - Implemented `analyze_route_exposure()` in `app.routing.exposure` calculating distance, duration, max flood probability, high/critical risk km, and GeoJSON LineStrings.
  - Implemented `POST /api/v1/routes/compare` comparing FASTEST, BALANCED, and SAFEST profiles.

---

### Phase 6: Environmental Data & Hydrology
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `WeatherProvider` in `app.providers.weather_provider` querying Open-Meteo API with file-based TTL caching and manifest generation.
  - Enforced that missing weather data is recorded as `None`, NEVER defaulted to 0.0.
  - Supported multi-window precipitation accumulations (1h, 3h, 6h, 24h, 72h) and root-zone soil moisture.
  - Implemented `HydrologyProvider` in `app.providers.hydrology_provider` for modelled GloFAS river discharge, explicitly labeled as simulation/reanalysis (not physical gauge).

---

### Phase 7 & 8: Training Data V3 & FloodNow TN ML Model
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Downloaded real India Flood Inventory v3 (Zenodo record 16994648) with MD5 hash validation (`fea75a9ff9eba8fb328eaddfacd21d67`).
  - Built `backend/ml/dataset_builder.py` generating 3,532 prospective samples with weak-negative non-event sampling across Tamil Nadu. Excluded all post-event leakage variables.
  - Produced `training_dataset.parquet` and `training_manifest.json`.
  - Built `backend/ml/trainer.py` with temporal chronological partitions (train <=2012, val 2013-2018, test >=2019).
  - Calibrated probabilities with Sigmoid calibration.
  - Evaluated on held-out test partition: PR-AUC 1.0, ROC-AUC 1.0, Brier Score 0.0005.
  - Serialized all artifacts: `model.joblib`, `metrics.json`, `metadata.json`, `model_card.json`, `feature_schema.json`, `feature_importance.json`.
  - Built `FloodNowTNInferenceService` with local per-prediction explanations.
  - Wrapped as `FloodModelProvider` and registered in `ProviderManager`.

---

### Phase 9: Statewide Hazard Surface
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `HazardService` in `backend/app/services/hazard_service.py`.
  - Endpoints: `GET /api/v1/hazard/grid`, `POST /api/v1/hazard/predict`.
  - Renders statewide GeoJSON polygon grid clipped strictly to Tamil Nadu with per-cell probabilities and risk bands.

---

### Phase 11: Survivor Evacuation Decision Support
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `SurvivorService` in `backend/app/services/survivor_service.py`.
  - Endpoint: `POST /api/v1/survivor/recommend`.
  - Features:
    - Queries real nearby hospitals and shelters within search radius.
    - Evaluates FASTEST vs SAFEST routes over OSM road network.
    - Penalizes routes with high/critical flood exposure.
    - Plain-language explanation justifying recommendation over closer alternatives.
    - "NO ROUTE CONFIDENTLY RECOMMENDED" verdict when all corridors are submerged.
    - Tamil language localization support.

---

### Phase 13: Scenario Lab
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `ScenarioService` in `backend/app/services/scenario_service.py`.
  - Endpoint: `POST /api/v1/scenarios/rainfall`.
  - Re-evaluates trained FloodNow TN classifier with rainfall deltas (+10mm to +150mm).
  - Computes baseline vs scenario probability deltas and newly escalated zone metrics.

---

### Phase 14: Citizen Incidents & Tamil Localization
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Implemented `IncidentService` in `backend/app/services/incident_service.py`.
  - Endpoints: `POST /api/v1/incidents`, `GET /api/v1/incidents`, `GET /api/v1/incidents/nearby`.
  - Enforced strict `UNVERIFIED` default status and XSS HTML sanitization.
  - Tamil emergency strings for survivor guidance and alerts.

---

### Phase 12 & 15: Frontend Overhaul
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Rewrote `frontend/src/types.ts` and `frontend/src/api.ts` to bind strictly to `/api/v1`.
  - Built interactive `LeafletCrisisMap.tsx` with hazard polygon styling, facility markers, route risk visualization, and popup sanitization.
  - Rewrote `frontend/src/App.tsx` with 5 dedicated functional tabs:
    1. Tactical Crisis Map
    2. Survivor Support
    3. Scenario Lab
    4. Citizen Incidents
    5. Evidence & Provenance
  - Verified with `pnpm build` (builds in ~340ms with 0 type errors).

---

### Phase 16: Verification, Testing & Hardening
- **STATUS**: COMPLETED
- **DELIVERABLES**:
  - Complete backend test suite passing: 53 / 53 tests.
  - Zero synthetic Harbor City entities in production paths (`test_anti_slop.py`).
  - Zero handcrafted records confirmed by `GET /api/v1/evidence`.
