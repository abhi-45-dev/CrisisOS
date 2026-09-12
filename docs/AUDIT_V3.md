# DisasterPulse TN — Repository Audit V3

Comprehensive audit of the codebase prior to the DisasterPulse TN transformation. Every synthetic dependency, architectural flaw, and ML limitation is documented here.

---

## 1. Fictional Operational World (Harbor City)

### Affected Files
- `backend/app/services/crisis_service.py`
- `backend/data/zones.json` (12 zones: `z01` Downtown through `z12` Greenbelt)
- `backend/data/roads.json` (24 roads: `road_01` through `road_24`, e.g., "Harbor Parkway", "Marina Boulevard")
- `backend/data/hospitals.json` (5 fake hospitals: `h01` "Harbor General Hospital" to `h05`)
- `backend/data/shelters.json` (6 fake shelters: `s01` "Downtown Convention Shelter" to `s06`)
- `backend/data/resources.json` (13 fake resource allocations: `res_01` to `res_13`)
- `backend/data/incidents.json`
- `backend/data/disaster_overlays.json`

### Problem
The entire crisis simulation runs on a synthetic, 12-zone Harbor City environment. Any routing, risk scoring, or shelter assignment queries rely on fictional IDs (`z01`, `h01`, `road_01`) rather than real Tamil Nadu coordinates.

---

## 2. Toy Routing Architecture

### Affected Files
- `backend/app/simulation/routes.py`
- `backend/app/api/router.py` (endpoints `/routes`, `/geo/nearest-shelter`, etc.)

### Identified Issues
- Dijkstra algorithm runs over fictional zone IDs (`origin: "z01"`, `destination: "z02"`).
- Contains arbitrary hand-coded penalty heuristics:
  - `DANGEROUS_PENALTY = 45.0`
  - `CONGESTED_MULTIPLIER = 2.2`
  - `DANGEROUS_MULTIPLIER = 3.5`
  - `cost += float(road.risk_score) * 0.15`
- Frontend UI forces the user to pick zone IDs from a dropdown instead of using GPS or map clicks.

---

## 3. Hardcoded Weather & Zero Fallback

### Affected Files
- `backend/app/services/weather_service.py`
- `backend/app/simulation/risk.py`

### Identified Issues
- Hardcoded Chennai coordinate: `CHENNAI_LAT = 13.0827`, `CHENNAI_LON = 80.2707`. Coimbatore, Madurai, or Nilgiris are evaluated using Chennai rainfall.
- Priority inversion bug: In `risk.py`, fictional `zone.parameters.rainfall_mm` is checked *before* live Open-Meteo rainfall. Fictional disaster overlays therefore mask live weather.
- Zero-rainfall fallback: If Open-Meteo is unavailable, `risk.py` sets `rainfall_24h = 0.0` ("unavailable; zero-observation input"). Missing data must never be treated as zero precipitation.

---

## 4. Machine Learning Limitations in Flood v2

### Affected Files
- `backend/ml/flood_ml.py`
- `backend/ml/training/train_flood_model.py`
- `backend/ml/predictor.py`

### Identified Issues
1. **Wrong Problem Formulation**: The model predicts historical flood `Severity` conditional on an event already occurring. It cannot predict flood *occurrence/hazard probability* for prospective space-time coordinates.
2. **Leakage Features**: Predictors include `duration_days` and `affected_districts`. These are post-event impact metrics unavailable prospective to a flood event.
3. **Data Imbalance & Lack of Negatives**: The training dataset consists entirely of positive historical flood events from IFI v3. It lacks genuine or defensible non-flood observations.
4. **Label Normalization Bug**: Label `'VERY_SEVERE'` is trained with an underscore, but inference logic in `predictor.py` checks `'VERY SEVERE'` with a space, resulting in severe probability omission.
5. **Hyperparameter CV Leakage**: Outer split is chronological, but inner tuning uses `RandomizedSearchCV(cv=3)` instead of `TimeSeriesSplit`.
6. **Model Artifact Disagreement**: If tuned XGBoost underperforms, `selected` is updated to the tuned model while `best_name` points to Random Forest, creating conflicting metadata.
7. **Explainability**: SHAP artifact stores only global feature importances, but is rendered in UI as if explaining local risks.

---

## 5. Hand-Authored Evacuation & Resource Heuristics

### Affected Files
- `backend/app/simulation/evacuation.py`
- `backend/app/simulation/resources.py`

### Identified Issues
- Hardcoded evacuation fractions:
  - `CRITICAL: 0.85`
  - `HIGH: 0.55`
  - `MODERATE: 0.15`
  - `LOW: 0.0`
- Heuristic resource demand: `_need_for_zone()` calculates arbitrary pump and team counts based on `population // 8000 * 2`. These were framed as AI-driven recommendations.

---

## 6. Fictional Agent Prompts & Context

### Affected Files
- `backend/app/agents/analyst.py`
- `backend/app/agents/planner.py`
- `backend/app/agents/critic.py`
- `backend/app/agents/what_if.py`

### Identified Issues
Agents explicitly reference `s02`, `res_04`, `road_22`, `h03` and Harbor City zones. Because LLMs were supplied fictional context, their outputs hallucinated advice for non-existent geography.

---

## 7. Frontend Flaws

### Affected Files
- `frontend/src/components/LeafletCrisisMap.tsx`
- `frontend/src/App.tsx`
- `frontend/src/api.ts`

### Identified Issues
- Popups construct raw HTML via string concatenation without sanitization, creating potential XSS injection vulnerabilities.
- Multi-disaster tabs (Cyclone, Earthquake) clutter the UI despite having no valid ML foundation or geographic datasets.
- Routes view relies on zone select boxes (`z01` to `z12`).

---

## 8. Summary of Components to Preserve vs. Replace

| Component | Status | Action |
|---|---|---|
| IFI v3 Zenodo Ingestion & Hash Validation | Keep | Reuse in `FloodNow TN` dataset pipeline |
| Temporal Chronological Split Pattern | Keep | Refine with strict `TimeSeriesSplit` inner CV |
| React/Vite/Tailwind + Leaflet Shell | Keep | Preserve visual theme, replace data contracts |
| Haversine Distance Calculation | Keep | Retain in `app/geo.py` |
| Harbor City JSONs & Simulation | Quarantine | Move to `app/legacy/`, remove from production |
| Severity Classifier | Replace | Build `FloodNow TN` occurrence model |
| Dijkstra over Zone IDs | Replace | Replace with OSM road network routing |
| Fixed Chennai Weather | Replace | Implement coordinate-specific `WeatherProvider` |
