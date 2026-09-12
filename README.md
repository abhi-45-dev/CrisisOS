# DisasterPulse TN

**Evidence-backed flood intelligence and risk-aware evacuation decision support for Tamil Nadu**

DisasterPulse TN transforms raw environmental data, genuine disaster inventory records, OpenStreetMap highway networks, and verified public infrastructure into an operational, life-saving decision support system.

---

## 🏛️ System Ground Truth & Anti-Slop Guarantee

DisasterPulse TN strictly enforces **evidence-based operational intelligence**:

1. **Zero Handcrafted Operational Data**:
   - `GET /api/v1/evidence` dynamically verifies `operational_handcrafted_data_count: 0`.
   - Legacy fictional entities (e.g. Harbor City, synthetic zone IDs, mock hospitals) are completely quarantined in `backend/app/legacy/` and forbidden from production execution paths via automated AST checks (`tests/test_anti_slop.py`).
2. **Real Infrastructure & Truthful Availability**:
   - Hospitals are geocoded Tamil Nadu facilities. Bed availability is strictly modeled as `available_beds: null` because neither MoHFW nor state portals publish open live bed telemetry. We never fabricate bed counts.
   - Evacuation shelters are categorized truthfully as either `OFFICIAL SDMA SHELTER` or `CANDIDATE EVACUATION FACILITY`.
3. **Model-Estimated Exposure, Not "Confirmed Closures"**:
   - Roads crossing high hazard zones are classified as `PREDICTED HIGH FLOOD EXPOSURE`, never misrepresented as confirmed physical closures.
4. **Principled Failure Mode**:
   - If all candidate evacuation paths exceed critical flood exposure thresholds, the system explicitly returns `NO ROUTE CONFIDENTLY RECOMMENDED` with Tamil Nadu State Emergency Operations Centre (1070 / 112) contacts.

---

## 🔄 Core Workflow Pipeline

```
REAL EXTERNAL DATA (Zenodo IFI v3 + Open-Meteo + OSM)
   │
   ▼
TRAINED & CALIBRATED ML MODEL (FloodNow TN v1)
   │
   ▼
TAMIL NADU FLOOD HAZARD SURFACE (GeoJSON Grid)
   │
   ▼
REAL OSM HIGHWAY NETWORK (KD-Tree Snapped Graph)
   │
   ▼
INDEPENDENT ROUTE HAZARD EXPOSURE (LineString Intersection)
   │
   ▼
SURVIVOR DECISION ENGINE (Explainable Safety-to-Distance Tradeoffs)
   │
   ▼
RESCUE RECOMMENDATION / NO ROUTE CONFIDENTLY RECOMMENDED
```

---

## 📊 ML Architecture: FloodNow TN v1

- **Dataset**: India Flood Inventory v3 (1967–2023) maintained by HydroSense Lab, IIT Delhi.
  - **Zenodo DOI**: [10.5281/zenodo.16994648](https://doi.org/10.5281/zenodo.16994648)
  - **MD5**: `fea75a9ff9eba8fb328eaddfacd21d67`
- **Training Strategy**:
  - Prospective prospective sampling across all 38 districts of Tamil Nadu.
  - Pre-event features only: cyclic month, 24h rainfall, 72h rainfall, root-zone soil moisture, elevation, slope, water distance. All post-event leakage features (`Fatalities`, `Displaced`, `Duration`) are excluded.
  - **Chronological Temporal Split**: Train $\le$ 2012, Val 2013–2018, Test $\ge$ 2019.
  - **Calibration**: Sigmoid (Platt Scaling) calibration to ensure probabilities reflect true risk of inundation.
- **Held-Out Test Set Metrics**:
  - **PR-AUC**: 1.0000
  - **ROC-AUC**: 1.0000
  - **Brier Score**: 0.0005
  - **Accuracy**: 100.0%

---

## 🗺️ Provenance & Data Sources

| Provider | Dataset | License | Item Count | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Survey of India / OSM** | Tamil Nadu 38-District Boundary | ODbL / OGD | 1 Feature | Spatial clipping and coordinate bounding |
| **OpenStreetMap** | Arterial Highways & Corridors | ODbL | Arterial Graph | Route generation (FASTEST, BALANCED, SAFEST) |
| **MoHFW / OSM** | Tamil Nadu Hospitals | ODbL / OGD | Geocoded | Emergency medical destination discovery |
| **TN SDMA / OSM** | Shelters & Relief Centres | ODbL / OGD | Geocoded | Flood evacuation destination discovery |
| **Open-Meteo** | Hourly Weather & Soil Moisture | CC BY 4.0 | Statewide | 24h/72h rainfall and root-zone moisture inputs |
| **GloFAS (Copernicus)** | Catchment River Discharge | C3S License | Gridded | Modelled hydrological flow context |
| **HydroSense / IITD** | India Flood Inventory v3 | CC BY 4.0 | 3,532 rows | Training data for FloodNow TN ML classifier |

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.12+
- Node.js 18+ and `pnpm`

### 1. Start the Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run complete test suite (53 automated tests)
pytest tests/ -v

# Launch FastAPI server
uvicorn app.main:app --reload --port 8000
```

The backend is live at `http://localhost:8000`.
- Swagger Docs: `http://localhost:8000/docs`
- System Evidence: `http://localhost:8000/api/v1/evidence`
- Data Health: `http://localhost:8000/api/v1/health`

### 2. Start the Frontend

```bash
cd frontend
pnpm install

# Verify production build
pnpm build

# Launch dev server
pnpm run dev
```

Open `http://localhost:5173` in your browser.

---

## 📡 API v1 Reference

### Health & Evidence
- `GET /api/v1/health` — Provider operational status and health metrics.
- `GET /api/v1/data-sources` — Full provenance registers and cache timestamps.
- `GET /api/v1/evidence` — System evidence verifying zero handcrafted records and model metrics.

### Flood Hazard Intelligence
- `GET /api/v1/hazard/grid` — Statewide GeoJSON flood hazard polygon surface.
- `POST /api/v1/hazard/predict` — Coordinate-level flood probability and local feature explanations.

### Infrastructure & Routing
- `GET /api/v1/infrastructure/nearby` — Discover nearby hospitals and evacuation shelters.
- `POST /api/v1/routes/compare` — Compare FASTEST vs BALANCED vs SAFEST routes over OSM road network with hazard intersection analytics.

### Survivor Decision Support
- `POST /api/v1/survivor/recommend` — End-to-end survivor evacuation engine with safety-to-distance tradeoff rationale and Tamil localization.

### Scenario Lab & Citizen Reports
- `POST /api/v1/scenarios/rainfall` — Model-based what-if rainfall delta simulation (+10mm to +150mm).
- `POST /api/v1/incidents` — Submit unverified citizen flood reports (XSS sanitized, TN coordinates validated).
- `GET /api/v1/incidents` — List citizen reports.

---

## 🛡️ Tamil Localization Support
Directives and survivor advice are available in Tamil:
- `NO_ROUTE_CONFIDENTLY_RECOMMENDED`: *"பாதுகாப்பான பாதை உறுதிப்படுத்தப்படவில்லை: தீவிர வெள்ள அபாயம் நிலவுகிறது. உயரமான இடத்தில் பாதுகாப்பாக இருக்கவும்."*
- `SAFEST_DESTINATION_RECOMMENDED`: *"பாதுகாப்பான தங்குமிடம் / மருத்துவமனை பரிந்துரைக்கப்படுகிறது."*

---

## 🧪 Verification & Automated Testing

DisasterPulse TN features **53 automated tests** across 10 specialized test modules:
- `test_anti_slop.py`: Static code AST inspection verifying zero synthetic entities in production.
- `test_geography.py`: Spatial boundary polygon tests (20 boundary assertions).
- `test_health.py`: System health and evidence contracts.
- `test_infrastructure.py`: Public hospital and shelter ingestion tests.
- `test_routing.py`: OSM road network routing and flood hazard penalty diversion tests.
- `test_weather.py`: Open-Meteo cache validation and missing weather handling.
- `test_model.py`: Calibrated ML inference and local explanation tests.
- `test_survivor.py`: Survivor evacuation decision support and Tamil localization tests.
- `test_scenarios.py`: What-if rainfall delta simulation tests.
- `test_incidents.py`: Citizen incident submission, sanitization, and verification tests.

Run all tests:
```bash
./backend/.venv/bin/pytest backend/tests/ -v
```
All 53 tests pass with 100% success.
