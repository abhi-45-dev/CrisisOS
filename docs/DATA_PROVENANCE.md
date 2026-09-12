# DisasterPulse TN — Data Provenance Register

Auditable provenance, licensing, retrieval details, and freshness semantics for every data source integrated into DisasterPulse TN.

---

## Provenance Summary Table

| Domain | Provider | Dataset / Collection | License / Terms | Retrieval Method | Freshness Policy |
|---|---|---|---|---|---|
| **Boundary** | Survey of India / OSM Admin Boundaries | Tamil Nadu State Boundary Polygon | Open Data / ODbL | Bundled GeoJSON (`backend/app/data/tamil_nadu_boundary.geojson`) | Static verified geometry |
| **Road Network** | OpenStreetMap / Geofabrik | Southern Zone India Road Extract | ODbL 1.0 | PBF extract / In-process road network graph | Static network baseline |
| **Meteorology** | Open-Meteo | Hourly Weather Forecast & Soil Moisture API | CC-BY 4.0 | REST API (`https://api.open-meteo.com/v1/forecast`) | Real-time / 30m cache TTL |
| **Hydrology** | GloFAS / Open-Meteo Flood API | Modelled River Discharge & Runoff | Copernicus Open Access / CC-BY 4.0 | REST API (`https://flood-api.open-meteo.com/v1/flood`) | Model run updates (daily) |
| **Hospitals** | Open Government Data (OGD) / National Health Portal TN / OSM | Tamil Nadu Public Hospitals & Health Centres | Government Open Data License India (GODL) / ODbL | Geocoded registry with district indexing | Static baseline, beds = null |
| **Evacuation Facilities** | OpenStreetMap / Disaster Relief Authorities | Verified Emergency Shelters & Community Centres | ODbL 1.0 | Tagged shelters + candidate assembly halls | Labeled `CANDIDATE EVACUATION FACILITY` |
| **Historical Floods** | HydroSense Lab / IIT Delhi / IMD | India Flood Inventory v3 (1967–2023) | CC-BY 4.0 (Zenodo record 16994648) | Direct download with MD5 checksum verification | Static historical baseline |

---

## Provider Status Lifecycle

Every provider exposes one of five discrete health states:

- **LIVE**: Fresh data retrieved within the defined TTL directly from upstream API.
- **READY**: Static baseline or cached real data validated and available for processing.
- **STALE**: Data present in cache but past expiration TTL; awaiting background refresh.
- **DEGRADED**: Upstream API returned an error or rate limit; operating with cached fallback or reduced precision.
- **UNAVAILABLE**: Provider offline, uninitialized, or required dataset missing; downstream requests fail explicitly.

---

## Dataset Hash Registry

| Dataset Name | File Path | Format | Expected Hash / MD5 |
|---|---|---|---|
| India Flood Inventory v3 | `backend/ml/data/India_Flood_Inventory_v3.csv` | CSV | `fea75a9ff9eba8fb328eaddfacd21d67` |
| Tamil Nadu Boundary | `backend/app/data/tamil_nadu_boundary.geojson` | GeoJSON | Verified polygon enclosing 38 districts |
| Tamil Nadu Hospitals | `backend/app/data/tamil_nadu_hospitals.json` | JSON | Validated geocoded registry |
| Tamil Nadu Shelters | `backend/app/data/tamil_nadu_shelters.json` | JSON | Validated geocoded registry |
| Tamil Nadu Road Graph | `backend/app/data/tamil_nadu_roads.json` | JSON/Graph | OSM primary/secondary road network |
