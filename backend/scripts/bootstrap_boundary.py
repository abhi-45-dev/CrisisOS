from __future__ import annotations

import json
import httpx
from pathlib import Path

from app.providers.manifest import save_manifest, compute_sha256

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FILE = DATA_DIR / "tamil_nadu_boundary.geojson"

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search?state=Tamil+Nadu&country=India&format=geojson&polygon_geojson=1"
)


def bootstrap_boundary() -> Path:
    """Download authentic Tamil Nadu OpenStreetMap administrative boundary and generate manifest."""
    print("Fetching authentic Tamil Nadu boundary from OpenStreetMap Nominatim...")
    headers = {"User-Agent": "DisasterPulseTN-BoundaryBootstrap/1.0 (disasterpulse-tn@local)"}
    
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(NOMINATIM_URL, headers=headers)
        resp.raise_for_status()
        raw_data = resp.json()

    features = raw_data.get("features", [])
    if not features:
        raise ValueError("Nominatim returned 0 features for Tamil Nadu")

    feature = features[0]
    geom = feature.get("geometry", {})
    geom_type = geom.get("type")
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geom_type}")

    point_count = sum(
        len(ring) for poly in (geom["coordinates"] if geom_type == "MultiPolygon" else [geom["coordinates"]]) for ring in poly
    )
    print(f"Retrieved {geom_type} with {point_count} coordinate vertices.")

    # Wrap as standard clean FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "name": "Tamil_Nadu_State_Boundary_OSM",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "osm_id": feature.get("properties", {}).get("osm_id", 96905),
                    "osm_type": feature.get("properties", {}).get("osm_type", "relation"),
                    "name": "Tamil Nadu",
                    "admin_level": 4,
                    "boundary": "administrative",
                    "country": "India",
                    "source": "OpenStreetMap / Nominatim",
                    "vertex_count": point_count,
                },
                "geometry": geom,
            }
        ],
    }

    with TARGET_FILE.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)

    sha = compute_sha256(TARGET_FILE)
    print(f"Saved {TARGET_FILE} (SHA-256: {sha})")

    manifest = save_manifest(
        dataset="tamil_nadu_boundary",
        provider="OpenStreetMap Contributors / Nominatim",
        source_url=NOMINATIM_URL,
        license="Open Database License (ODbL) / OGD",
        record_count=1,
        generated_by="backend.scripts.bootstrap_boundary",
        transformation="OSM administrative boundary relation extracted to WGS84 GeoJSON MultiPolygon",
        filepath=TARGET_FILE,
        sha256=sha,
        extra={
            "geometry_type": geom_type,
            "vertex_count": point_count,
            "osm_relation_id": feature.get("properties", {}).get("osm_id", 96905),
        },
    )
    print("Generated manifest:", manifest)
    return TARGET_FILE


if __name__ == "__main__":
    bootstrap_boundary()
