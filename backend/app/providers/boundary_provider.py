from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep

from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata
from app.providers.manifest import load_manifest


class BoundaryProvider(BaseProvider):
    """Provides validated Tamil Nadu boundary geometry and coordinate verification."""

    def __init__(self, geojson_path: Path | None = None):
        self._geojson_path = geojson_path or (
            Path(__file__).resolve().parent.parent / "data" / "tamil_nadu_boundary.geojson"
        )
        self._raw_geojson: dict[str, Any] = {}
        self._geometry: Polygon | MultiPolygon | None = None
        self._prepared_polygon = None
        self._manifest: dict[str, Any] | None = None
        self._load_boundary()

    def _load_boundary(self) -> None:
        if not self._geojson_path.exists():
            raise FileNotFoundError(f"Tamil Nadu boundary GeoJSON missing at {self._geojson_path}")

        with self._geojson_path.open("r", encoding="utf-8") as f:
            self._raw_geojson = json.load(f)

        features = self._raw_geojson.get("features", [])
        if not features:
            raise ValueError("Boundary GeoJSON contains no features")

        geom = shape(features[0]["geometry"])
        self._geometry = geom
        self._prepared_polygon = prep(geom)
        self._manifest = load_manifest("tamil_nadu_boundary")

    @property
    def name(self) -> str:
        return self._manifest.get("provider") if self._manifest else "OpenStreetMap Contributors / Nominatim"

    @property
    def dataset_name(self) -> str:
        return self._manifest.get("dataset") if self._manifest else "Tamil Nadu State Boundary (38 Districts)"

    def get_metadata(self) -> SourceMetadata:
        manifest = self._manifest or {}
        retrieved_at = (
            datetime.fromisoformat(manifest["retrieved_at"])
            if manifest.get("retrieved_at")
            else datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
        )
        return SourceMetadata(
            provider=manifest.get("provider", "OpenStreetMap Contributors / Nominatim"),
            dataset=manifest.get("dataset", "Tamil Nadu State Boundary (38 Districts)"),
            source_url=manifest.get("source_url", "https://www.openstreetmap.org/relation/96905"),
            source_id=str(manifest.get("osm_relation_id", "IN-TN-ADM4")),
            retrieved_at=retrieved_at,
            license=manifest.get("license", "Open Database License (ODbL) / OGD"),
            status=ProviderStatus.READY,
            notes=f"Authentic administrative boundary ({manifest.get('geometry_type', 'MultiPolygon')}, {manifest.get('vertex_count', 24812)} vertices). SHA256: {manifest.get('sha256', '')[:12]}...",
        )

    def is_inside_tamil_nadu(self, lat: float, lon: float) -> bool:
        """Check if coordinates lie within Tamil Nadu state boundary (allowing a small coastal margin for ports/harbors)."""
        if self._prepared_polygon is None or self._geometry is None:
            return False
        point = Point(lon, lat)
        if self._prepared_polygon.contains(point) or self._geometry.touches(point):
            return True
        return bool(self._geometry.distance(point) <= 0.005)

    def validate_coordinates(self, lat: float, lon: float) -> None:
        """Validate coordinates, raising HTTP 422 if outside Tamil Nadu."""
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid WGS84 coordinates: ({lat}, {lon}).",
            )
        if not self.is_inside_tamil_nadu(lat, lon):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Coordinates ({lat:.4f}, {lon:.4f}) are outside Tamil Nadu. "
                    "DisasterPulse TN serves Tamil Nadu exclusively."
                ),
            )

    def get_boundary_geojson(self) -> dict[str, Any]:
        return self._raw_geojson

    def generate_state_grid(self, step_deg: float = 0.25) -> list[dict[str, Any]]:
        """Generate a regular spatial grid clipped strictly to Tamil Nadu."""
        if self._geometry is None:
            return []

        min_lon, min_lat, max_lon, max_lat = self._geometry.bounds
        cells = []
        cell_index = 0

        lat = min_lat + (step_deg / 2)
        while lat <= max_lat:
            lon = min_lon + (step_deg / 2)
            while lon <= max_lon:
                if self.is_inside_tamil_nadu(lat, lon):
                    cell_index += 1
                    half = step_deg / 2
                    cell_poly = [
                        [round(lon - half, 4), round(lat - half, 4)],
                        [round(lon + half, 4), round(lat - half, 4)],
                        [round(lon + half, 4), round(lat + half, 4)],
                        [round(lon - half, 4), round(lat + half, 4)],
                        [round(lon - half, 4), round(lat - half, 4)],
                    ]
                    cells.append({
                        "cell_id": f"tn_grid_{cell_index:03d}",
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "bounds": {
                            "min_lat": round(lat - half, 4),
                            "max_lat": round(lat + half, 4),
                            "min_lon": round(lon - half, 4),
                            "max_lon": round(lon + half, 4),
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [cell_poly],
                        },
                    })
                lon += step_deg
            lat += step_deg

        return cells


boundary_provider = BoundaryProvider()
