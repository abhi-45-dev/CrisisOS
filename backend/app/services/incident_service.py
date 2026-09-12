from __future__ import annotations

import html
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.geo import haversine_km
from app.providers.boundary_provider import boundary_provider


class IncidentService:
    """Citizen incident reporting and verification manager for Tamil Nadu."""

    DATA_FILE = Path(__file__).resolve().parent.parent.parent / "cache" / "incidents" / "incidents.json"

    def __init__(self):
        self.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._incidents: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.DATA_FILE.exists():
            try:
                with self.DATA_FILE.open("r", encoding="utf-8") as f:
                    self._incidents = json.load(f)
            except Exception:
                self._incidents = []
        else:
            self._incidents = []

    def _save(self):
        with self.DATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(self._incidents, f, indent=2)

    def report_incident(
        self,
        latitude: float,
        longitude: float,
        incident_type: str,
        description: str,
        severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM",
        reporter_name: str | None = None,
    ) -> dict[str, Any]:
        """Report a new citizen incident with strict validation and XSS sanitization."""
        # 1. Geographic validation: must be within Tamil Nadu boundary
        boundary_provider.validate_coordinates(latitude, longitude)

        # 2. XSS sanitization of all string inputs
        clean_desc = html.escape(description.strip())
        clean_type = html.escape(incident_type.strip().lower())
        clean_reporter = html.escape(reporter_name.strip()) if reporter_name else "Anonymous Citizen"

        incident_id = f"inc_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "id": incident_id,
            "latitude": round(latitude, 5),
            "longitude": round(longitude, 5),
            "incident_type": clean_type,
            "description": clean_desc,
            "severity": severity,
            "status": "UNVERIFIED",  # Ground-truth: strictly unverified until confirmed
            "verification_source": None,
            "reported_by": clean_reporter,
            "created_at": now,
        }

        self._incidents.insert(0, record)
        self._save()
        return record

    def list_incidents(
        self,
        status_filter: str | None = None,
        severity_filter: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent incident reports."""
        results = self._incidents
        if status_filter:
            results = [i for i in results if i.get("status") == status_filter]
        if severity_filter:
            results = [i for i in results if i.get("severity") == severity_filter]
        return results[:limit]

    def get_nearby_incidents(
        self,
        lat: float,
        lon: float,
        radius_km: float = 25.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find incidents within radius km of a coordinate."""
        nearby = []
        for inc in self._incidents:
            dist = haversine_km(lat, lon, inc["latitude"], inc["longitude"])
            if dist <= radius_km:
                nearby.append({**inc, "distance_km": round(dist, 2)})
        nearby.sort(key=lambda x: x["distance_km"])
        return nearby[:limit]


incident_service = IncidentService()
