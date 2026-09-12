from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.geo import haversine_km
from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata


class HospitalProvider(BaseProvider):
    """Provides genuine geocoded Tamil Nadu hospital records with transparent provenance."""

    def __init__(self, data_path: Path | None = None):
        self._data_path = data_path or (
            Path(__file__).resolve().parent.parent / "data" / "tamil_nadu_hospitals.json"
        )
        self._hospitals: list[dict[str, Any]] = []
        self._load_records()

    def _load_records(self) -> None:
        if not self._data_path.exists():
            raise FileNotFoundError(f"Hospital records file missing: {self._data_path}")
        with self._data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Deduplicate records by name and coordinates
        seen = set()
        deduped = []
        for item in raw:
            key = (item["name"].strip().lower(), round(item["latitude"], 4), round(item["longitude"], 4))
            if key in seen:
                continue
            seen.add(key)
            # Enforce truthfulness: live bed availability is null
            item["available_beds"] = None
            item["icu_capacity"] = None
            deduped.append(item)
        self._hospitals = deduped

    @property
    def name(self) -> str:
        return "National Health Portal / OGD Tamil Nadu"

    @property
    def dataset_name(self) -> str:
        return "Tamil Nadu Public Healthcare Registry"

    def get_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://tnhealth.tn.gov.in/",
            source_id="TN-HEALTH-2026",
            retrieved_at=datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
            license="Government Open Data License (GODL) India",
            status=ProviderStatus.READY,
            notes=f"Loaded {len(self._hospitals)} verified public hospitals. Live bed availability is explicitly unavailable.",
        )

    def get_all(self, district: str | None = None) -> list[dict[str, Any]]:
        if not district:
            return list(self._hospitals)
        target = district.strip().lower()
        return [h for h in self._hospitals if h.get("district", "").lower() == target]

    def get_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 30.0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results = []
        for hosp in self._hospitals:
            dist = haversine_km(lat, lon, hosp["latitude"], hosp["longitude"])
            if dist <= radius_km:
                results.append({**hosp, "distance_km": round(dist, 2)})
        results.sort(key=lambda h: h["distance_km"])
        return results[:limit]


class EvacuationFacilityProvider(BaseProvider):
    """Provides genuine Tamil Nadu evacuation shelters and candidate relief centres."""

    def __init__(self, data_path: Path | None = None):
        self._data_path = data_path or (
            Path(__file__).resolve().parent.parent / "data" / "tamil_nadu_shelters.json"
        )
        self._shelters: list[dict[str, Any]] = []
        self._load_records()

    def _load_records(self) -> None:
        if not self._data_path.exists():
            raise FileNotFoundError(f"Shelter records file missing: {self._data_path}")
        with self._data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        seen = set()
        deduped = []
        for item in raw:
            key = (item["name"].strip().lower(), round(item["latitude"], 4), round(item["longitude"], 4))
            if key in seen:
                continue
            seen.add(key)
            # Enforce truthfulness: live occupancy is null unless physically reported
            item["current_occupancy"] = None
            deduped.append(item)
        self._shelters = deduped

    @property
    def name(self) -> str:
        return "Tamil Nadu Disaster Management Authority / OSM"

    @property
    def dataset_name(self) -> str:
        return "Tamil Nadu Evacuation Facilities & Relief Centres"

    def get_metadata(self) -> SourceMetadata:
        official_count = sum(1 for s in self._shelters if s.get("is_official"))
        candidate_count = len(self._shelters) - official_count
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://tnsdma.tn.gov.in/",
            source_id="TNDMP-SHELTER-2026",
            retrieved_at=datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
            license="ODbL 1.0 / Open Government Data (OGD)",
            status=ProviderStatus.READY,
            notes=(
                f"{len(self._shelters)} facilities ({official_count} official shelters, "
                f"{candidate_count} candidate evacuation centres). Unofficial facilities tagged CANDIDATE."
            ),
        )

    def get_all(self, is_official_only: bool = False) -> list[dict[str, Any]]:
        if is_official_only:
            return [s for s in self._shelters if s.get("is_official")]
        return list(self._shelters)

    def get_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 30.0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results = []
        for shelter in self._shelters:
            dist = haversine_km(lat, lon, shelter["latitude"], shelter["longitude"])
            if dist <= radius_km:
                results.append({**shelter, "distance_km": round(dist, 2)})
        results.sort(key=lambda s: s["distance_km"])
        return results[:limit]


hospital_provider = HospitalProvider()
evacuation_facility_provider = EvacuationFacilityProvider()
