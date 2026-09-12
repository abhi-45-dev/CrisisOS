from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata


class ProviderManager:
    """Central registry for all active data providers in DisasterPulse TN."""

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}

    def register(self, key: str, provider: BaseProvider) -> None:
        self._providers[key] = provider

    def get(self, key: str) -> BaseProvider | None:
        return self._providers.get(key)

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key, provider in self._providers.items():
            try:
                meta = provider.get_metadata()
                result[key] = meta.model_dump(mode="json")
            except Exception as exc:
                result[key] = {
                    "provider": getattr(provider, "name", key),
                    "dataset": getattr(provider, "dataset_name", "unknown"),
                    "status": ProviderStatus.DEGRADED.value,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "notes": f"Provider inspection failed: {exc}",
                }
        return result

    def get_overall_health(self) -> dict[str, Any]:
        all_meta = self.get_all_metadata()
        has_unavailable = any(m.get("status") == ProviderStatus.UNAVAILABLE.value for m in all_meta.values())
        has_degraded = any(m.get("status") == ProviderStatus.DEGRADED.value for m in all_meta.values())
        
        status = "healthy"
        if has_unavailable:
            status = "degraded"
        elif has_degraded:
            status = "operational_with_warnings"

        return {
            "status": status,
            "system": "DisasterPulse TN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "providers_count": len(self._providers),
            "providers": all_meta,
        }


provider_manager = ProviderManager()

# Register core providers
from app.providers.boundary_provider import boundary_provider
from app.providers.hydrology_provider import hydrology_provider
from app.providers.infrastructure_provider import evacuation_facility_provider, hospital_provider
from app.providers.weather_provider import weather_provider
from app.routing.provider import osm_routing_provider

from app.providers.flood_model_provider import flood_model_provider

provider_manager.register("boundary", boundary_provider)
provider_manager.register("hospitals", hospital_provider)
provider_manager.register("shelters", evacuation_facility_provider)
provider_manager.register("roads", osm_routing_provider)
provider_manager.register("weather", weather_provider)
provider_manager.register("hydrology", hydrology_provider)
provider_manager.register("flood_model", flood_model_provider)
