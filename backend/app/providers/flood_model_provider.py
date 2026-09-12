from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata
from ml.inference import flood_now_inference


class FloodModelProvider(BaseProvider):
    """Provides FloodNow TN machine-learning model status and probability predictions."""

    @property
    def name(self) -> str:
        return "FloodNow TN Machine Learning Engine"

    @property
    def dataset_name(self) -> str:
        return "India Flood Inventory v3 Supervised Occurrence Model"

    def get_metadata(self) -> SourceMetadata:
        status_data = flood_now_inference.get_status()
        is_ready = status_data.get("trained", False)
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://zenodo.org/records/16994648",
            source_id="FLOODNOW-TN-V1",
            retrieved_at=datetime.now(timezone.utc),
            license="CC-BY 4.0 / Zenodo",
            status=ProviderStatus.READY if is_ready else ProviderStatus.UNAVAILABLE,
            notes=(
                f"Trained on IFI v3 with TimeSeriesSplit tuning. "
                f"Selected Algorithm: {status_data.get('algorithm')}. "
                f"Model status: {status_data.get('status')}."
            ),
        )

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return flood_now_inference.predict(features)


flood_model_provider = FloodModelProvider()
