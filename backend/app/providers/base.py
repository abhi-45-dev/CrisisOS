from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


class ProviderStatus(str, Enum):
    LIVE = "LIVE"
    READY = "READY"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class SourceMetadata(BaseModel):
    provider: str
    dataset: str
    source_url: str | None = None
    source_id: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observed_at: datetime | None = None
    license: str | None = None
    freshness_seconds: float | None = None
    status: ProviderStatus = ProviderStatus.READY
    notes: str | None = None

    model_config = {"populate_by_name": True}


T = TypeVar("T")


class ProviderResult(BaseModel, Generic[T]):
    data: T
    metadata: SourceMetadata


class BaseProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider human-readable identifier."""
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Dataset or feed identifier."""
        pass

    @abstractmethod
    def get_metadata(self) -> SourceMetadata:
        """Return current status and provenance metadata."""
        pass
