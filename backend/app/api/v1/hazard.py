from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.hazard_service import hazard_service

router = APIRouter(prefix="/hazard")


class CoordinatePredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    rainfall_override_mm: float | None = Field(default=None, ge=0.0, le=1000.0)


@router.get("/grid")
def get_hazard_grid(
    rainfall_override_mm: float | None = Query(default=None),
    force_refresh: bool = Query(default=False),
):
    """Retrieve statewide GeoJSON flood hazard polygon surface clipped to Tamil Nadu."""
    return hazard_service.get_statewide_hazard_grid(
        rainfall_override_mm=rainfall_override_mm,
        force_refresh=force_refresh,
    )


@router.post("/predict")
def predict_hazard_at_coordinate(body: CoordinatePredictRequest):
    """Predict flood occurrence probability and local explanations for an arbitrary coordinate inside Tamil Nadu."""
    return hazard_service.predict_coordinate(
        lat=body.latitude,
        lon=body.longitude,
        rainfall_override_mm=body.rainfall_override_mm,
    )
