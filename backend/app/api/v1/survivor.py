from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.survivor_service import survivor_service

router = APIRouter(prefix="/survivor")


class SurvivorRecommendRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Survivor GPS latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Survivor GPS longitude")
    preferred_type: Literal["all", "hospital", "shelter"] = Field(
        default="all",
        description="Facility category filter: 'all', 'hospital', or 'shelter'",
    )
    language: Literal["en", "ta"] = Field(
        default="en",
        description="Output language: 'en' for English, 'ta' for Tamil",
    )
    radius_km: float = Field(
        default=35.0,
        ge=1.0,
        le=150.0,
        description="Search radius in kilometers",
    )


@router.post("/recommend")
def recommend_evacuation(body: SurvivorRecommendRequest):
    """
    Recommend the safest reachable evacuation facility and route for a survivor GPS position.
    
    Evaluates:
    1. Local flood hazard at survivor position.
    2. Real public hospitals & evacuation facilities within radius.
    3. OSM road network routing (FASTEST vs. SAFEST with road exposure penalties).
    4. Rejecting destinations in submerged zones or impassable corridors.
    5. Plain-language explainable recommendation with Tamil localization support.
    """
    try:
        result = survivor_service.evaluate_evacuation(
            survivor_lat=body.latitude,
            survivor_lon=body.longitude,
            preferred_type=body.preferred_type,
            language=body.language,
            radius_km=body.radius_km,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate survivor recommendation: {str(e)}",
        )
