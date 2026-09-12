from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.services.incident_service import incident_service

router = APIRouter(prefix="/incidents")


class IncidentReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude of the incident")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude of the incident")
    incident_type: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Type of incident e.g. 'waterlogging', 'road_blocked', 'stranded_citizens'",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Detailed observation description",
    )
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        default="MEDIUM",
        description="Reported severity level",
    )
    reporter_name: str | None = Field(
        default=None,
        max_length=100,
        description="Optional reporter display name",
    )


@router.post("")
def report_incident(body: IncidentReportRequest):
    """
    Submit a citizen flood or hazard observation.
    
    All incoming reports default strictly to 'UNVERIFIED' status until corroborated
    by official emergency personnel. Coordinates must fall inside Tamil Nadu.
    """
    try:
        return incident_service.report_incident(
            latitude=body.latitude,
            longitude=body.longitude,
            incident_type=body.incident_type,
            description=body.description,
            severity=body.severity,
            reporter_name=body.reporter_name,
        )
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
            detail=f"Failed to submit incident report: {str(e)}",
        )


@router.get("")
def list_incidents(
    status: str | None = Query(default=None, description="Filter by status ('UNVERIFIED', etc.)"),
    severity: str | None = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List submitted citizen incident reports."""
    return incident_service.list_incidents(status_filter=status, severity_filter=severity, limit=limit)


@router.get("/nearby")
def get_nearby_incidents(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(default=25.0, ge=1.0, le=100.0),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Find incidents within radius km of a coordinate."""
    return incident_service.get_nearby_incidents(lat=lat, lon=lon, radius_km=radius_km, limit=limit)
