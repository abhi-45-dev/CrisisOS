"""DisasterPulse TN API v1."""

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.infrastructure import router as infrastructure_router
from app.api.v1.routes import router as routes_router
from app.api.v1.hazard import router as hazard_router
from app.api.v1.survivor import router as survivor_router
from app.api.v1.scenarios import router as scenarios_router
from app.api.v1.incidents import router as incidents_router

v1_router = APIRouter()
v1_router.include_router(health_router, tags=["health & provenance"])
v1_router.include_router(infrastructure_router, tags=["infrastructure"])
v1_router.include_router(routes_router, tags=["routing"])
v1_router.include_router(hazard_router, tags=["hazard"])
v1_router.include_router(survivor_router, tags=["survivor"])
v1_router.include_router(scenarios_router, tags=["scenarios"])
v1_router.include_router(incidents_router, tags=["incidents"])
