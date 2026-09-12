from __future__ import annotations

"""DisasterPulse TN API Root Router.

Directs incoming requests to API v1 endpoints and provides root metadata.
No synthetic prototype entities or toy simulations are imported here.
"""

from fastapi import APIRouter
from app.config import settings
from app.api.v1 import v1_router
from app.providers.manager import provider_manager

router = APIRouter()

# Mount API v1
router.include_router(v1_router, prefix="/api/v1")


@router.get("/", tags=["meta"])
def root():
    return {
        "name": settings.APP_NAME,
        "status": "online",
        "version": settings.APP_VERSION,
        "region": "Tamil Nadu, India",
        "human_in_the_loop": True,
        "docs": "/docs",
        "api_v1": "/api/v1",
    }


@router.get("/health", tags=["meta"])
def health():
    return provider_manager.get_overall_health()


@router.get("/data-sources", tags=["meta"])
def data_sources():
    return provider_manager.get_all_metadata()


@router.get("/evidence", tags=["meta"])
def evidence():
    from app.api.v1.health import get_system_evidence
    return get_system_evidence()
