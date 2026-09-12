import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.crisis_service import crisis_service
from app.services.scenario_service import scenario_service


@pytest.fixture
def client():
    crisis_service.reset()
    scenario_service.history.clear()
    return TestClient(app)
