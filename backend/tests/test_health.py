def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in ("healthy", "degraded", "operational_with_warnings")
    assert payload["system"] == "DisasterPulse TN"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DisasterPulse TN"
    assert data["region"] == "Tamil Nadu, India"


def test_api_v1_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["system"] == "DisasterPulse TN"


def test_api_v1_data_sources(client):
    response = client.get("/api/v1/data-sources")
    assert response.status_code == 200
    assert "sources" in response.json()


def test_api_v1_evidence(client):
    response = client.get("/api/v1/evidence")
    assert response.status_code == 200
    data = response.json()
    assert data["operational_handcrafted_data_count"] == 0
    assert data["geographic_scope"] == "Tamil Nadu, India"
