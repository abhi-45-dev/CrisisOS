def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "crisis_flood" in payload["crises_loaded"]
    assert "crisis_cyclone" in payload["crises_loaded"]
    assert "crisis_earthquake" in payload["crises_loaded"]


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "CrisisOS"


def test_docs(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert "/map/flood" in spec.json()["paths"]
