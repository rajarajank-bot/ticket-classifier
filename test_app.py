from fastapi.testclient import TestClient
from app import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_predict_billing():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "I was charged twice"})
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Billing"
        assert data["priority"] == "High"
        assert 0.0 <= data["confidence"] <= 1.0

def test_predict_technical():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "The mobile app crashes every time"})
        assert response.status_code == 200
        assert response.json()["category"] == "Technical"

def test_predict_empty_text_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

def test_predict_batch():
    with TestClient(app) as client:
        payload = {"tickets": [{"text": "I was charged twice"}, {"text": "I forgot my password"}]}
        response = client.post("/predict-batch", json=payload)
        assert response.status_code == 200
        assert len(response.json()) == 2