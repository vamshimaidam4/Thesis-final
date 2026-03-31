import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    # If frontend is built, serves HTML; otherwise returns JSON
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        data = response.json()
        assert "message" in data
    else:
        assert "html" in content_type.lower() or len(response.text) > 0


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "device" in data


def test_seed_reviews_list():
    response = client.get("/api/seed/reviews")
    assert response.status_code == 200
    data = response.json()
    assert "reviews" in data
    assert data["count"] == 15


def test_seed_review_by_id():
    response = client.get("/api/seed/reviews/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1


def test_seed_review_not_found():
    response = client.get("/api/seed/reviews/9999")
    assert response.status_code == 404


def test_seed_stats():
    response = client.get("/api/seed/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 15
    assert "sentiments" in data


def test_aws_status():
    response = client.get("/api/aws/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_document_analysis_validation():
    response = client.post("/api/analysis/document", json={"text": ""})
    assert response.status_code == 422


def test_batch_analysis_validation():
    response = client.post("/api/analysis/batch", json={"reviews": []})
    assert response.status_code == 422
