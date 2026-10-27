import pytest
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["total_products"] > 0


def test_recommend_endpoint(client):
    response = client.get("/recommend", params={"product_title": "Ice Maker", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["recommendations"]) == 3
    assert "title" in data["recommendations"][0]


def test_recommend_query_endpoint(client):
    response = client.get("/recommend/query", params={"query": "microwave turntable glass plate", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "semantic_search"
    assert data["count"] == 3


def test_search_endpoint(client):
    response = client.get("/search", params={"query": "LG", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data["products"]) <= 5


def test_product_not_found(client):
    response = client.get("/product/NON_EXISTENT_ASIN_99999")
    assert response.status_code == 404
