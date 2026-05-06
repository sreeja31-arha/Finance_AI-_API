import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_ok(client):
    response = client.get("/health")
    data = json.loads(response.data)
    assert data["status"] == "ok"


def test_health_has_model_loaded_key(client):
    response = client.get("/health")
    data = json.loads(response.data)
    assert "model_loaded" in data


def test_predict_returns_200(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    assert response.status_code == 200


def test_predict_returns_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert "category" in data


def test_predict_returns_confidence(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert "confidence" in data


def test_predict_confidence_between_0_and_1(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_returns_description(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert "description" in data


def test_predict_food_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Swiggy biryani order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Food"


def test_predict_transport_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Ola cab booking"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Transport"


def test_predict_entertainment_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Netflix monthly subscription"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Entertainment"


def test_predict_groceries_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "BigBasket grocery order"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Groceries"


def test_predict_bills_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Electricity bill payment"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Bills"


def test_predict_healthcare_category(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "Apollo pharmacy medicines"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["category"] == "Healthcare"


def test_empty_description_returns_400(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": ""}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_missing_description_key_returns_400(client):
    response = client.post(
        "/predict",
        data=json.dumps({"amount": 500}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_too_short_description_returns_400(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "ab"}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_too_long_description_returns_400(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": "a" * 301}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_no_json_body_returns_400(client):
    response = client.post(
        "/predict",
        data="not json",
        content_type="text/plain"
    )
    assert response.status_code == 400


def test_error_response_has_error_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"description": ""}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert "error" in data


def test_categories_returns_200(client):
    response = client.get("/categories")
    assert response.status_code == 200


def test_categories_returns_list(client):
    response = client.get("/categories")
    data = json.loads(response.data)
    assert isinstance(data["categories"], list)