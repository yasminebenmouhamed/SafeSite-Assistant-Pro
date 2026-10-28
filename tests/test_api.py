"""API smoke tests that don't download models or call external providers."""
from fastapi.testclient import TestClient
from api import app


def test_health():
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_invalid_request():
    response = TestClient(app).post('/ask', json={'question': '', 'k': 4})
    assert response.status_code == 422
