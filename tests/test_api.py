import pytest
from fastapi.testclient import TestClient

from src.api import app, get_model
from src.train import build_pipeline


@pytest.fixture
def client(cfg, tiny_df):
    pipe = build_pipeline(cfg)
    pipe.fit(tiny_df["text"], tiny_df["label"])
    app.dependency_overrides[get_model] = lambda: pipe
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_expected_label(client):
    r = client.post(
        "/predict",
        json={"clauses": ["either party may terminate this agreement upon thirty days written notice"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["label"] == "termination"
    assert len(body["results"][0]["top_k"]) == 3


def test_top_k_is_respected(client):
    r = client.post("/predict", json={"clauses": ["governed by the laws of new york"], "top_k": 2})
    assert len(r.json()["results"][0]["top_k"]) == 2


def test_empty_clause_list_rejected(client):
    assert client.post("/predict", json={"clauses": []}).status_code == 422


def test_blank_clause_rejected(client):
    assert client.post("/predict", json={"clauses": ["   "]}).status_code == 422