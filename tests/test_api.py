import io
import json
import logging

import joblib
import pytest
from fastapi.testclient import TestClient

from heartrisk.api import JsonFormatter, create_app, logger


@pytest.fixture()
def client(tmp_path, fitted_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_name": "test_lr", "model_version": "0.0.1", "metrics": {}})
    )
    with TestClient(create_app(path)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_name": "test_lr"}


def test_predict_returns_prediction_and_confidence(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart_disease", "no_heart_disease")
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["model_version"] == "0.0.1"
    assert "x-request-id" in r.headers


def test_predict_accepts_missing_optional_fields(client, sample_record):
    rec = {k: v for k, v in sample_record.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=rec).status_code == 200


@pytest.mark.parametrize("patch", [{"age": 5}, {"sex": 3}, {"chol": "high"}, {"cp": 0}])
def test_predict_rejects_invalid_input(client, sample_record, patch):
    r = client.post("/predict", json=sample_record | patch)
    assert r.status_code == 422


def test_predict_rejects_missing_required(client, sample_record):
    rec = dict(sample_record)
    rec.pop("age")
    assert client.post("/predict", json=rec).status_code == 422


def test_metrics_exposed(client, sample_record):
    client.post("/predict", json=sample_record)
    text = client.get("/metrics").text
    assert "heartrisk_http_requests_total" in text
    assert "heartrisk_predictions_total" in text
    assert 'path="/predict"' in text


def test_request_is_logged_as_json(client):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        client.get("/health")
    finally:
        logger.removeHandler(handler)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    req = [r for r in records if r["msg"] == "request" and r["path"] == "/health"]
    assert req and req[-1]["status"] == 200 and "latency_ms" in req[-1]
