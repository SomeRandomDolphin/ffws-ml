"""Integration tests for the Dhompo Flood Prediction API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import make_history_payload

client = TestClient(app)


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_model_info(self):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "file"
        assert "mlflow" in data["available_backends"]
        assert "h1" in data["models"]
        assert "h5" in data["models"]
        assert data["required_history_rows"] == 24
        assert "Dhompo" in data["required_stations"]

    def test_model_info_mlflow_backend(self, monkeypatch):
        monkeypatch.setenv("PREDICTOR_BACKEND", "mlflow")
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "mlflow"
        assert data["models"]["h1"] == "models:/dhompo_h1@production"
        monkeypatch.delenv("PREDICTOR_BACKEND", raising=False)


class TestPredict:
    def test_predict_success(self, sample_history_payload):
        resp = client.post("/predict", json={"history": sample_history_payload})
        assert resp.status_code == 200
        data = resp.json()
        preds = data["predictions"]
        for key in ("h1", "h2", "h3", "h4", "h5"):
            assert key in preds
            assert isinstance(preds[key], float)
        assert data["backend"] == "file"
        assert "h1" in data["models"]
        assert "timestamp" in data
        assert "prediction_time" in data

    def test_predict_insufficient_history(self):
        payload = make_history_payload(n_rows=10)
        resp = client.post("/predict", json={"history": payload})
        assert resp.status_code == 422

    def test_predict_missing_stations(self):
        """Send rows with only Dhompo — missing upstream stations."""
        from datetime import datetime, timedelta

        rows = []
        start = datetime(2022, 11, 21, 0, 0)
        for i in range(24):
            rows.append({
                "timestamp": (start + timedelta(minutes=30 * i)).isoformat(),
                "readings": {"Dhompo": 1.0},
            })
        resp = client.post("/predict", json={"history": rows})
        assert resp.status_code == 422

    def test_predict_response_shape(self, sample_history_payload):
        resp = client.post("/predict", json={"history": sample_history_payload})
        assert resp.status_code == 200
        data = resp.json()
        # Verify all expected top-level keys
        assert set(data.keys()) == {
            "predictions", "backend", "models", "timestamp", "prediction_time",
        }
        # Verify predictions has exactly h1..h5
        assert set(data["predictions"].keys()) == {"h1", "h2", "h3", "h4", "h5"}

    def test_predict_empty_body(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_bad_timestamp_boundary(self):
        """Timestamps not on 30-min boundary should fail validation."""
        from datetime import datetime, timedelta

        rows = []
        start = datetime(2022, 11, 21, 0, 15)  # :15 — not a boundary
        for i in range(24):
            rows.append({
                "timestamp": (start + timedelta(minutes=30 * i)).isoformat(),
                "readings": {"Dhompo": 1.0},
            })
        resp = client.post("/predict", json={"history": rows})
        assert resp.status_code == 422
