from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import HorizonPredictions, PredictRequest, PredictResponse
from dhompo.config import load_serving_config
from dhompo.serving.file_predictor import FilePredictor

router = APIRouter()

_predictor: Any | None = None
_SERVING_CFG = load_serving_config()


def _configured_backend() -> str:
    return os.getenv("PREDICTOR_BACKEND", "file").strip().lower()


def _mlflow_tracking_uri() -> str:
    return os.getenv(
        "MLFLOW_TRACKING_URI", _SERVING_CFG.get("mlflow_uri", "http://localhost:5000")
    )


def _model_alias() -> str:
    return os.getenv("MODEL_ALIAS", _SERVING_CFG.get("model_alias", "production"))


def get_predictor():
    global _predictor
    if _predictor is None:
        backend = _configured_backend()
        if backend == "mlflow":
            try:
                from dhompo.serving.predictor import HistoricalPredictor
            except ImportError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "MLflow backend requested but MLflow dependencies are not "
                        "available in this environment."
                    ),
                ) from exc

            _predictor = HistoricalPredictor(
                mlflow_tracking_uri=_mlflow_tracking_uri(),
                alias=_model_alias(),
            )
        elif backend == "file":
            _predictor = FilePredictor()
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported predictor backend: {backend}",
            )
    return _predictor


def _history_to_dataframe(request: PredictRequest) -> pd.DataFrame:
    """Convert validated PredictRequest.history → DataFrame with DatetimeIndex."""
    records = []
    timestamps = []
    for row in request.history:
        records.append(row.readings)
        timestamps.append(row.timestamp)

    df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="Datetime"))
    df = df.asfreq("30min")
    return df


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict water levels at Dhompo for +1 to +5 hours.

    Requires at least 24 rows (12 hours) of 30-minute historical sensor
    data from all upstream stations and Dhompo.
    """
    try:
        predictor = get_predictor()
        df = _history_to_dataframe(request)
        result = predictor.predict_from_history(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    preds = result.predictions
    last_ts = request.history[-1].timestamp
    backend = getattr(predictor, "backend_name", _configured_backend())
    models = predictor.model_mapping() if hasattr(predictor, "model_mapping") else {}
    return PredictResponse(
        predictions=HorizonPredictions(
            h1=preds["h1"],
            h2=preds["h2"],
            h3=preds["h3"],
            h4=preds["h4"],
            h5=preds["h5"],
        ),
        backend=backend,
        models=models,
        timestamp=last_ts,
        prediction_time=datetime.now(timezone.utc),
    )
