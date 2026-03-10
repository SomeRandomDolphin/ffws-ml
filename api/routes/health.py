import os
from datetime import datetime, timezone

from fastapi import APIRouter

from dhompo.config import load_serving_config
from dhompo.data.loader import UPSTREAM_STATIONS, TARGET_STATION
from dhompo.serving.file_predictor import BEST_MODEL_FILES, SCALER_FILENAME

router = APIRouter()
_SERVING_CFG = load_serving_config()


@router.get("/health", tags=["health"])
async def health_check() -> dict:
    """Return service liveness status."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/model-info", tags=["health"])
async def model_info() -> dict:
    """Return info about loaded models and required input shape."""
    backend = os.getenv("PREDICTOR_BACKEND", "file").strip().lower()
    model_alias = os.getenv("MODEL_ALIAS", _SERVING_CFG.get("model_alias", "production"))
    if backend == "mlflow":
        model_refs = {f"h{h}": f"models:/dhompo_h{h}@{model_alias}" for h in range(1, 6)}
    else:
        model_refs = {f"h{h}": fname for h, fname in BEST_MODEL_FILES.items()}
    return {
        "backend": backend,
        "available_backends": ["file", "mlflow"],
        "models": model_refs,
        "scaler": SCALER_FILENAME,
        "required_history_rows": 24,
        "required_stations": UPSTREAM_STATIONS + [TARGET_STATION],
    }
