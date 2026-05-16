from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Support running without package install
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.predictor_state import load_predictor_state
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from dhompo.config import load_serving_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor_state(app)
    yield


app = FastAPI(
    title="Dhompo Flood Prediction API",
    description=(
        "Multi-horizon (1-5 jam) water level prediction for Sungai Dhompo "
        "early-warning flood system."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "Dhompo Flood Prediction API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    serving_cfg = load_serving_config()
    api_cfg = serving_cfg.get("api", {})
    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=int(api_cfg.get("port", 8000)),
        reload=False,
    )
