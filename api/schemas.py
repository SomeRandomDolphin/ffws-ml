from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from dhompo.data.loader import TARGET_STATION, UPSTREAM_STATIONS


class HistoryRow(BaseModel):
    """Single timestamped observation across all stations."""

    timestamp: datetime = Field(
        ...,
        description="ISO 8601 datetime on a 30-minute boundary.",
        examples=["2022-11-21T14:00:00"],
    )
    readings: dict[str, float] = Field(
        ...,
        description="Station name → water level (m).",
        examples=[{"Dhompo": 1.23, "Bd. Suwoto": 0.45}],
    )

    @field_validator("timestamp")
    @classmethod
    def timestamp_on_30min_boundary(cls, v: datetime) -> datetime:
        if v.minute not in (0, 30) or v.second != 0:
            raise ValueError(
                "timestamp must be on a 30-minute boundary (e.g. :00 or :30)"
            )
        return v


# Required station columns for feature engineering
REQUIRED_STATIONS: list[str] = UPSTREAM_STATIONS + [TARGET_STATION]

MIN_HISTORY_ROWS = 24


class PredictRequest(BaseModel):
    """POST /predict request body — requires ≥24 rows of 30-min history."""

    history: list[HistoryRow] = Field(
        ...,
        description=(
            f"Time series of station readings, minimum {MIN_HISTORY_ROWS} rows, "
            "30-minute intervals, sorted ascending by timestamp."
        ),
    )

    @model_validator(mode="after")
    def validate_history(self) -> PredictRequest:
        rows = self.history
        # Minimum length
        if len(rows) < MIN_HISTORY_ROWS:
            raise ValueError(
                f"history must contain at least {MIN_HISTORY_ROWS} rows, "
                f"got {len(rows)}."
            )

        # Check ascending order and 30-min gaps
        for i in range(1, len(rows)):
            delta = (rows[i].timestamp - rows[i - 1].timestamp).total_seconds()
            if delta != 1800:
                raise ValueError(
                    f"Rows {i - 1}→{i}: expected 30-minute gap (1800s), "
                    f"got {delta}s between {rows[i - 1].timestamp} and "
                    f"{rows[i].timestamp}."
                )

        # Check required stations present in every row
        for i, row in enumerate(rows):
            missing = set(REQUIRED_STATIONS) - set(row.readings.keys())
            if missing:
                raise ValueError(
                    f"Row {i} (timestamp={row.timestamp}): missing stations "
                    f"{sorted(missing)}."
                )

        return self


class HorizonPredictions(BaseModel):
    h1: float = Field(..., description="Water level in +1 hour (m)")
    h2: float = Field(..., description="Water level in +2 hours (m)")
    h3: float = Field(..., description="Water level in +3 hours (m)")
    h4: float = Field(..., description="Water level in +4 hours (m)")
    h5: float = Field(..., description="Water level in +5 hours (m)")


class PredictResponse(BaseModel):
    """POST /predict response body."""

    predictions: HorizonPredictions
    backend: str = Field(..., description="Predictor backend: 'file' or 'mlflow'")
    models: dict[str, str] = Field(
        ...,
        description="Model identifier used per horizon (filename or MLflow registry URI).",
        examples=[{"h1": "xgboost_h1.pkl"}],
    )
    timestamp: datetime = Field(..., description="Last observation timestamp")
    prediction_time: datetime = Field(..., description="Server time of prediction")
    serving_tier: Literal["A", "B"] = Field(
        "A",
        description=(
            "Which tier produced the served prediction. 'A' = adaptive 14-station "
            "model (production); 'B' = autoregressive Dhompo-only fallback, "
            "activated when all three telemetry stations carry bad flags."
        ),
    )
    degradation: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-horizon degradation reason when the horizon's primary upstream "
            "station carries a bad quality flag. Empty when all primaries OK. "
            "Format: {'h4': 'PRIMARY_STATION_FLATLINE:Purwodadi'}."
        ),
        examples=[{"h4": "PRIMARY_STATION_FLATLINE:Purwodadi"}],
    )
    shadow_predictions: Optional[HorizonPredictions] = Field(
        None,
        description=(
            "Tier-B prediction logged in shadow when Tier-A is serving. None when "
            "Tier-B is itself serving (no shadow Tier-A in that case)."
        ),
    )
    quality_flags: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Latest per-station quality flag at the request's most recent "
            "timestep. Values: OK, STUCK, FLATLINE, OUT_OF_RANGE, MISSING, STALE."
        ),
    )
