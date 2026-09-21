"""Muat data, pilih window demo, dan jalankan prediksi model nyata.

Basis demo adalah dataset gold (2022 + 2023) bila tersedia, fallback ke
`data-clean.csv`. Datum elevasi (Dhompo sekitar 7-15 m). Window dipilih
otomatis: saat pertama Dhompo melewati persentil alert, dengan
`history_rows` baris 30 menit sebelumnya sebagai input prediksi.

Catatan scaler: model Lasso horizon h4/h5 butuh fitur terskala. `scaler.pkl`
tidak tersedia di folder model, sehingga dashboard memakai
`standard_scaler_global.pkl` yang ada. Ini murni untuk dashboard dan tidak
mengubah perilaku serving API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import pandas as pd

from dhompo.config import PROJECT_ROOT, load_yaml_config, resolve_path_from_config
from dhompo.data.loader import (
    ALL_STATIONS,
    TARGET_STATION,
    load_data,
    load_gold_data,
    split_gold_segments,
)
from dhompo.serving.file_predictor import FilePredictor

from dashboard.status import Thresholds, compute_thresholds

DASHBOARD_CONFIG = PROJECT_ROOT / "configs" / "dhompo" / "dashboard.yaml"
SCALER_GLOBAL = PROJECT_ROOT / "models" / "sklearn" / "standard_scaler_global.pkl"

logger = logging.getLogger(__name__)


@dataclass
class DemoData:
    df: pd.DataFrame
    window: pd.DataFrame
    last_ts: pd.Timestamp
    predictions: dict[str, float]
    thresholds: dict[str, Thresholds]
    config: dict
    serving_tier: str
    model_versions: dict[str, str]
    degraded: list[str]
    rainfall: pd.Series | None = None
    source: str = "2022_clean"


def _load_dashboard_config() -> dict:
    cfg = load_yaml_config(DASHBOARD_CONFIG)
    if not cfg:
        cfg = {}
    cfg.setdefault("data", {})
    cfg.setdefault("thresholds", {})
    cfg.setdefault("demo", {})
    cfg.setdefault("propagation", {})
    return cfg


def _resolve_data_path(cfg: dict) -> tuple[Path, bool]:
    """Kembalikan (path, is_gold). Gold dataset diprioritaskan kalau ada."""
    gold_value = cfg["data"].get("gold_path", "../../data/gold/hydro.csv")
    gold_path = resolve_path_from_config(DASHBOARD_CONFIG, gold_value)
    if gold_path and gold_path.exists():
        return gold_path, True
    value = cfg["data"].get("path", "../../data/data-clean.csv")
    return resolve_path_from_config(DASHBOARD_CONFIG, value) or (
        PROJECT_ROOT / "data" / "data-clean.csv"
    ), False


def _load_station_frame(cfg: dict) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """Muat kolom stasiun kanonik + curah hujan, pilih segmen window.

    Gold dataset punya dua segmen (2022 clean, 2023 generated) dipisah gap
    ~26 hari. Segmen dipilih dari ``demo.segment``; default segmen terakhir.
    Threshold tetap dihitung dari seluruh periode.
    """
    path, is_gold = _resolve_data_path(cfg)
    if not is_gold:
        df = load_data(path)
        cols = [c for c in ALL_STATIONS if c in df.columns]
        label = "2022_clean"
        return df[cols], None, label

    gold = load_gold_data(path)
    rain_all = gold["rain_mm"]
    sta = [c for c in ALL_STATIONS if c in gold.columns]
    segments = split_gold_segments(gold.drop(columns=["rain_mm"]))
    if not segments:
        raise ValueError(f"gold dataset kosong: {path}")

    def _label(seg: pd.DataFrame) -> str:
        src = str(seg["source"].iloc[0])
        return src if src else ("2023_generated" if seg.index[0].year >= 2023 else "2022_clean")

    wanted = cfg["demo"].get("segment")
    if wanted:
        seg = next((s for s in segments if _label(s) == wanted), segments[-1])
        label = str(wanted)
    else:
        seg = segments[-1]
        label = _label(seg)

    df = seg[sta]
    rain = rain_all.reindex(seg.index)
    if rain.notna().sum() == 0:
        rain = None
    return df, rain, label


def _ensure_scaler(predictor: FilePredictor) -> None:
    if not (predictor._model_dir / "scaler.pkl").exists() and SCALER_GLOBAL.exists():
        predictor._scaler = joblib.load(SCALER_GLOBAL)


def _plausible(value: float, last: float, lo: float, hi: float) -> bool:
    if value != value:  # NaN
        return False
    if value < lo or value > hi:
        return False
    if abs(value - last) > (hi - lo) * 0.75:
        return False
    return True


def _select_window(
    df: pd.DataFrame,
    alert_value: float,
    history_rows: int,
    window_end: str | None,
) -> pd.DataFrame:
    if window_end:
        ts = pd.Timestamp(window_end)
        i = max(df.index.get_loc(ts, method="nearest"), history_rows)
    else:
        crosses = df.index[df[TARGET_STATION] >= alert_value]
        candidates = [
            idx for idx in crosses if df.index.get_loc(idx) >= history_rows
        ]
        i = df.index.get_loc(candidates[0])
    return df.iloc[i - history_rows : i + 1]


def _predict(window: pd.DataFrame, cfg: dict) -> tuple[dict[str, float], str, dict[str, str], list[str]]:
    lead_hours = [int(h) for h in cfg["data"].get("lead_hours", [1, 2, 3, 4, 5]) if int(h) > 0]
    predictor = FilePredictor()
    _ensure_scaler(predictor)
    degraded: list[str] = []
    try:
        result = predictor.predict_from_history(window)
    except Exception:
        # Prediksi gagal (model/scaler/data tak lengkap). Fallback persistence
        # tetap menghasilkan UI yang layak, tapi harus terlihat di UI (tier B).
        logger.warning("Prediksi model gagal; fallback persistence", exc_info=True)
        last = float(window[TARGET_STATION].dropna().iloc[-1])
        predictions = {f"h{h}": round(last, 4) for h in lead_hours}
        degraded.append("prediction_error:fallback_persistence")
        return predictions, "B", {}, degraded

    predictions = result.predictions
    last = float(window[TARGET_STATION].dropna().iloc[-1])
    lo = float(window[TARGET_STATION].min()) - 2.0
    hi = float(window[TARGET_STATION].max()) + 2.0
    for h in lead_hours:
        if not _plausible(predictions[f"h{h}"], last, lo, hi):
            predictions[f"h{h}"] = round(last, 4)
            degraded.append(f"h{h}:plausibility_fallback")

    serving_tier = "B" if degraded else "A"
    return predictions, serving_tier, predictor.model_mapping(), degraded


def load_demo_data() -> DemoData:
    cfg = _load_dashboard_config()
    df, rainfall, source = _load_station_frame(cfg)
    cols = [c for c in ALL_STATIONS if c in df.columns]
    df = df[cols]

    th_cfg = cfg["thresholds"]
    overrides = th_cfg.get("station_overrides") or {}
    thresholds = compute_thresholds(
        df,
        alert_quantile=float(th_cfg.get("alert_quantile", 0.90)),
        danger_quantile=float(th_cfg.get("danger_quantile", 0.99)),
        overrides=overrides,
    )

    history_rows = int(cfg["data"].get("history_rows", 24))
    alert_value = thresholds[TARGET_STATION].alert
    window = _select_window(
        df, alert_value, history_rows, cfg["demo"].get("window_end")
    )

    predictions, serving_tier, model_versions, degraded = _predict(window, cfg)

    return DemoData(
        df=df,
        window=window,
        last_ts=window.index[-1],
        predictions=predictions,
        thresholds=thresholds,
        config=cfg,
        serving_tier=serving_tier,
        model_versions=model_versions,
        degraded=degraded,
        rainfall=rainfall,
        source=source,
    )


_demo_cache: DemoData | None = None


def get_demo(reload: bool = False) -> DemoData:
    global _demo_cache
    if _demo_cache is None or reload:
        _demo_cache = load_demo_data()
    return _demo_cache