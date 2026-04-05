"""Flood Event Leave-One-Out CV + Peak-Weighted RMSE evaluation framework.

Mengatasi masalah test set (Feb-Mar 2023) yang tidak mengandung flood events,
sehingga metric overall tidak merefleksikan kualitas prediksi banjir.

Metode
------
1. Identifikasi major flood events (peak >= 13m, duration >= 10 timesteps).
2. Untuk tiap event:
   - Buffer exclusion: hapus window ±24h di sekitar peak dari training (cegah
     leakage via rolling features 12h).
   - Test window: evaluasi prediksi pada ±6h di sekitar peak.
   - Train model pada sisa data, predict pada test window.
3. Report per-event metrics + aggregate (mean/median across events).
4. Compute peak-weighted RMSE pada existing temporal test set (untuk
   single-number metric yang flood-sensitive).

Usage
-----
    python training/flood_event_cv.py
    python training/flood_event_cv.py --model CatBoost --horizon 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_features_from_segments,
    build_targets,
)
from dhompo.data.loader import RAINFALL_COLUMN, UPSTREAM_STATIONS, load_combined_data
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


# Event detection parameters
PEAK_THRESHOLD = 13.0       # minimum peak height to qualify as "major flood"
MIN_DURATION_STEPS = 10     # minimum consecutive steps >= 12m
FLOOD_THRESHOLD = 12.0      # water-level defining flood state
BUFFER_HOURS = 24           # train exclusion buffer around peak (prevents leakage)
TEST_WINDOW_HOURS = 6       # evaluation window ±hours around peak
STEP_MINUTES = 30


def detect_major_floods(df_target: pd.Series) -> pd.DataFrame:
    """Detect contiguous runs of y >= FLOOD_THRESHOLD that reach PEAK_THRESHOLD."""
    is_flood = df_target >= FLOOD_THRESHOLD
    groups = (is_flood != is_flood.shift()).cumsum()
    events = []
    for gid, subdf in df_target[is_flood].groupby(groups[is_flood]):
        peak = subdf.max()
        if peak < PEAK_THRESHOLD or len(subdf) < MIN_DURATION_STEPS:
            continue
        events.append({
            "event_id": f"flood_{subdf.idxmax().strftime('%Y%m%d_%H%M')}",
            "peak_time": subdf.idxmax(),
            "peak_value": peak,
            "start": subdf.index[0],
            "end": subdf.index[-1],
            "duration_steps": len(subdf),
        })
    return pd.DataFrame(events).sort_values("peak_time").reset_index(drop=True)


def peak_weighted_rmse(
    y_true: np.ndarray, y_pred: np.ndarray,
    flood_weight: float = 10.0,
    elevated_weight: float = 3.0,
) -> float:
    """Weighted RMSE: flood samples weighted 10×, elevated 3×, normal 1×."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    w = np.ones_like(y_true)
    w[y_true >= 12.0] = flood_weight
    w[(y_true >= 10.0) & (y_true < 12.0)] = elevated_weight
    return float(np.sqrt(np.average((y_pred - y_true) ** 2, weights=w)))


def _prepare_features_and_targets(train_cfg: dict):
    """Build feature matrix + horizon targets on segment-aware combined dataset."""
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")

    source_paths = {}
    for src in train_cfg["data_sources"]:
        source_paths[src["label"]] = resolve_path_from_config(
            "configs/training.yaml", src["path"]
        )
    segments = load_combined_data(
        clean_path=source_paths["2022_clean"],
        generated_path=source_paths["2023_generated"],
    )

    X_features = build_features_from_segments(
        segments, upstream_stations=UPSTREAM_STATIONS, target=target_station,
        extra_columns=[RAINFALL_COLUMN],
    )
    y_parts = []
    for seg in segments:
        ys = build_targets(seg.df, horizons, horizon_steps, target=target_station)
        y_parts.append(pd.concat({h: s for h, s in ys.items()}, axis=1))
    y_all = pd.concat(y_parts, axis=0)
    y_horizons = {h: y_all[h] for h in horizons}
    X_full, y_horizons = align_features_targets(X_features, y_horizons)

    # Also return full target series (raw levels) for event detection
    target_full = pd.concat([seg.df[target_station] for seg in segments])
    target_full = target_full.loc[~target_full.index.duplicated()]

    return X_full, y_horizons, target_full, horizons, target_station, segments


def _get_best_model(name: str, model_cfg: dict):
    defs = get_model_definitions(model_cfg)
    if name not in defs:
        raise KeyError(f"{name} not in model defs. Available: {list(defs)}")
    template, use_scaled = defs[name]
    return template, use_scaled


def evaluate_flood_event(
    event: pd.Series,
    X_full: pd.DataFrame,
    y_horizons: dict[int, pd.Series],
    horizons: list[int],
    model_cfg: dict,
    model_name: str,
) -> list[dict]:
    """Train on all-except-buffer, test on window around peak. Returns rows."""
    buffer = pd.Timedelta(hours=BUFFER_HOURS)
    test_w = pd.Timedelta(hours=TEST_WINDOW_HOURS)

    train_exclude = (event["peak_time"] - buffer, event["peak_time"] + buffer)
    test_window = (event["peak_time"] - test_w, event["peak_time"] + test_w)

    # Training mask: exclude buffer around this event
    idx = X_full.index
    train_mask = ~((idx >= train_exclude[0]) & (idx <= train_exclude[1]))
    test_mask = (idx >= test_window[0]) & (idx <= test_window[1])

    if test_mask.sum() == 0:
        return []

    X_train = X_full.loc[train_mask]
    X_test = X_full.loc[test_mask]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    template, use_scaled = _get_best_model(model_name, model_cfg)
    rows = []
    for h in horizons:
        y_train = y_horizons[h].loc[train_mask]
        y_test = y_horizons[h].loc[test_mask]
        if len(y_test) == 0:
            continue

        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        model = clone(template)
        model.fit(Xtr, y_train)
        y_pred = pd.Series(model.predict(Xte), index=y_test.index)

        # Metrics on test window
        met = calc_metrics(y_test.values, y_pred.values)

        # Peak prediction quality
        peak_idx = y_test.idxmax()
        peak_true = float(y_test.max())
        peak_pred = float(y_pred.loc[peak_idx])

        rows.append({
            "event_id": event["event_id"],
            "peak_time": event["peak_time"],
            "peak_value": event["peak_value"],
            "horizon": h,
            "model": model_name,
            "n_test": len(y_test),
            "RMSE": met["RMSE"],
            "MAE": met["MAE"],
            "NSE": met["NSE"],
            "PBIAS": met["PBIAS"],
            "peak_true": peak_true,
            "peak_pred": peak_pred,
            "peak_error": peak_pred - peak_true,
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flood Event LOOCV framework")
    p.add_argument("--model", default="CatBoost", help="Model to evaluate")
    p.add_argument("--horizons", nargs="+", type=int, default=None,
                   help="Subset of horizons (default: all)")
    p.add_argument("--max-events", type=int, default=None,
                   help="Limit to N events (default: all detected)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")
    horizons = args.horizons or train_cfg.get("horizons", HORIZONS)

    t0 = time.time()
    print("=" * 75)
    print("FLOOD EVENT LEAVE-ONE-OUT CV")
    print("=" * 75)

    X_full, y_horizons, target_full, _horizons_cfg, target_station, segments = \
        _prepare_features_and_targets(train_cfg)

    events = detect_major_floods(target_full)
    if args.max_events:
        events = events.head(args.max_events)
    print(f"\nDetected {len(events)} major flood events "
          f"(peak >= {PEAK_THRESHOLD}m, duration >= {MIN_DURATION_STEPS} steps):")
    print(events[["event_id", "peak_time", "peak_value", "duration_steps"]].to_string())
    print(f"\nModel: {args.model}  |  Horizons: {horizons}")
    print(f"Buffer: ±{BUFFER_HOURS}h  |  Test window: ±{TEST_WINDOW_HOURS}h")

    all_rows: list[dict] = []
    for i, ev in events.iterrows():
        print(f"\n[{i+1}/{len(events)}] {ev['event_id']} peak={ev['peak_value']:.2f}m")
        rows = evaluate_flood_event(ev, X_full, y_horizons, horizons, model_cfg, args.model)
        all_rows.extend(rows)
        for r in rows:
            print(f"  h{r['horizon']}: RMSE={r['RMSE']:.4f}  NSE={r['NSE']:+.4f}  "
                  f"peak_pred={r['peak_pred']:.2f} vs {r['peak_true']:.2f}  "
                  f"(err={r['peak_error']:+.3f})")

    df = pd.DataFrame(all_rows)
    out_path = PROJECT_ROOT / "reports" / "tables" / "flood_event_cv.xlsx"
    df.to_excel(out_path, index=False)

    # Aggregate per horizon
    print("\n" + "=" * 75)
    print("FLOOD-EVENT AGGREGATE METRICS (across all events)")
    print("=" * 75)
    agg = df.groupby("horizon").agg(
        n_events=("event_id", "count"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_median=("RMSE", "median"),
        RMSE_max=("RMSE", "max"),
        NSE_mean=("NSE", "mean"),
        peak_err_mean=("peak_error", "mean"),
        peak_err_abs_mean=("peak_error", lambda x: x.abs().mean()),
    )
    print(agg.round(4).to_string())

    # === PEAK-WEIGHTED RMSE on NORMAL TEST SET (existing temporal split) ===
    print("\n" + "=" * 75)
    print("PEAK-WEIGHTED RMSE on existing temporal test set (20% of 2023)")
    print("=" * 75)
    train_split = train_cfg.get("train_split", 0.8)
    seg_2023_start = segments[1].df.index.min()
    idx_2023 = X_full.index >= seg_2023_start
    n_2023_train = int(idx_2023.sum() * train_split)
    split_idx = (~idx_2023).sum() + n_2023_train

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index,
    )

    template, use_scaled = _get_best_model(args.model, model_cfg)
    w_rows = []
    print(f"{'H':>2s} {'RMSE':>8s} {'wRMSE':>8s} {'NSE':>8s} {'peak_err':>10s}")
    for h in horizons:
        y_train = y_horizons[h].iloc[:split_idx]
        y_test = y_horizons[h].iloc[split_idx:]
        Xtr = X_train_s if use_scaled else X_train_raw
        Xte = X_test_s if use_scaled else X_test_raw
        model = clone(template)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        met = calc_metrics(y_test.values, y_pred)
        wrmse = peak_weighted_rmse(y_test.values, y_pred)
        w_rows.append({
            "horizon": h, "model": args.model,
            "RMSE": met["RMSE"], "wRMSE": wrmse, "NSE": met["NSE"], "MAE": met["MAE"],
        })
        print(f"{h:>2d} {met['RMSE']:>8.4f} {wrmse:>8.4f} {met['NSE']:>8.4f} "
              f"{(np.abs(y_pred - y_test.values)).max():>10.4f}")

    # Save weighted metrics
    w_df = pd.DataFrame(w_rows)
    w_out = PROJECT_ROOT / "reports" / "tables" / "peak_weighted_rmse.xlsx"
    w_df.to_excel(w_out, index=False)

    print(f"\nCompleted in {time.time()-t0:.1f}s")
    print(f"Flood events CV : {out_path}")
    print(f"Peak-weighted    : {w_out}")


if __name__ == "__main__":
    main()
