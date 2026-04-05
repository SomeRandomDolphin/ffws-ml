"""Target smoothing experiment — uji apakah men-smooth target saat training
menurunkan RMSE dengan menghilangkan noise sensor.

Hipotesis
---------
RMSE h1 (0.175) sudah mendekati std 30-menit differential target (0.179) —
yaitu noise floor natural. Sebagian besar dari error mungkin berasal dari
noise sensor, bukan kesalahan model. Dengan smoothing target saat training,
model akan belajar pola "benar" (denoised) dan prediksinya bisa lebih dekat
ke ground truth yang sebenarnya.

Setup
-----
- Training target: ``y_smoothed`` dari rolling median/mean
- Evaluation target: **y_raw** (original) — apples-to-apples dengan baseline
- Feature matrix tidak di-smooth (menghindari leakage)
- Smoothing menggunakan centered window (boleh untuk target, karena label)

Configurations
--------------
- RAW           : baseline (no smoothing)
- MED3          : rolling median window=3 (90 menit, center=True)
- MED5          : rolling median window=5 (150 menit)
- MEAN3         : rolling mean   window=3

Usage
-----
    python training/run_smoothing_experiment.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_features_from_segments,
    build_targets,
)
from dhompo.data.loader import (
    RAINFALL_COLUMN,
    UPSTREAM_STATIONS,
    load_combined_data,
)
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


SMOOTHING_CONFIGS = {
    "RAW":   {"window": None, "method": None},
    "MED3":  {"window": 3,    "method": "median"},
    "MED5":  {"window": 5,    "method": "median"},
    "MEAN3": {"window": 3,    "method": "mean"},
}


def _smooth_series(y: pd.Series, window: int | None, method: str | None) -> pd.Series:
    """Apply centered rolling smoothing. Returns original if window is None."""
    if window is None or method is None:
        return y
    roll = y.rolling(window, center=True, min_periods=1)
    if method == "median":
        return roll.median()
    if method == "mean":
        return roll.mean()
    raise ValueError(f"Unknown smoothing method: {method}")


def _prepare_data(train_cfg: dict, smoothing: dict, include_rainfall: bool = True):
    """Return train/test matrices + BOTH smoothed & raw horizon targets.

    Model dilatih pada y_smoothed, dievaluasi pada y_raw.
    """
    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    extra_columns = [RAINFALL_COLUMN] if include_rainfall else None

    data_sources = train_cfg["data_sources"]
    source_paths = {}
    for src in data_sources:
        resolved = resolve_path_from_config("configs/training.yaml", src["path"])
        source_paths[src["label"]] = resolved

    segments = load_combined_data(
        clean_path=source_paths["2022_clean"],
        generated_path=source_paths["2023_generated"],
    )

    # Features dari seg.df ORIGINAL (tidak di-smooth → no leakage)
    X_features = build_features_from_segments(
        segments,
        upstream_stations=UPSTREAM_STATIONS,
        target=target_station,
        extra_columns=extra_columns,
    )

    # Target: build dari dua versi seg.df — smoothed & raw
    y_parts_smooth, y_parts_raw = [], []
    for seg in segments:
        df_raw = seg.df
        df_smooth = df_raw.copy()
        df_smooth[target_station] = _smooth_series(
            df_raw[target_station], smoothing["window"], smoothing["method"]
        )
        ys = build_targets(df_smooth, horizons, horizon_steps, target=target_station)
        yr = build_targets(df_raw,    horizons, horizon_steps, target=target_station)
        y_parts_smooth.append(pd.concat({h: s for h, s in ys.items()}, axis=1))
        y_parts_raw.append(pd.concat({h: s for h, s in yr.items()}, axis=1))

    y_smooth_all = pd.concat(y_parts_smooth, axis=0)
    y_raw_all = pd.concat(y_parts_raw, axis=0)
    y_smooth_h = {h: y_smooth_all[h] for h in horizons}
    y_raw_h = {h: y_raw_all[h] for h in horizons}

    X_full, y_smooth_h = align_features_targets(X_features, y_smooth_h)
    # align raw to same index as X_full
    y_raw_h = {h: y_raw_all[h].loc[X_full.index] for h in horizons}

    # Split: all 2022 + 80% of 2023 → train
    seg_2023_start = segments[1].df.index.min()
    idx_2023 = X_full.index >= seg_2023_start
    n_2023 = idx_2023.sum()
    n_2023_train = int(n_2023 * train_split)
    split_idx = (~idx_2023).sum() + n_2023_train

    return (
        X_full.iloc[:split_idx],
        X_full.iloc[split_idx:],
        y_smooth_h, y_raw_h,
        split_idx, horizons,
    )


def _run_one_config(
    config_name: str,
    smoothing: dict,
    train_cfg: dict, model_cfg: dict,
    include_rainfall: bool,
    save_models_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}  (window={smoothing['window']}, method={smoothing['method']})")
    print(f"{'='*70}")

    X_train_raw, X_test_raw, y_smooth_h, y_raw_h, split_idx, horizons = _prepare_data(
        train_cfg, smoothing, include_rainfall=include_rainfall,
    )
    print(f"Features: {X_train_raw.shape[1]} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index,
    )

    model_defs = get_model_definitions(model_cfg)
    results, best_models = [], {}

    for h in horizons:
        y_train_fit = y_smooth_h[h].iloc[:split_idx]   # train on smoothed
        y_test_raw = y_raw_h[h].iloc[split_idx:]       # eval on raw
        y_test_smooth = y_smooth_h[h].iloc[split_idx:] # also track vs smoothed

        print(f"\n--- [{config_name}] Horizon +{h}h ---")
        for name, (model_template, use_scaled) in model_defs.items():
            Xtr = X_train_s if use_scaled else X_train_raw
            Xte = X_test_s if use_scaled else X_test_raw

            model = clone(model_template)
            model.fit(Xtr, y_train_fit)
            y_pred = model.predict(Xte)

            te_raw = calc_metrics(y_test_raw.values, y_pred)
            te_smooth = calc_metrics(y_test_smooth.values, y_pred)

            y_mean = y_test_raw.mean()
            nrmse = te_raw["RMSE"] / y_mean * 100 if y_mean else float("inf")

            results.append({
                "config": config_name,
                "horizon": h,
                "model": name,
                "NSE_vs_raw": te_raw["NSE"],
                "RMSE_vs_raw": te_raw["RMSE"],
                "MAE_vs_raw": te_raw["MAE"],
                "PBIAS_vs_raw": te_raw["PBIAS"],
                "NSE_vs_smooth": te_smooth["NSE"],
                "RMSE_vs_smooth": te_smooth["RMSE"],
                "nRMSE_mean_%": nrmse,
            })
            print(
                f"  {name:30s} NSE(raw)={te_raw['NSE']:.4f}  "
                f"RMSE(raw)={te_raw['RMSE']:.4f}  RMSE(smooth)={te_smooth['RMSE']:.4f}"
            )

            key = h
            if key not in best_models or te_raw["NSE"] > best_models[key][1]:
                best_models[key] = (name, te_raw["NSE"], te_raw["RMSE"], model, scaler if use_scaled else None)

    if save_models_dir is not None:
        save_models_dir.mkdir(parents=True, exist_ok=True)
        scaler_saved = False
        for h, (mname, nse, rmse, model, sc) in best_models.items():
            fname = f"{mname.lower().replace(' ', '_')}_h{h}.pkl"
            joblib.dump(model, save_models_dir / fname)
            print(f"  Saved: {save_models_dir.name}/{fname} (NSE={nse:.4f}, RMSE={rmse:.4f})")
            if sc is not None and not scaler_saved:
                joblib.dump(sc, save_models_dir / "scaler.pkl")
                scaler_saved = True

    return pd.DataFrame(results), best_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target smoothing experiment")
    p.add_argument("--no-rainfall", action="store_true", help="disable rainfall feature")
    p.add_argument("--save-best", action="store_true",
                   help="Save best models from the best smoothing config")
    p.add_argument("--configs", nargs="+", default=None,
                   help=f"Subset of configs to run. Options: {list(SMOOTHING_CONFIGS)}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")
    include_rainfall = not args.no_rainfall

    configs_to_run = args.configs or list(SMOOTHING_CONFIGS)

    t0 = time.time()
    all_results = []
    all_best = {}

    for cname in configs_to_run:
        if cname not in SMOOTHING_CONFIGS:
            print(f"Unknown config: {cname}"); continue
        df, best = _run_one_config(
            cname, SMOOTHING_CONFIGS[cname],
            train_cfg, model_cfg,
            include_rainfall=include_rainfall,
            save_models_dir=None,
        )
        all_results.append(df)
        all_best[cname] = best

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = PROJECT_ROOT / "reports" / "tables" / "experiment_target_smoothing.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(out_path, index=False)

    # Summary: best per config × horizon (by NSE_vs_raw)
    print("\n" + "=" * 75)
    print("SUMMARY: Best model per config × horizon (evaluated on RAW targets)")
    print("=" * 75)
    print(f"{'Config':<8s} {'H':>2s} {'Model':<22s} {'NSE_raw':>9s} {'RMSE_raw':>10s} {'nRMSE%':>8s}")
    print("-" * 65)
    best_rows = results_df.loc[
        results_df.groupby(["config", "horizon"])["NSE_vs_raw"].idxmax()
    ]
    for row in best_rows.itertuples():
        print(
            f"{row.config:<8s} {row.horizon:>2d} {row.model:<22s} "
            f"{row.NSE_vs_raw:>9.4f} {row.RMSE_vs_raw:>10.4f} "
            f"{row._asdict()['nRMSE_mean_%']:>7.2f}%"
        )

    # Head-to-head: each smoothed config vs RAW baseline
    print("\n" + "=" * 75)
    print("RMSE REDUCTION vs RAW baseline (best per horizon, RAW-scale evaluation)")
    print("=" * 75)
    raw_best = best_rows[best_rows["config"] == "RAW"].set_index("horizon")
    print(f"{'H':>2s} {'RMSE_RAW':>10s}", end="")
    for c in configs_to_run:
        if c == "RAW": continue
        print(f" {c:>10s}", end="")
    print()
    print("-" * 70)
    for h in sorted(raw_best.index):
        ra = raw_best.loc[h, "RMSE_vs_raw"]
        line = f"{h:>2d} {ra:>10.4f}"
        for c in configs_to_run:
            if c == "RAW": continue
            crow = best_rows[(best_rows["config"] == c) & (best_rows["horizon"] == h)]
            if crow.empty:
                line += f" {'—':>10s}"
            else:
                rc = crow.iloc[0]["RMSE_vs_raw"]
                pct = (ra - rc) / ra * 100
                line += f" {rc:.4f}({pct:+.1f}%)".rjust(19)
        print(line)

    # Save-best from best config (by average NSE_vs_raw)
    if args.save_best:
        avg_nse = {
            c: best_rows[best_rows["config"] == c]["NSE_vs_raw"].mean()
            for c in configs_to_run
        }
        winner = max(avg_nse, key=avg_nse.get)
        print(f"\nBest config by avg NSE(raw): {winner} (NSE={avg_nse[winner]:.4f})")
        save_dir = PROJECT_ROOT / "models" / f"sklearn_smoothed_{winner.lower()}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for h, (mname, nse, rmse, model, sc) in all_best[winner].items():
            fname = f"{mname.lower().replace(' ', '_')}_h{h}.pkl"
            joblib.dump(model, save_dir / fname)
            print(f"  Saved: {save_dir.name}/{fname} (NSE={nse:.4f})")
            if sc is not None and not (save_dir / "scaler.pkl").exists():
                joblib.dump(sc, save_dir / "scaler.pkl")

    print(f"\nCompleted in {time.time() - t0:.1f}s")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
