"""Peak-weighted training experiment.

Tujuan: perbaiki systematic peak under-prediction yang terdiagnosis dari
flood_event_cv (mean peak_error -0.38 s/d -0.82 m di semua horizon).

Dengan memberi bobot lebih besar pada sampel flood/elevated saat training,
model dipaksa mempelajari pola peak lebih akurat.

Weight schemes
--------------
- W0_uniform   : baseline (tanpa sample weights)
- W1_moderate  : normal=1, elevated=3, flood=10
- W2_aggressive: normal=1, elevated=5, flood=20
- W3_quadratic : w = 1 + max(0, y - 9)²  (continuous boost)

Evaluation
----------
- Flood Event LOOCV (8 events) → RMSE/NSE/peak_error per event
- NormOps temporal test → RMSE/NSE/wRMSE
- Side-by-side comparison across weight schemes

Usage
-----
    python training/run_peak_weighted_experiment.py
    python training/run_peak_weighted_experiment.py --schemes W0_uniform W1_moderate
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

from dhompo.config import load_yaml_config
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics
from training.flood_event_cv import (
    _prepare_features_and_targets,
    detect_major_floods,
    peak_weighted_rmse,
    BUFFER_HOURS,
    TEST_WINDOW_HOURS,
)


def make_weights(y: np.ndarray | pd.Series, scheme: str) -> np.ndarray:
    y = np.asarray(y)
    if scheme == "W0_uniform":
        return np.ones_like(y, dtype=float)
    if scheme == "W1_moderate":
        w = np.ones_like(y, dtype=float)
        w[(y >= 10.0) & (y < 12.0)] = 3.0
        w[y >= 12.0] = 10.0
        return w
    if scheme == "W2_aggressive":
        w = np.ones_like(y, dtype=float)
        w[(y >= 10.0) & (y < 12.0)] = 5.0
        w[y >= 12.0] = 20.0
        return w
    if scheme == "W3_quadratic":
        return 1.0 + np.maximum(0.0, y - 9.0) ** 2
    raise ValueError(f"Unknown scheme: {scheme}")


def _fit_with_weights(model, X, y, sample_weight):
    """Fit sklearn estimator passing sample_weight if supported."""
    # CatBoost, sklearn tree models, linear regression all support sample_weight
    model.fit(X, y, sample_weight=sample_weight)
    return model


def run_flood_event_cv_weighted(
    events: pd.DataFrame,
    X_full: pd.DataFrame,
    y_horizons: dict,
    horizons: list,
    model_cfg: dict,
    model_name: str,
    scheme: str,
) -> list[dict]:
    """Flood event LOOCV with peak-weighted training."""
    buffer = pd.Timedelta(hours=BUFFER_HOURS)
    test_w = pd.Timedelta(hours=TEST_WINDOW_HOURS)
    defs = get_model_definitions(model_cfg)
    template, use_scaled = defs[model_name]

    rows = []
    idx = X_full.index
    for _, event in events.iterrows():
        train_exclude = (event["peak_time"] - buffer, event["peak_time"] + buffer)
        test_window = (event["peak_time"] - test_w, event["peak_time"] + test_w)
        train_mask = ~((idx >= train_exclude[0]) & (idx <= train_exclude[1]))
        test_mask = (idx >= test_window[0]) & (idx <= test_window[1])
        if test_mask.sum() == 0:
            continue

        X_train = X_full.loc[train_mask]
        X_test = X_full.loc[test_mask]

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        for h in horizons:
            y_train = y_horizons[h].loc[train_mask]
            y_test = y_horizons[h].loc[test_mask]
            Xtr = X_train_s if use_scaled else X_train
            Xte = X_test_s if use_scaled else X_test

            w = make_weights(y_train, scheme)
            model = clone(template)
            _fit_with_weights(model, Xtr, y_train, w)
            y_pred = pd.Series(model.predict(Xte), index=y_test.index)
            met = calc_metrics(y_test.values, y_pred.values)
            peak_idx = y_test.idxmax()
            rows.append({
                "scheme": scheme,
                "event_id": event["event_id"],
                "peak_time": event["peak_time"],
                "horizon": h,
                "RMSE": met["RMSE"], "MAE": met["MAE"], "NSE": met["NSE"],
                "peak_true": float(y_test.max()),
                "peak_pred": float(y_pred.loc[peak_idx]),
                "peak_error": float(y_pred.loc[peak_idx] - y_test.max()),
            })
    return rows


def run_normops_weighted(
    X_full: pd.DataFrame, y_horizons: dict, horizons: list,
    segments: list, train_cfg: dict, model_cfg: dict,
    model_name: str, scheme: str,
) -> list[dict]:
    """Train with weights, eval on existing temporal test split."""
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

    defs = get_model_definitions(model_cfg)
    template, use_scaled = defs[model_name]
    rows = []
    for h in horizons:
        y_train = y_horizons[h].iloc[:split_idx]
        y_test = y_horizons[h].iloc[split_idx:]
        Xtr = X_train_s if use_scaled else X_train_raw
        Xte = X_test_s if use_scaled else X_test_raw
        w = make_weights(y_train, scheme)
        model = clone(template)
        _fit_with_weights(model, Xtr, y_train, w)
        y_pred = model.predict(Xte)
        met = calc_metrics(y_test.values, y_pred)
        wrmse = peak_weighted_rmse(y_test.values, y_pred)
        rows.append({
            "scheme": scheme, "horizon": h,
            "RMSE": met["RMSE"], "wRMSE": wrmse,
            "NSE": met["NSE"], "MAE": met["MAE"], "PBIAS": met["PBIAS"],
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Peak-weighted training experiment")
    p.add_argument("--model", default="CatBoost")
    p.add_argument("--schemes", nargs="+",
                   default=["W0_uniform", "W1_moderate", "W2_aggressive", "W3_quadratic"])
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")
    horizons = args.horizons or train_cfg.get("horizons", HORIZONS)

    t0 = time.time()
    print("=" * 75)
    print(f"PEAK-WEIGHTED TRAINING — model={args.model} | schemes={args.schemes}")
    print("=" * 75)

    X_full, y_horizons, target_full, _, _, segments = _prepare_features_and_targets(train_cfg)
    events = detect_major_floods(target_full)
    print(f"\nDetected {len(events)} major flood events")

    # Run all schemes on Flood Event CV
    flood_rows = []
    for scheme in args.schemes:
        print(f"\n[FloodCV] scheme={scheme}...")
        s_t = time.time()
        rows = run_flood_event_cv_weighted(
            events, X_full, y_horizons, horizons, model_cfg, args.model, scheme,
        )
        flood_rows.extend(rows)
        print(f"  done in {time.time() - s_t:.1f}s ({len(rows)} rows)")

    flood_df = pd.DataFrame(flood_rows)

    # Run all schemes on NormOps
    normops_rows = []
    for scheme in args.schemes:
        print(f"\n[NormOps] scheme={scheme}...")
        rows = run_normops_weighted(
            X_full, y_horizons, horizons, segments,
            train_cfg, model_cfg, args.model, scheme,
        )
        normops_rows.extend(rows)
    normops_df = pd.DataFrame(normops_rows)

    # Save
    out_flood = PROJECT_ROOT / "reports" / "tables" / "peak_weighted_flood_cv.xlsx"
    out_norm = PROJECT_ROOT / "reports" / "tables" / "peak_weighted_normops.xlsx"
    flood_df.to_excel(out_flood, index=False)
    normops_df.to_excel(out_norm, index=False)

    # ============ REPORTS ============
    print("\n" + "=" * 75)
    print("FLOOD EVENT CV — Aggregate per (scheme × horizon)")
    print("=" * 75)
    agg = flood_df.groupby(["scheme", "horizon"]).agg(
        RMSE_mean=("RMSE", "mean"),
        NSE_mean=("NSE", "mean"),
        peak_err_mean=("peak_error", "mean"),
        peak_err_abs_mean=("peak_error", lambda x: x.abs().mean()),
    ).round(4)
    print(agg.to_string())

    print("\n" + "=" * 75)
    print("NORMOPS — Per (scheme × horizon)")
    print("=" * 75)
    print(normops_df.round(4).to_string(index=False))

    # ============ DELTA vs baseline (W0_uniform) ============
    if "W0_uniform" in args.schemes:
        print("\n" + "=" * 75)
        print("HEAD-TO-HEAD vs W0_uniform baseline (FLOOD CV)")
        print("=" * 75)
        base_flood = agg.loc["W0_uniform"]
        print(f"{'Scheme':<17s} {'H':>2s} {'ΔRMSE':>9s} {'%':>7s} {'Δ|peak_err|':>14s} "
              f"{'ΔNSE':>9s}")
        print("-" * 70)
        for scheme in args.schemes:
            if scheme == "W0_uniform": continue
            sub = agg.loc[scheme]
            for h in horizons:
                br, bpe, bn = base_flood.loc[h, ["RMSE_mean", "peak_err_abs_mean", "NSE_mean"]]
                sr, spe, sn = sub.loc[h, ["RMSE_mean", "peak_err_abs_mean", "NSE_mean"]]
                dr = sr - br; dpe = spe - bpe; dn = sn - bn
                print(f"{scheme:<17s} {h:>2d} {dr:>+9.4f} {dr/br*100:>+6.1f}% "
                      f"{dpe:>+14.4f} {dn:>+9.4f}")

        print("\n" + "=" * 75)
        print("HEAD-TO-HEAD vs W0_uniform baseline (NORMOPS)")
        print("=" * 75)
        base_nrm = normops_df[normops_df["scheme"] == "W0_uniform"].set_index("horizon")
        print(f"{'Scheme':<17s} {'H':>2s} {'ΔRMSE':>9s} {'ΔwRMSE':>10s} {'ΔNSE':>9s}")
        print("-" * 56)
        for scheme in args.schemes:
            if scheme == "W0_uniform": continue
            sub = normops_df[normops_df["scheme"] == scheme].set_index("horizon")
            for h in horizons:
                dr = sub.loc[h, "RMSE"] - base_nrm.loc[h, "RMSE"]
                dw = sub.loc[h, "wRMSE"] - base_nrm.loc[h, "wRMSE"]
                dn = sub.loc[h, "NSE"] - base_nrm.loc[h, "NSE"]
                print(f"{scheme:<17s} {h:>2d} {dr:>+9.4f} {dw:>+10.4f} {dn:>+9.4f}")

    print(f"\nCompleted in {time.time() - t0:.1f}s")
    print(f"Flood CV: {out_flood}")
    print(f"NormOps : {out_norm}")


if __name__ == "__main__":
    main()
