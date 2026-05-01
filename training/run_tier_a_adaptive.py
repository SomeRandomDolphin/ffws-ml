"""Train Tier-A adaptive model (ARCHITECTURE.md §3, §6).

Wires :class:`AdaptiveTierA` + sensor dropout augmentation + composite
peak-weighted/auxiliary loss against the cleaned 30-minute history. The
default run is a smoke test — small batch size, few epochs — sized to verify
the pipeline end-to-end on a developer laptop. Production training runs
override the relevant CLI flags.

Usage
-----
    python training/run_tier_a_adaptive.py --epochs 5
    python training/run_tier_a_adaptive.py --epochs 200 --batch-size 256 --mlflow

Inputs are reshaped from the existing tabular feature pipeline into a
per-station tensor of shape (batch, n_stations, features_per_station). The
seven features per station are:

    [t0, lag1, lag2, lag3, rolling_mean_3h, rolling_std_3h, diff1]

The autoregressive lag tensor takes the last 6 readings of Dhompo (3 h of
30-minute history). Targets are Dhompo water level at h+1..h+5 hours.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dhompo.data.loader import (
    ALL_STATIONS,
    TARGET_STATION,
    load_data,
)
from dhompo.models.adaptive import AdaptiveTierA, AdaptiveTierAConfig
from dhompo.training.losses import CompositeLoss, LossConfig
from dhompo.training.sensor_dropout import (
    DropoutSchedule,
    apply_sensor_dropout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tier_a")

FEATURES_PER_STATION = 7
AR_LAG_DIM = 6
HORIZON_STEPS_PER_HOUR = 2  # 30-min cadence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mlflow", action="store_true",
                    help="Log run to MLflow (skip for smoke tests).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default=None,
                    help="Override path to data-clean.csv.")
    p.add_argument("--checkpoint-dir", default=None,
                    help="Where to write best.pt + normalizer.pkl. "
                         "Defaults to artifacts/tier_a_adaptive/.")
    return p.parse_args()


@dataclass
class Normalizer:
    """Per-station/feature z-score stats fit on training data only.

    Targets and auxiliary targets are intentionally NOT normalized — the
    peak-weighted loss thresholds (7.0 elevated, 9.0 flood) are expressed
    in raw water-level units, and station-t0 aux targets need to remain
    comparable to predictions emitted in raw units at inference time.
    """

    mean_feats: np.ndarray   # (n_stations, features_per_station)
    std_feats: np.ndarray
    mean_ar: np.ndarray      # (ar_lag_dim,)
    std_ar: np.ndarray

    @classmethod
    def fit(cls, feats: torch.Tensor, ar: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        f = feats.numpy()
        a = ar.numpy()
        mean_feats = f.mean(axis=0)
        std_feats = f.std(axis=0).clip(min=eps)
        mean_ar = a.mean(axis=0)
        std_ar = a.std(axis=0).clip(min=eps)
        return cls(mean_feats, std_feats, mean_ar, std_ar)

    def apply(self, feats: torch.Tensor, ar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mf = torch.from_numpy(self.mean_feats).to(feats)
        sf = torch.from_numpy(self.std_feats).to(feats)
        ma = torch.from_numpy(self.mean_ar).to(ar)
        sa = torch.from_numpy(self.std_ar).to(ar)
        return (feats - mf) / sf, (ar - ma) / sa


def build_per_station_features(df: pd.DataFrame) -> np.ndarray:
    """Reshape sensor history into (n_timesteps, n_stations, features_per_station).

    Features per station (in fixed order):
      0 t0 value
      1 lag-1 (30 min)
      2 lag-2 (60 min)
      3 lag-3 (90 min)
      4 rolling mean 3 h (6 readings)
      5 rolling std  3 h (6 readings)
      6 first-difference at t0
    """
    n_steps = len(df)
    n_stations = len(ALL_STATIONS)
    out = np.zeros((n_steps, n_stations, FEATURES_PER_STATION), dtype=np.float32)
    for s_idx, station in enumerate(ALL_STATIONS):
        col = df[station]
        out[:, s_idx, 0] = col.to_numpy()
        out[:, s_idx, 1] = col.shift(1).to_numpy()
        out[:, s_idx, 2] = col.shift(2).to_numpy()
        out[:, s_idx, 3] = col.shift(3).to_numpy()
        out[:, s_idx, 4] = col.rolling(window=6, min_periods=6).mean().to_numpy()
        out[:, s_idx, 5] = col.rolling(window=6, min_periods=6).std(ddof=0).to_numpy()
        out[:, s_idx, 6] = col.diff().to_numpy()
    return out


def build_ar_lags(df: pd.DataFrame) -> np.ndarray:
    target = df[TARGET_STATION]
    lags = np.stack([target.shift(i).to_numpy() for i in range(AR_LAG_DIM)], axis=-1)
    return lags.astype(np.float32)


def build_targets(df: pd.DataFrame) -> np.ndarray:
    target = df[TARGET_STATION]
    horizons = []
    for h in range(1, 6):
        horizons.append(target.shift(-h * HORIZON_STEPS_PER_HOUR).to_numpy())
    return np.stack(horizons, axis=-1).astype(np.float32)


def assemble_tensors(df: pd.DataFrame):
    feats = build_per_station_features(df)
    ar = build_ar_lags(df)
    y = build_targets(df)

    valid = (
        ~np.isnan(feats).any(axis=(1, 2))
        & ~np.isnan(ar).any(axis=1)
        & ~np.isnan(y).any(axis=1)
    )
    feats = feats[valid]
    ar = ar[valid]
    y = y[valid]
    aux_y = feats[:, :, 0]  # auxiliary target = each station's t0 value
    mask = np.ones((feats.shape[0], feats.shape[1]), dtype=bool)
    return (
        torch.from_numpy(feats),
        torch.from_numpy(mask),
        torch.from_numpy(ar),
        torch.from_numpy(y),
        torch.from_numpy(aux_y),
    )


def temporal_split(*tensors, train_frac: float = 0.8):
    n = tensors[0].shape[0]
    cutoff = int(n * train_frac)
    train = [t[:cutoff] for t in tensors]
    test = [t[cutoff:] for t in tensors]
    return train, test


def train_one_epoch(model, loader, loss_fn, optimiser, device, schedule):
    model.train()
    totals = {"total": 0.0, "main": 0.0, "aux": 0.0}
    n_batches = 0
    for feats, mask, ar, y, aux_y in loader:
        feats = feats.to(device); mask = mask.to(device); ar = ar.to(device)
        y = y.to(device); aux_y = aux_y.to(device)
        target_h1 = y[:, 0]
        feats_aug, mask_aug = apply_sensor_dropout(
            feats, mask, target_h1, schedule=schedule,
        )
        pred_h, pred_aux = model(feats_aug, mask_aug, ar)
        out = loss_fn(pred_h, y, pred_aux, aux_y, aux_mask=mask_aug)
        optimiser.zero_grad()
        out["total"].backward()
        optimiser.step()
        for k in totals:
            totals[k] += out[k].item()
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = {"total": 0.0, "main": 0.0, "aux": 0.0}
    n_batches = 0
    for feats, mask, ar, y, aux_y in loader:
        feats = feats.to(device); mask = mask.to(device); ar = ar.to(device)
        y = y.to(device); aux_y = aux_y.to(device)
        pred_h, pred_aux = model(feats, mask, ar)
        out = loss_fn(pred_h, y, pred_aux, aux_y)
        for k in totals:
            totals[k] += out[k].item()
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = args.data or str(PROJECT_ROOT / "data" / "data-clean.csv")
    log.info("Loading data from %s", data_path)
    df = load_data(Path(data_path)).interpolate(limit_direction="both")

    feats, mask, ar, y, aux_y = assemble_tensors(df)
    log.info("Assembled %d samples (after NaN drop).", feats.shape[0])

    (tr_feats, tr_mask, tr_ar, tr_y, tr_aux), \
        (te_feats, te_mask, te_ar, te_y, te_aux) = temporal_split(
            feats, mask, ar, y, aux_y,
        )

    normalizer = Normalizer.fit(tr_feats, tr_ar)
    tr_feats, tr_ar = normalizer.apply(tr_feats, tr_ar)
    te_feats, te_ar = normalizer.apply(te_feats, te_ar)

    train_ds = TensorDataset(tr_feats, tr_mask, tr_ar, tr_y, tr_aux)
    test_ds = TensorDataset(te_feats, te_mask, te_ar, te_y, te_aux)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    cfg = AdaptiveTierAConfig(features_per_station=FEATURES_PER_STATION,
                              ar_lag_dim=AR_LAG_DIM)
    model = AdaptiveTierA(cfg).to(args.device)
    loss_fn = CompositeLoss(LossConfig())
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = DropoutSchedule()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir \
        else PROJECT_ROOT / "artifacts" / "tier_a_adaptive"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "normalizer.pkl", "wb") as fh:
        pickle.dump(normalizer, fh)
    best_test_main = float("inf")

    if args.mlflow:
        import mlflow
        mlflow.set_experiment("dhompo_tier_a_adaptive")
        run_ctx = mlflow.start_run()
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "embedding_dim": cfg.embedding_dim,
            "hidden_dim": cfg.hidden_dim, "dropout_p": cfg.dropout_p,
            "drop_normal": schedule.normal,
            "drop_elevated": schedule.elevated,
            "drop_flood": schedule.flood,
        })
    else:
        run_ctx = None

    try:
        for epoch in range(args.epochs):
            tr_metrics = train_one_epoch(
                model, train_loader, loss_fn, optimiser, args.device, schedule,
            )
            te_metrics = evaluate(model, test_loader, loss_fn, args.device)
            log.info(
                "epoch=%d train_total=%.4f train_main=%.4f train_aux=%.4f "
                "test_total=%.4f test_main=%.4f",
                epoch, tr_metrics["total"], tr_metrics["main"], tr_metrics["aux"],
                te_metrics["total"], te_metrics["main"],
            )
            if te_metrics["main"] < best_test_main:
                best_test_main = te_metrics["main"]
                torch.save({"model_state": model.state_dict(),
                            "config": cfg, "epoch": epoch,
                            "test_main": best_test_main},
                           ckpt_dir / "best.pt")
                log.info("  ↳ new best test_main=%.4f saved to %s",
                         best_test_main, ckpt_dir / "best.pt")
            if args.mlflow:
                import mlflow
                mlflow.log_metrics({
                    f"train_{k}": v for k, v in tr_metrics.items()
                }, step=epoch)
                mlflow.log_metrics({
                    f"test_{k}": v for k, v in te_metrics.items()
                }, step=epoch)
    finally:
        if run_ctx is not None:
            import mlflow
            mlflow.end_run()


if __name__ == "__main__":
    main()
