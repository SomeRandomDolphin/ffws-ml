"""Training model adaptive Tier-A.

Usage:
    python training/run_tier_a_adaptive.py --epochs 5
    python training/run_tier_a_adaptive.py --epochs 200 --batch-size 256 --mlflow
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dhompo.data.loader import load_data
from dhompo.data.synthesis import (
    AugmentationConfig,
    AugmentationSchedule,
    SyntheticAugmenter,
)
from dhompo.data.tier_a_features import (
    AR_LAG_DIM,
    FEATURES_PER_STATION,
    build_ar_lags,
    build_per_station_features,
    build_targets,
)
from dhompo.models.adaptive import AdaptiveTierA, AdaptiveTierAConfig
from dhompo.training.losses import CompositeLoss, LossConfig
from dhompo.training.normalizer import Normalizer
from dhompo.training.sensor_dropout import (
    DropoutSchedule,
    apply_sensor_dropout,
)

AUTO_TIGHTEN_THRESHOLD: float = 0.30
AUTO_TIGHTEN_FACTOR: float = 0.20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tier_a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mlflow", action="store_true",
                    help="Log run ke MLflow (skip untuk smoke test).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default=None,
                    help="Override path ke data-clean.csv.")
    p.add_argument("--checkpoint-dir", default=None,
                    help="Lokasi tulis best.pt + normalizer.pkl. "
                         "Default: artifacts/tier_a_adaptive/.")
    p.add_argument("--no-synthesis", action="store_true",
                    help="Matikan augmenter sintetis basin-coherent "
                         "untuk run ablasi.")
    return p.parse_args()


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
    aux_y = feats[:, :, 0]  # target auxiliary = nilai t0 tiap stasiun
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


def train_one_epoch(
    model, loader, loss_fn, optimiser, device, schedule, normalizer, augmenter,
):
    """Satu pass training loader; mengembalikan rata-rata loss + hitungan aug."""
    model.train()
    totals = {"total": 0.0, "main": 0.0, "aux": 0.0}
    n_batches = 0
    n_attempted = 0
    n_rejected = 0
    for feats_raw, mask, ar_raw, y, aux_y in loader:
        feats_raw = feats_raw.to(device); mask = mask.to(device)
        ar_raw = ar_raw.to(device)
        y = y.to(device); aux_y = aux_y.to(device)
        target_h1 = y[:, 0]

        if augmenter is not None:
            feats_raw, ar_raw, _, stats = augmenter.augment(
                feats_raw, ar_raw, target_h1,
            )
            n_attempted += stats.n_attempted
            n_rejected += stats.n_rejected

        feats, ar = normalizer.apply(feats_raw, ar_raw)
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
    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics["aug_attempted"] = n_attempted
    metrics["aug_rejected"] = n_rejected
    metrics["aug_rejection_rate"] = (
        n_rejected / n_attempted if n_attempted > 0 else 0.0
    )
    return metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, normalizer):
    model.eval()
    totals = {"total": 0.0, "main": 0.0, "aux": 0.0}
    n_batches = 0
    for feats_raw, mask, ar_raw, y, aux_y in loader:
        feats_raw = feats_raw.to(device); mask = mask.to(device)
        ar_raw = ar_raw.to(device)
        y = y.to(device); aux_y = aux_y.to(device)
        feats, ar = normalizer.apply(feats_raw, ar_raw)
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
    # Normalisasi dilakukan per batch supaya augmenter melihat nilai mentah
    # (jitter sigma didefinisikan dalam satuan tinggi muka air mentah).

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

    if args.no_synthesis:
        augmenter = None
        log.info("Synthetic augmenter DISABLED via --no-synthesis.")
    else:
        station_std = tr_feats[:, :, 0].std(dim=0).clamp(min=1e-6)
        training_max = tr_feats[:, :, 0].max(dim=0).values
        augmenter = SyntheticAugmenter(
            station_std=station_std,
            training_max=training_max,
            schedule=AugmentationSchedule(),
            config=AugmentationConfig(),
        )
        log.info(
            "Synthetic augmenter enabled (sigma=%.4f, scale=[%.2f,%.2f]).",
            augmenter.config.jitter_sigma_pct,
            augmenter.config.scale_low, augmenter.config.scale_high,
        )

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
                normalizer, augmenter,
            )
            te_metrics = evaluate(
                model, test_loader, loss_fn, args.device, normalizer,
            )
            log.info(
                "epoch=%d train_total=%.4f train_main=%.4f train_aux=%.4f "
                "test_total=%.4f test_main=%.4f aug_rate=%.2f%%",
                epoch, tr_metrics["total"], tr_metrics["main"], tr_metrics["aux"],
                te_metrics["total"], te_metrics["main"],
                100.0 * tr_metrics["aug_rejection_rate"],
            )
            if augmenter is not None and tr_metrics["aug_rejection_rate"] > AUTO_TIGHTEN_THRESHOLD:
                new_cfg = augmenter.tighten(AUTO_TIGHTEN_FACTOR)
                log.info(
                    "  auto-tighten triggered (rate=%.2f%%): sigma=%.4f scale=[%.3f,%.3f]",
                    100.0 * tr_metrics["aug_rejection_rate"],
                    new_cfg.jitter_sigma_pct,
                    new_cfg.scale_low, new_cfg.scale_high,
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
                if augmenter is not None:
                    aug_cfg = augmenter.config
                    mlflow.log_metrics({
                        "synthesis_jitter_sigma_pct": aug_cfg.jitter_sigma_pct,
                        "synthesis_scale_low": aug_cfg.scale_low,
                        "synthesis_scale_high": aug_cfg.scale_high,
                    }, step=epoch)
    finally:
        if run_ctx is not None:
            import mlflow
            mlflow.end_run()


if __name__ == "__main__":
    main()
