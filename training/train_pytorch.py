"""Train DhompoLSTM and log to MLflow.

Usage
-----
    python training/train_pytorch.py
    python training/train_pytorch.py --config configs/lstm_model.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow running as `python training/...py` without package install.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dhompo.config import load_serving_config, load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_forecast_features,
    build_targets,
)
from dhompo.data.loader import UPSTREAM_STATIONS, load_data
from dhompo.models.lstm import DhompoLSTM
from dhompo.models.sklearn_models import HORIZON_STEPS, HORIZONS
from training.evaluate import calc_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/lstm_model.yaml")
    p.add_argument("--train-config", default="configs/training.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--experiment", default="dhompo_pytorch")
    p.add_argument("--data", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.train_config)

    serving_cfg = load_serving_config()
    mlflow.set_tracking_uri(serving_cfg.get("mlflow_uri", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    hidden_size = cfg.get("hidden_size", 128)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.2)
    seq_len = cfg.get("seq_len", 24)
    lr = cfg.get("learning_rate", 1e-3)
    batch_size = cfg.get("batch_size", 64)
    epochs = args.epochs or cfg.get("epochs", 50)
    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    data_path = args.data or resolve_path_from_config(
        args.train_config, train_cfg.get("data_path")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    df = load_data(data_path)
    X_features = build_forecast_features(df, UPSTREAM_STATIONS, target=target_station)
    y_horizons = build_targets(df, horizons, horizon_steps, target=target_station)
    X_full, y_horizons = align_features_targets(X_features, y_horizons)

    # Build sequence tensors (batch, seq_len, features)
    X_arr = X_full.values.astype(np.float32)
    # Stack targets: (n_samples, 5)
    y_arr = np.column_stack([y_horizons[h].values for h in horizons]).astype(np.float32)

    # Create sliding windows
    n = len(X_arr) - seq_len
    X_seq = np.stack([X_arr[i : i + seq_len] for i in range(n)])
    y_seq = y_arr[seq_len:]

    split = int(n * train_split)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Model
    model = DhompoLSTM(
        input_size=X_arr.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(horizons),
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="dhompo_lstm"):
        mlflow.log_params(
            {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "seq_len": seq_len,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg_loss = epoch_loss / len(X_tr)

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    X_te_t = torch.from_numpy(X_te).to(device)
                    preds = model(X_te_t).cpu().numpy()
                # Log per-horizon NSE
                for i, h in enumerate(horizons):
                    m = calc_metrics(y_te[:, i], preds[:, i])
                    mlflow.log_metric(f"test_NSE_h{h}", m["NSE"], step=epoch)
                    mlflow.log_metric(f"test_RMSE_h{h}", m["RMSE"], step=epoch)
                print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}")

        # Save model
        mlflow.pytorch.log_model(
            model, name="lstm_model", registered_model_name="dhompo_lstm"
        )
        print("LSTM model logged to MLflow.")


if __name__ == "__main__":
    main()
