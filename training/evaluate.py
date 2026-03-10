"""Shared evaluation metrics for Dhompo flood forecasting.

Ported from research/create_02_modeling.py — calc_metrics().

NSE (Nash-Sutcliffe Efficiency) is the primary metric for hydrological models:
  NSE = 1  → perfect model
  NSE = 0  → model no better than mean prediction
  NSE < 0  → worse than mean
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, R², NSE, and PBIAS.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values.

    Returns
    -------
    dict with keys: RMSE, MAE, R2, NSE, PBIAS
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("-inf")

    pbias = float(100.0 * np.sum(y_pred - y_true) / np.sum(y_true)) if np.sum(y_true) != 0 else float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "PBIAS": pbias}


def performance_grade(nse: float) -> str:
    """Classify NSE into performance category (Moriasi et al. 2007)."""
    if nse > 0.75:
        return "Very Good"
    elif nse > 0.65:
        return "Good"
    elif nse > 0.50:
        return "Satisfactory"
    else:
        return "Unsatisfactory"
