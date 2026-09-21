"""Shared evaluation metrics for Dhompo and Surabaya water-level forecasting.

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
    """Compute RMSE, MAE, R², NSE, KGE, and PBIAS.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values.

    Returns
    -------
    dict with keys: RMSE, MAE, R2, NSE, KGE, PBIAS
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("-inf")

    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    true_mean = float(np.mean(y_true))
    if true_std > 0 and pred_std > 0 and true_mean != 0:
        correlation = float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
        variability_ratio = pred_std / true_std
        bias_ratio = float(np.mean(y_pred)) / true_mean
        kge = float(1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
        ))
    else:
        kge = float("nan")

    pbias = float(100.0 * np.sum(y_pred - y_true) / np.sum(y_true)) if np.sum(y_true) != 0 else float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "KGE": kge, "PBIAS": pbias}


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


def calc_probabilistic_metrics(
    y_true: np.ndarray,
    ensemble: np.ndarray,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> dict[str, float]:
    """Hitung coverage, lebar interval, interval score, dan empirical CRPS."""
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    members = np.asarray(ensemble, dtype=float)
    if members.ndim != 2 or members.shape[0] != len(observed):
        raise ValueError("ensemble harus berbentuk (n_observations, n_members).")
    if members.shape[1] < 2:
        raise ValueError("Minimal dua ensemble member diperlukan.")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Quantile interval harus memenuhi 0 < lower < upper < 1.")
    if not np.isfinite(observed).all() or not np.isfinite(members).all():
        raise ValueError("Observasi dan ensemble harus finite.")

    lower = np.quantile(members, lower_quantile, axis=1)
    upper = np.quantile(members, upper_quantile, axis=1)
    median = np.quantile(members, 0.5, axis=1)
    nominal_coverage = upper_quantile - lower_quantile
    alpha = 1.0 - nominal_coverage
    below_penalty = (2.0 / alpha) * (lower - observed) * (observed < lower)
    above_penalty = (2.0 / alpha) * (observed - upper) * (observed > upper)
    interval_score = upper - lower + below_penalty + above_penalty

    absolute_error = np.abs(members - observed[:, None]).mean(axis=1)
    pairwise_distance = np.abs(
        members[:, :, None] - members[:, None, :]
    ).mean(axis=(1, 2))
    crps = absolute_error - 0.5 * pairwise_distance

    return {
        "COVERAGE": float(np.mean((observed >= lower) & (observed <= upper))),
        "MEAN_WIDTH": float(np.mean(upper - lower)),
        "INTERVAL_SCORE": float(np.mean(interval_score)),
        "CRPS": float(np.mean(crps)),
        "MEDIAN_MAE": float(np.mean(np.abs(median - observed))),
    }
