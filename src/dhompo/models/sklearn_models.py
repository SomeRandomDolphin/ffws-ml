from __future__ import annotations

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

HORIZONS: list[int] = [1, 2, 3, 4, 5]
HORIZON_STEPS: dict[int, int] = {h: h * 2 for h in HORIZONS}  # 30-min steps


def get_model_definitions(params: dict | None = None) -> dict[str, tuple]:
    """Return model definitions as {name: (estimator, use_scaled)}.

    Parameters
    ----------
    params:
        Optional hyperparameter overrides loaded from configs/sklearn_model.yaml.
        Keys should match model names below.

    Returns
    -------
    dict
        Mapping from model name to (sklearn estimator, bool use_scaled).
        use_scaled=True → StandardScaler applied; False → raw features.
    """
    if params is None:
        params = {}

    rf_params = params.get("random_forest", {})
    gb_params = params.get("gradient_boosting", {})
    xgb_params = params.get("xgboost", {})
    ridge_params = params.get("ridge", {})
    lasso_params = params.get("lasso", {})

    return {
        "Linear Regression": (LinearRegression(), True),
        "Ridge": (
            Ridge(alpha=ridge_params.get("alpha", 1.0)),
            True,
        ),
        "Lasso": (
            Lasso(
                alpha=lasso_params.get("alpha", 0.01),
                max_iter=lasso_params.get("max_iter", 10000),
            ),
            True,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=rf_params.get("n_estimators", 200),
                max_depth=rf_params.get("max_depth", 15),
                random_state=rf_params.get("random_state", 42),
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=gb_params.get("n_estimators", 200),
                max_depth=gb_params.get("max_depth", 5),
                learning_rate=gb_params.get("learning_rate", 0.1),
                random_state=gb_params.get("random_state", 42),
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=xgb_params.get("n_estimators", 200),
                max_depth=xgb_params.get("max_depth", 5),
                learning_rate=xgb_params.get("learning_rate", 0.1),
                random_state=xgb_params.get("random_state", 42),
                n_jobs=-1,
                verbosity=0,
            ),
            False,
        ),
    }
