import os
import pickle
import logging
import pandas as pd
from functools import lru_cache
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GRU
from keras import backend as K

from config.settings import MODEL_DIR, SCALER_DIR

logger = logging.getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    """Custom RMSE metric for model loading."""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


@lru_cache(maxsize=10)
def load_scaler(path_to_scaler: str):
    """Load and cache scaler from pickle file."""
    logger.info(f"Loading scaler from {path_to_scaler}")
    with open(path_to_scaler, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=10)
def cache_load_model(path_to_model: str):
    """Load and cache Keras model."""
    logger.info(f"Loading model from {path_to_model}")
    return load_model(
        path_to_model,
        custom_objects={
            'root_mean_squared_error': root_mean_squared_error,
            'LSTM': LSTM,
            'GRU': GRU
        }
    )


class TimeSeriesModel:
    """Wrapper class for time series prediction model."""

    def __init__(self, path_to_model: str, x_scaler_path: str, y_scaler_path: str):
        self.model = cache_load_model(path_to_model)
        self.x_scaler = load_scaler(x_scaler_path)
        self.y_scaler = load_scaler(y_scaler_path)

    def predict(self, scaled_data, y_scaler, n_steps_out: int) -> pd.DataFrame:
        """Make predictions and inverse transform."""
        raw_predictions = self.model.predict(scaled_data, verbose=0)
        inverse_scaled = y_scaler.inverse_transform(
            raw_predictions.reshape(-1, n_steps_out)
        )
        return pd.DataFrame(inverse_scaled)


# =============================================================================
# MODEL REGISTRY - Lazy loading
# =============================================================================
_model_registry = {}


def _get_or_create_preprocessor(model_name: str):
    """Lazy load preprocessor on first access."""
    from ml.preprocessing import DhompoDataPreprocessor, PurwodadiDataPreprocessor

    if model_name not in _model_registry:
        if model_name.startswith("dhompo"):
            _model_registry[model_name] = DhompoDataPreprocessor(
                model_name, MODEL_DIR, SCALER_DIR
            )
        elif model_name.startswith("purwodadi"):
            _model_registry[model_name] = PurwodadiDataPreprocessor(
                model_name, MODEL_DIR, SCALER_DIR
            )
        else:
            logger.warning(f"Unknown model type: {model_name}")
            return None

    return _model_registry[model_name]


def get_model(model_name: str):
    """Get model preprocessor by name."""
    return _get_or_create_preprocessor(model_name)
