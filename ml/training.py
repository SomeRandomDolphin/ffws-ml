import os
import logging
from keras import backend as K

from config.settings import TRAINING_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    """Custom RMSE loss/metric function."""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def train_model(model_time_series, x_data_scaled, n_features: int, model_name: str) -> str:
    """
    Train a time series model.

    Args:
        model_time_series: The Keras model to train
        x_data_scaled: Preprocessed training data
        n_features: Number of input features
        model_name: Name of the model (for saving)

    Returns:
        "successful" or "failed"
    """
    try:
        epochs = TRAINING_CONFIG["epochs"]
        batch_size = TRAINING_CONFIG["batch_size"]

        logger.info(f"Starting training for {model_name}: epochs={epochs}, batch_size={batch_size}")

        model_time_series.fit(
            x_data_scaled,
            batch_size=batch_size,
            epochs=epochs
        )

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
        model_time_series.save(model_path)
        logger.info(f"Model saved to {model_path}")

        return "successful"

    except ValueError as e:
        logger.error(f"Value error during training {model_name}: {e}")
        return "failed"
    except IOError as e:
        logger.error(f"IO error saving model {model_name}: {e}")
        return "failed"
    except Exception as e:
        logger.exception(f"Unexpected error during training {model_name}: {e}")
        return "failed"
