import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": os.getenv("DB_PORT", "3306"),
    "name": os.getenv("DB_NAME", "flood_forecasting"),
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800,
}

DATABASE_URL = (
    f"mysql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['name']}"
)

# model configuration
MODEL_CONFIG = {
    "dhompo_gru": {"n_steps_in": 5, "n_steps_out": 5, "n_features": 4},
    "dhompo_lstm": {"n_steps_in": 5, "n_steps_out": 5, "n_features": 4},
    "dhompo_tcn": {"n_steps_in": 5, "n_steps_out": 5, "n_features": 4},
    "purwodadi_gru": {"n_steps_in": 3, "n_steps_out": 3, "n_features": 3},
    "purwodadi_lstm": {"n_steps_in": 3, "n_steps_out": 3, "n_features": 3},
    "purwodadi_tcn": {"n_steps_in": 3, "n_steps_out": 3, "n_features": 3},
}

ACTIVE_MODELS = [
    "dhompo_lstm",
    "purwodadi_gru",
]

# training configuration
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 64,
    "validation_split": 0.2,
}

# data configuration
DATA_CONFIG = {
    "training_limit": 720,
    "prediction_limit": 5,
    "decimal_columns": ["RC", "RL", "LP", "LD"],
}

# path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")


# helper functions 
def get_n_steps_in_for_model(model_name: str) -> int:
    config = MODEL_CONFIG.get(model_name)
    if config:
        return config["n_steps_in"]
    logger.warning(f"Model '{model_name}' not found in config")
    return 0


def get_n_steps_out_for_model(model_name: str) -> int:
    config = MODEL_CONFIG.get(model_name)
    if config:
        return config["n_steps_out"]
    logger.warning(f"Model '{model_name}' not found in config")
    return 0


def get_n_features_for_model(model_name: str) -> int:
    config = MODEL_CONFIG.get(model_name)
    if config:
        return config["n_features"]
    logger.warning(f"Model '{model_name}' not found in config")
    return 0
