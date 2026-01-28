import logging
from TimeSeriesClass.train_test_split import train_test_split_data
from TimeSeriesClass.scaling import scaling_data
from ml.training import train_model
from ml.data import get_data_for_train
from ml.models import get_model
from config.settings import (
    get_n_steps_in_for_model,
    get_n_steps_out_for_model,
    MODEL_CONFIG
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def training_init(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah):
    (x_data_scaled, n_features) = scaling_data(x_scaler, y_scaler, n_steps_in, n_steps_out, merge_df)
    result = train_model(model_time_series, x_data_scaled, n_features, model_daerah)

    return result


def process_model(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah):
    result = training_init(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah)
    return model_daerah, result


def start():
    arr_model_daerah = list(MODEL_CONFIG.keys())

    merge_df = get_data_for_train()
    if merge_df is None or merge_df.empty:
        logger.error("No data available for training")
        return

    results = []
    for model_daerah in arr_model_daerah:
        logger.info(f"Training model: {model_daerah}")
        n_steps_in = get_n_steps_in_for_model(model_daerah)
        n_steps_out = get_n_steps_out_for_model(model_daerah)
        select_model = get_model(model_daerah)

        if select_model is None:
            logger.warning(f"Model '{model_daerah}' not found, skipping")
            continue

        y_scaler = select_model.model.y_scaler
        x_scaler = select_model.model.x_scaler
        model_time_series = select_model.model
        logger.info(f"Model architecture: {model_time_series.model}")

        result = process_model(x_scaler, y_scaler, n_steps_in, n_steps_out, model_time_series, merge_df, model_daerah)
        results.append(result)

    model_dict = {model: result for model, result in results}
    logger.info(f"Training results: {model_dict}")


if __name__ == '__main__':
    start()
