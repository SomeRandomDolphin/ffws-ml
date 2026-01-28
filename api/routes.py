import logging
from flask import Blueprint, jsonify
from datetime import timedelta
import pandas as pd

from ml.models import get_model
from ml.data import get_latest_rows
from config.settings import (
    get_n_steps_in_for_model,
    get_n_steps_out_for_model,
    ACTIVE_MODELS
)

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)


@api_bp.route("/")
def home():
    """Home endpoint."""
    return "<h1>Flood Forecasting Warning System</h1>"


@api_bp.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "OK"})


@api_bp.route("/api/models")
def list_models():
    """List available models."""
    return jsonify({
        "active_models": ACTIVE_MODELS,
        "status": "OK"
    })


@api_bp.route("/api/predict", methods=['POST'])
def api_predict():
    """
    Make flood predictions using active models.

    Returns predictions for water levels at Dhompo and Purwodadi.
    """
    try:
        model_dict = {item: {'predictions': {}} for item in ACTIVE_MODELS}

        for model_name in ACTIVE_MODELS:
            select_model = get_model(model_name)
            if select_model is None:
                logger.error(f"Model '{model_name}' not found")
                continue

            n_steps_in = get_n_steps_in_for_model(model_name)
            n_steps_out = get_n_steps_out_for_model(model_name)

            # Fetch input data
            input_data = get_latest_rows()
            if input_data is None or input_data.empty:
                logger.error("No data available for prediction")
                return jsonify({"error": "No data available for prediction"}), 400

            if len(input_data) < n_steps_in:
                logger.error(f"Insufficient data: need {n_steps_in} rows, got {len(input_data)}")
                return jsonify({"error": "Insufficient data for prediction"}), 400

            input_data = input_data.tail(n_steps_in)

            # Get timestamp
            date = input_data.tail(1).iloc[0]['DateTime']
            predicted_from_time = date.strftime("%Y-%m-%d %H:%M:%S")

            # Preprocess and predict
            preprocessed_data = select_model.preprocess_data(
                pd.DataFrame(input_data), n_steps_in
            )
            prediction = select_model.model.predict(
                preprocessed_data, select_model.model.y_scaler, n_steps_out
            )

            # Format predictions
            prediction_values = prediction.values.flatten()
            prediction_dict = {
                (date + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M:%S"): {
                    'value': float(value)
                } for i, value in enumerate(prediction_values)
            }

            predicted_until_time = (date + timedelta(hours=n_steps_out)).strftime("%Y-%m-%d %H:%M:%S")

            model_dict[model_name]['predicted_from_time'] = predicted_from_time
            model_dict[model_name]['predicted_until_time'] = predicted_until_time
            model_dict[model_name]['predictions'] = prediction_dict

        return jsonify(model_dict)

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500
