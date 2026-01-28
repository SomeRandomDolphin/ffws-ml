import os
import logging
import pandas as pd

from config.settings import get_n_features_for_model

logger = logging.getLogger(__name__)


class BaseDataPreprocessor:
    """Base class for data preprocessing."""

    def __init__(self, model_name: str, model_dir: str, scaler_dir: str):
        from ml.models import TimeSeriesModel

        self.model_name = model_name
        self.model = TimeSeriesModel(
            os.path.join(model_dir, f"{model_name}.h5"),
            os.path.join(scaler_dir, f"{model_name}_x_scaler.pkl"),
            os.path.join(scaler_dir, f"{model_name}_y_scaler.pkl")
        )

    def preprocess_data(self, df_test: pd.DataFrame, n_steps_in: int):
        raise NotImplementedError("Subclasses must implement preprocess_data")


class DhompoDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for Dhompo location models."""

    def preprocess_data(self, df_test: pd.DataFrame, n_steps_in: int):
        df = df_test.copy()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values(by='DateTime', ascending=True)
        df = df.drop(columns=['DateTime'])

        n_features = get_n_features_for_model(self.model_name)
        x_test_scaled = self.model.x_scaler.transform(df.to_numpy()).reshape(
            -1, n_steps_in, n_features
        )
        return x_test_scaled


class PurwodadiDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for Purwodadi location models."""

    def preprocess_data(self, df_test: pd.DataFrame, n_steps_in: int):
        df = df_test.copy()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values(by='DateTime', ascending=True)
        df = df.drop(columns=['DateTime', 'LD'])

        n_features = get_n_features_for_model(self.model_name)
        x_test_scaled = self.model.x_scaler.transform(df.to_numpy()).reshape(
            -1, n_steps_in, n_features
        )
        return x_test_scaled
