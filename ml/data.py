import logging
import pandas as pd
from database.db_utils import get_db_connection, execute_sql_query
from config.settings import DATA_CONFIG

logger = logging.getLogger(__name__)


def get_data_for_train() -> pd.DataFrame:
    """Fetch training data from database."""
    limit = DATA_CONFIG["training_limit"]
    decimal_columns = DATA_CONFIG["decimal_columns"]

    sql_query = f"""
        SELECT RC, RL, LP, LD, DateTime FROM (
            SELECT
                curah_hujan_cendono AS RC,
                curah_hujan_lawang AS RL,
                level_muka_air_purwodadi AS LP,
                level_muka_air_dhompo AS LD,
                tanggal AS DateTime
            FROM awlr_arr_per_jam
            ORDER BY tanggal DESC
            LIMIT {limit}
        ) AS latest_data
        ORDER BY latest_data.DateTime ASC
    """

    try:
        with get_db_connection() as connection:
            result_data, result_column = execute_sql_query(connection, sql_query)

            df = pd.DataFrame(result_data, columns=result_column)
            df[decimal_columns] = df[decimal_columns].astype(float)
            df.drop('DateTime', axis=1, inplace=True)

            logger.info(f"Loaded {len(df)} rows for training")
            return df
    except Exception as e:
        logger.exception(f"Error fetching training data: {e}")
        return None


def get_latest_rows() -> pd.DataFrame:
    """Fetch latest rows for prediction."""
    limit = DATA_CONFIG["prediction_limit"]
    decimal_columns = DATA_CONFIG["decimal_columns"]

    sql_query = f"""
        SELECT RC, RL, LP, LD, DateTime FROM (
            SELECT
                curah_hujan_cendono AS RC,
                curah_hujan_lawang AS RL,
                level_muka_air_purwodadi AS LP,
                level_muka_air_dhompo AS LD,
                tanggal AS DateTime
            FROM awlr_arr_per_jam
            ORDER BY tanggal DESC
            LIMIT {limit}
        ) AS latest_data
        ORDER BY latest_data.DateTime ASC
    """

    try:
        with get_db_connection() as connection:
            result_data, result_column = execute_sql_query(connection, sql_query)

            df = pd.DataFrame(result_data, columns=result_column)
            df[decimal_columns] = df[decimal_columns].astype(float)
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

            logger.info(f"Loaded {len(df)} rows for prediction")
            return df
    except Exception as e:
        logger.exception(f"Error fetching latest data: {e}")
        return None
