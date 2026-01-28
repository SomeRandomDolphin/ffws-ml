import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

from config.settings import DATABASE_URL, DATABASE_CONFIG

logger = logging.getLogger(__name__)

# Create engine with connection pooling (singleton)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=DATABASE_CONFIG["pool_size"],
            max_overflow=DATABASE_CONFIG["max_overflow"],
            pool_timeout=DATABASE_CONFIG["pool_timeout"],
            pool_recycle=DATABASE_CONFIG["pool_recycle"]
        )
    return _engine


def connect_db():
    engine = get_engine()
    connection = engine.connect()
    return connection


@contextmanager
def get_db_connection():
    """Context manager for database connections - auto close on exit"""
    connection = None
    try:
        connection = connect_db()
        yield connection
    finally:
        if connection:
            close_db_connection(connection)


def execute_sql_query(connection, sql_query):
    try:
        result = connection.execute(text(sql_query))
        return result.fetchall(), result.keys()
    except SQLAlchemyError as e:
        logger.error(f"Database query error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}")
        raise


def close_db_connection(connection):
    try:
        connection.close()
    except Exception as e:
        logger.warning(f"Error closing connection: {e}")
