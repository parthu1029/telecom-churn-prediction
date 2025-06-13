import pandas as pd

from utils.utils import setup_logger

logger = setup_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load telecom churn dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        logger (Logger): Configured logger instance.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data load: {e}")
        raise
