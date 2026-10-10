"""
data_loader.py
---------------
Single responsibility: load raw data and do basic sanity checks.
Keeping this separate from training logic means you can swap the data
source (CSV -> SQL query -> API call) without touching train.py at all —
same idea as an interface/DAO layer in a Java backend.
"""

import logging
import pandas as pd
from housing.config import RAW_DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from housing.data.validate import validate_and_clean

logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No data found at {RAW_DATA_PATH}. Run `python -m housing.data.simulator.data_generator` first.")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info('Loaded %d rows, %d cols from %s',len(df),df.shape[1],RAW_DATA_PATH)

    clean_df = validate_and_clean(df.to_dict('records'))
    return clean_df

