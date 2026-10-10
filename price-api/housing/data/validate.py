import pandas as pd
import logging

from housing.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


def validate_and_clean(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    original_count = len(df)
    expected_columns = set(NUMERIC_FEATURES) | set(
        CATEGORICAL_FEATURES) | {TARGET_COLUMN}

    # --- 1. Unexpected extra fields (schema drift) ------------------------
    extra_columns = set(df.columns) - expected_columns
    if extra_columns:
        logger.info(
            "Dropping unexpected columns, not in schema, %s", extra_columns)
        df = df.drop(columns=list(extra_columns))

    # --- 2. Missing columns entirely ---------------------------------------
    missing_entirely = expected_columns - set(df.columns)
    if missing_entirely:
        raise ValueError(f"data is missing col entirely, {missing_entirely}")

    # --- 3. Numbers sent as strings -----------------------------------------
    numeric_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    for col in numeric_columns:
        before_numeric = df[col].apply(
            lambda v: isinstance(v, (int, float))).sum()
        # "N/A", "garbage_val" , "None", "" -> NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_numeric = df[col].notna().sum()
        if before_numeric != len(df):
            logger.info(
                "Column '%s': coerced string/garbage values to numeric (or NaN)", col)
            logger.info("'%s': %d numeric values found ",
                         col, after_numeric)

    # --- 4. Inconsistent categorical casing/formatting ---------------------
    # "tier1", "TIER2", "Tier 3" all need to normalize to "Tier1"/"Tier2"/"Tier3"
    before_values = df["location_tier"].copy()
    cleaned_series = df["location_tier"].astype(
        str).str.replace(" ", "").str.lower()
    tier_map = {"tier1": "Tier1", "tier2": "Tier2", "tier3": "Tier3"}
    mapped_series = cleaned_series.map(tier_map)
    df["location_tier"] = mapped_series.fillna(df["location_tier"])
    normalized_col = (before_values != df["location_tier"]).sum()
    if normalized_col:
        logger.info(
            "Normalized casing/formatting on %d location_tier values", normalized_col)

    # --- 5. Drop rows with missing target (unusable for training) ----------
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    if len(df) < before:
        logger.warning(
            "Dropped %d rows with missing/unparseable price", before - len(df))

    # --- 6. Drop rows with missing critical features ------------------------
    before = len(df)
    # area is too important to impute at ingestion time
    df = df.dropna(subset=["area_sqft"])
    if len(df) < before:
        logger.warning("Dropped %d rows with missing area_sqft",
                       before - len(df))

    # --- 7. Outlier / sanity bounds ------------------------------------------
    before = len(df)
    df = df[df["area_sqft"].between(100, 20000)]
    df = df[df["bedrooms"].between(0, 20) | df["bedrooms"].isna()]
    df = df[df["age_years"].between(0, 300) | df["age_years"].isna()]
    if len(df) < before:
        logger.warning(
            "Dropped %d rows with out-of-range outlier values", before - len(df))

    # --- 8. Unrecognized categorical values (after normalization) -----------
    before = len(df)
    df = df[df["location_tier"].isin(["Tier1", "Tier2", "Tier3"])]
    if len(df) < before:
        logger.warning(
            "Dropped %d rows with unrecognized location_tier values", before - len(df))

    # --- 9. Exact duplicate records ------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.warning("Dropped %d exact duplicate records", before - len(df))

    logger.info(
        "Validation: %d raw records in -> %d clean records out (%.0f%% kept)",
        original_count, len(df), 100 * len(df) /
        original_count if original_count else 0,
    )

    return df
