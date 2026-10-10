"""
fetch_new_data.py
-------------------
Pulls fresh data from a LIVE source (here: our own /new-listings endpoint,
standing in for a real vendor/internal API) and appends it to housing.csv.

This is the piece we'd schedule to run periodically in production (cron,
Airflow, a Kubernetes CronJob) -- NOT something a person runs by hand
every time. Run it manually for now to see it work.

Real-world differences to keep in mind (noted inline below):
  - A real third-party API needs an API key/auth header
  - A real API can rate-limit you (HTTP 429) -- handle it, don't hammer it
  - A real API call can time out or fail -- never let that crash a
    scheduled job silently; log it and exit non-zero so monitoring notices

Run: python -m housing.data.fetch_new_data
"""

import logging
import sys
import pandas as pd
import requests
import os
from housing.config import RAW_DATA_PATH
from housing.data.validate import validate_and_clean



logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 10


def fetch_new_listings(count: int = 100) -> list[dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/new-listings",
                                params={"count": count},
                                timeout=REQUEST_TIMEOUT_SECONDS,
                                )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Request to %s timed out after %ss",
                     API_BASE_URL, REQUEST_TIMEOUT_SECONDS)
        raise
    except requests.exceptions.ConnectionError:
        logger.error(
            "Could not connect to %s -- is the API running?", API_BASE_URL)
        raise
    except requests.exceptions.HTTPError as exc:
        # A real API returning 429 (rate limited) would need a backoff-and-retry
        # here instead of just failing. Keeping it simple for now.
        logger.error("API returned an error status: %s", exc)
        raise

    payload = response.json()
    listings = payload.get("listings", [])
    logger.info("Fetched %d new listings from %s", len(listings), API_BASE_URL)
    return listings

def append_to_dataset(new_df : pd.DataFrame) -> int:
    if new_df.empty:
        logger.info("No valid new rows to append.")
        return 0
    existing_df = pd.read_csv(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else pd.DataFrame()
    combined_df = pd.concat([existing_df,new_df], ignore_index = True)
    combined_df.to_csv(RAW_DATA_PATH, index = False)
    logger.info(
        "Appended %d new rows. Dataset size: %d -> %d",
        len(new_df), len(existing_df), len(combined_df),
    )
    return len(new_df)

def main() -> int:
    try:
        raw_listings = fetch_new_listings(count=1000)
    except requests.exceptions.RequestException:
        logger.error("Fetch failed -- exiting without modifying the dataset.")
        # non-zero return so a scheduler (cron/Airflow) flags this run as failed
        sys.exit(1)

    if not raw_listings:
        logger.info("API returned zero new listings this run.")
        return 0

    clean_df = validate_and_clean(raw_listings)
    return append_to_dataset(clean_df)
    


if __name__ == "__main__":
    main()