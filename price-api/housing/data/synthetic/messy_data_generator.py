import numpy as np
import random

from housing.config import NUMERIC_FEATURES

# Python loops execute row-by-row,
# whereas vectorized NumPy calls run optimized C loops across the entire array at once.
# using loop for creating diff sort of  messiness in random rows
# Vectorized approach in generate_data is faster,
# but we need to create different type of messiness

def generate_messy_data(n_rows: int = 100) -> list:
    """Generates messy data to simulate real data-- INTENTIONALLY messy,
    Messiness deliberately injected (with real-world likelihoods):
      - ~5% of records missing a random field entirely
      - ~5% send a number as a STRING ("3" instead of 3)
      - ~5% have inconsistent categorical casing ("tier1", "TIER2")
      - ~3% are exact duplicates of the previous record (vendor re-sent it)
      - ~3% contain an extreme outlier value (bad sensor/data-entry error)
      - ~2% use "N/A" or null for price
      - occasionally an unexpected EXTRA field appears (schema drift)
    """
    listings = []
    previous_record = None
    for _ in range(n_rows):
        record = create_row(n_rows)
        inject_messiness(record)
        listings.append(record)
        add_duplicate(listings, previous_record)
        previous_record = record

    return listings


def add_duplicate(listings: list, previous_record: dict | None) -> None:
    """Occasionally resend the previous record -- vendor feeds duplicate
    entries across polls more often than you'd expect. Mutates listings."""
    if previous_record is not None and random.random() < 0.03:
        listings.append(dict(previous_record))


def create_row(n_rows: int) -> dict:
    area = float(np.clip(random.gauss(1500, 500), 300, 5000))
    bedrooms = random.randint(1, 5)
    bathrooms = float(random.randint(1, 3))
    age = random.randint(0, 40)
    distance = float(np.clip(random.expovariate(1/8), 0.5, 60))
    tier = random.choices(["Tier1", "Tier2", "Tier3"],
                          weights=[0.3, 0.45, 0.25])[0]
    tier_premium = {"Tier1": 40000, "Tier2": 15000, "Tier3": 0}[tier]
    noise = random.gauss(0,15000)
    price = (
        50 * area + 8000 * bedrooms + 5000 * bathrooms - 800 * age
        - 1200 * distance + tier_premium + 20000 + noise
    )

    price = max(price, 20000)

    record = {
        "area_sqft": round(area, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "distance_to_city_km": round(distance, 2),
        "location_tier": tier,
        "price": price
    }

    return record

# --- Inject real-world messiness ------------------------------
def inject_messiness(record: dict) -> None:
    """Mutates record in place."""
    roll = random.random()

    if roll < 0.05:
    # Missing field entirely -- vendor's system dropped it
        field_to_drop = random.choice(list(record.keys()))
        del record[field_to_drop]

    elif roll < 0.10:
    # Number sent as a string -- extremely common with real APIs
        field_numeric = random.choice(NUMERIC_FEATURES)
        record[field_numeric] = str(record[field_numeric])

    elif roll < 0.15:
    # Inconsistent categorical casing/formatting
    # "tier1", "TIER1", "Tier 1"
        tier = record["location_tier"]
        record["location_tier"] = random.choice(
            [tier.lower(), tier.upper(), f"{tier[:4]} {tier[4:]}"]
        )

    elif roll <0.18:
    # Extreme outlier / bad sensor value
        outlier_field = random.choice(["area_sqft","bedrooms","age_years"])
        record[outlier_field] = record[outlier_field] * 1000

    elif roll < 0.20:
    # Missing/garbage price -- unusable for training
        record["price"] = random.choice([None, "N/A", ""])

    elif random.random() < 0.03:
    # Unexpected extra field -- schema drift, e.g. vendor added a
    # new attribute you were never told about
        record["listing_source"] = random.choice(["partner_feed_v2", "walk_in"])

def main() -> None:
    listings = generate_messy_data()
    print(f"Generated {len(listings)} using util for messy data generator")


if __name__ == "__main__":
    main()
