import logging 
import joblib
import pandas as pd
from housing import config

logger = logging.getLogger(__name__)

def load_model():
    if not config.PRODUCTION_MODEL_PATH.exists():
       raise FileNotFoundError(
           f"No production model at {config.PRODUCTION_MODEL_PATH}. Run `python -m housing.ml.train` first."
       )
    return joblib.load(config.PRODUCTION_MODEL_PATH)

def predict(records: list[dict]) -> list[float]:
    model = load_model()
    df = pd.DataFrame(records)
    preds = model.predict(df)
    return preds.tolist()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sample_houses = [
        {
            "area_sqft": 1400, "bedrooms": 3, "bathrooms": 2,
            "age_years": 5, "distance_to_city_km": 6.2, "location_tier": "Tier2",
        },
        {
            "area_sqft": 2500, "bedrooms": 4, "bathrooms": 3,
            "age_years": 1, "distance_to_city_km": 2.0, "location_tier": "Tier1",
        },
        {
            "area_sqft": 800, "bedrooms": 1, "bathrooms": 1,
            "age_years": 25, "distance_to_city_km": 30.0, "location_tier": "Tier3",
        },
    ]

    predictions = predict(sample_houses)
    for house,price in zip(sample_houses,predictions):
        logger.info("Predicted price for %s: ₹%.0f", house, price)