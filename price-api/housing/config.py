from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = BASE_DIR.parent / "price-storage"

DATA_DIR = STORAGE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "housing.csv"

MODEL_DIR = STORAGE_DIR / "model"
VERSION_DIR = MODEL_DIR / "versions"
PRODUCTION_MODEL_PATH = MODEL_DIR / "production_model.joblib"
PRODUCTION_METRICS_PATH = MODEL_DIR / "production_metrics.json"

OUTPUT_DIR = STORAGE_DIR / "outputs"
IMG_DIR = OUTPUT_DIR / "img"

MAX_ACCEPTABLE_R2_DROP = 0.02

LOG_DIR = STORAGE_DIR / "logs"
LOG_PATH = LOG_DIR / "training.log"

TARGET_COLUMN = "price"

NUMERIC_FEATURES = ["area_sqft","bedrooms",
                    "bathrooms","age_years","distance_to_city_km"]

CATEGORICAL_FEATURES = ["location_tier"]

TEST_SIZE = 0.2
RANDOM_SEED = 42

for _dir in (VERSION_DIR,DATA_DIR,MODEL_DIR,LOG_DIR,IMG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

#print("---------------------------------------------------------------------------------")

#print("RAW_DATA_PATH:",RAW_DATA_PATH)
#print("MODEL>VERSIONS_DIR:",VERSION_DIR)
#print("PRODUCTION_MODEL_PATH:",PRODUCTION_MODEL_PATH)
#print("PRODUCTION_METRICS_PATH:",PRODUCTION_METRICS_PATH)
#print("TEST_SIZE:",TEST_SIZE)
#print("MAX_ACCEPTABLE_R2_DROP:",MAX_ACCEPTABLE_R2_DROP)
#print("LOG_PATH:",LOG_PATH)

#print("---------------------------------------------------------------------------------")



