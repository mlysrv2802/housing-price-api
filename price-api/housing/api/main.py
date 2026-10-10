import logging
import threading
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from housing import config
from housing.domain.HouseFeatures import HouseFeatures
from housing.domain.PredictionResponse import PredictionResponse
from housing.domain.NewListingsResponse import NewListingsResponse
from housing.data.synthetic import messy_data_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app  = FastAPI(
    title="Housing Price Predictor API",
    description="Serves prediction from a trained linear regression pipeling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_mtime: float | None = None
_model_lock = threading.Lock()


def _load_model() -> None:
    """Load the production model from disk and remember the file's mtime."""
    global _model, _model_mtime
    mtime = config.PRODUCTION_MODEL_PATH.stat().st_mtime
    _model = joblib.load(config.PRODUCTION_MODEL_PATH)
    _model_mtime = mtime
    logger.info("Model loaded from %s", config.PRODUCTION_MODEL_PATH)


def _reload_if_stale() -> None:
    """The retrain job promotes models by replacing the file on disk, from a
    separate process. Reload when the file changed so we don't serve a stale
    model. A failed reload keeps the current model serving."""
    try:
        current_mtime = config.PRODUCTION_MODEL_PATH.stat().st_mtime
    except OSError:
        return
    if current_mtime == _model_mtime:
        return
    with _model_lock:
        if current_mtime == _model_mtime:  # another request already reloaded
            return
        try:
            _load_model()
            logger.info("Production model changed on disk -- reloaded.")
        except Exception:
            logger.exception("Reloading the production model failed -- still serving the previous one.")


@app.on_event("startup")
def load_model_on_startup() -> None:
    if not config.PRODUCTION_MODEL_PATH.exists():
        raise RuntimeError(
            f"No production model found at {config.PRODUCTION_MODEL_PATH}. Run `python -m housing.ml.train` first."
        )
    _load_model()

@app.get("/health")
def health_check() -> dict:
    """Liveness/readiness check — is the process up AND is the model loaded?"""
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(house: HouseFeatures) -> PredictionResponse:
    _reload_if_stale()
    model = _model  # local ref: a concurrent reload can't swap it mid-request
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([house.model_dump()])
    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        logger.exception("Prediction failed for input: %s", house)
        raise HTTPException(status_code=500, detail="Prediction Failed.") from exc
    return PredictionResponse(predicted_price=round(float(prediction),2))

@app.get("/new-listings", response_model=NewListingsResponse)
def get_new_listings(count: int = Query(100, ge=1, le=5000)) -> NewListingsResponse:
    listings = messy_data_generator.generate_messy_data(count)
    return NewListingsResponse(count=len(listings), listings=listings)


