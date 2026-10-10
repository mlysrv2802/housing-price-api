# Housing Price Predictor

An end-to-end ML project: generate (messy) housing data, clean it, analyse it, train a model, serve predictions through an API, and keep the model fresh with a scheduled fetch → retrain → promote loop.

All commands are run from the `price-api` folder, using module syntax so that `from housing import config` resolves:

```bash
cd price-api
pip install -r requirements.txt
python -m housing.data.synthetic.data_generator
```

## End-to-end flow

```text
messy_data_generator → GET /new-listings
        ↓
fetch_new_data → validate_and_clean → append to housing.csv
        ↓
train (impute → scale/one-hot → LinearRegression) → save version
        ↓
registry quality gate (R² drop ≤ 0.02) → promote production_model.joblib
        ↓
API detects the new file, reloads, and serves /predict
        ↓
Angular UI sends house features → receives predicted price
```

## Retraining flow

The scheduler (`jobs/retrain_scheduler.py`) repeats this cycle every 5 minutes, and once immediately on start:

```text
Scheduler tick
        ↓
fetch_new_data → GET /new-listings (messy records)
        ↓            ✗ API down / timeout / HTTP error → log, keep current model, wait for next tick
validate_and_clean → append clean rows to housing.csv
        ↓            ✗ no new rows → skip retraining
train.main() → fit on the full, grown dataset → save as a new version
        ↓
model_registry.promote_if_better → compare R² with production
        ├── R² drop > 0.02 → REJECTED, production model unchanged
        └── otherwise → PROMOTED: production_model.joblib replaced atomically
                              ↓
                  API sees the changed file on its next /predict and hot-reloads it
```

Any exception inside a cycle is logged and swallowed, so one bad run never stops the scheduler. Details are in [section 4](#4-api-and-retraining-scheduler).

## Key insights

**Real-world data handling**
- Clean synthetic data generator
- Messy data simulator (7 types of real-world issues)
- 9-step validation pipeline with detailed logging

**Production ML pipeline**
- `ColumnTransformer` with parallel preprocessing
- Proper imputation strategies (median for numeric, most_frequent for categorical)
- `OneHotEncoder` with `handle_unknown="ignore"` for robustness

**Model management**
- Version tracking with timestamps
- Quality gate (R² drop ≤ 0.02)
- Atomic file operations (prevent corruption)
- Rollback capability

**Zero-downtime API**
- Hot-reload without restart
- File mtime checking
- Thread-safe model swapping
- Swagger/ReDoc docs

**Automated retraining**
- APScheduler fetch → retrain → promote loop
- Bootstrap for cold start
- Error recovery (doesn't crash on bad data)
- Graceful timeout handling

**Analysis suite**
- EDA, linearity checks, model comparison
- Hyperparameter tuning (GridSearch, RandomSearch)

## Project layout

```text
first-prj/
├── README.md
├── price-api/
│   └── housing/
│       ├── config.py                 # paths, feature lists, split/seed, quality-gate threshold
│       ├── data/
│       │   ├── synthetic/
│       │   │   ├── data_generator.py        # clean synthetic dataset (vectorised)
│       │   │   └── messy_data_generator.py  # deliberately messy "vendor feed"
│       │   ├── validate.py                  # cleaning of messy records
│       │   ├── data_loader.py               # load housing.csv + clean
│       │   └── fetch_new_data.py            # pull new listings from the API, append to CSV
│       ├── ml/
│       │   ├── analysis/                    # EDA, linearity check, model comparison, tuning
│       │   ├── train.py                     # final training pipeline
│       │   ├── predict.py                   # batch prediction from the production model
│       │   └── model_registry.py            # versioning, quality gate, promote, rollback
│       ├── api/main.py                      # FastAPI app
│       ├── jobs/retrain_scheduler.py        # APScheduler fetch → retrain → promote
│       └── domain/                          # pydantic request/response models
└── price-storage/                    # everything generated at runtime
    ├── data/housing.csv              # training data
    ├── model/                        # production_model.joblib, production_metrics.json, versions/
    ├── outputs/img/                  # plots from the analysis scripts
    └── logs/training.log
```

Features (see `config.py`): numeric `area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km`; categorical `location_tier` (Tier1/Tier2/Tier3); target `price`.

---

## 1. Data generation and cleaning

### Clean data: `data/synthetic/data_generator.py`
Generates 2,000 rows with NumPy (vectorised, seeded with `RANDOM_SEED`). Price is a linear function of the features plus Gaussian noise, with a premium per location tier (Tier1 +40,000, Tier2 +15,000, Tier3 +0), so a linear model should be able to recover it. About 2% of `bathrooms` values are blanked out to simulate missing data. Writes `price-storage/data/housing.csv`.

```bash
python -m housing.data.synthetic.data_generator
```

### Messy data: `data/synthetic/messy_data_generator.py`
Simulates an unreliable vendor feed. It builds records one at a time in a Python loop (the variety of corruption is easier to express row by row) and injects problems at realistic rates:

| Problem | Approx. rate |
|---|---|
| A field missing entirely | 5% |
| A number sent as a string (`"3"`) | 5% |
| Inconsistent casing of `location_tier` (`tier1`, `TIER1`, `Tier 1`) | 5% |
| Extreme outlier (value × 1000 in area, bedrooms or age) | 3% |
| Price missing or garbage (`None`, `"N/A"`, `""`) | 2% |
| Exact duplicate of the previous record | 3% |
| Unexpected extra field (`listing_source`, schema drift) | occasional |

This generator backs the API's `/new-listings` endpoint.

### Cleaning: `data/validate.py` → `validate_and_clean(records)`
Takes a list of raw dicts and returns a clean DataFrame. Steps, in order:

1. Drop unexpected extra columns (schema drift).
2. Raise `ValueError` if an expected column is missing entirely.
3. Coerce numeric columns (including price) with `pd.to_numeric(errors="coerce")`, so strings like `"3"` become numbers and `"N/A"` becomes NaN.
4. Normalise `location_tier` (`"tier 1"`, `"TIER1"` → `"Tier1"`).
5. Drop rows with no price (unusable for training).
6. Drop rows with no `area_sqft` (too important to impute).
7. Drop out-of-range outliers (area 100–20,000, bedrooms 0–20, age 0–300).
8. Drop rows whose `location_tier` is still unrecognised.
9. Drop exact duplicates.

Other missing values (for example `bathrooms`) are kept and imputed later inside the model pipeline, so the imputer is fitted on training data only. Each step logs how many rows it changed or dropped.

`data_loader.load_raw_data()` reads `housing.csv` and runs it through `validate_and_clean`, so every script gets the same cleaned data.

---

## 2. Analysis (`ml/analysis/`)

These are exploration scripts. They don't change the production model; `train.py` holds the finalised pipeline.

| Script | Purpose | Run |
|---|---|---|
| `eda/feature_analysis.py` | Basic overview (shape, missing values, summary stats), correlation of each numeric feature with price, multicollinearity check via VIF, and RandomForest feature importance across numeric and categorical features. | `python -m housing.ml.analysis.eda.feature_analysis` |
| `check_linearity.py` | Checks whether a linear model is appropriate. Saves scatter plots of each feature against price (`feature_vs_target.png`) and residual diagnostics (`residual_diagnostics.png`: residuals vs predicted, residual histogram) to `price-storage/outputs/img/`. | `python -m housing.ml.analysis.check_linearity` |
| `evaluate/compare_models.py` | Trains LinearRegression, Lasso, Ridge and RandomForest on the same split and preprocessing. Prints RMSE, MAE and R² sorted by R², plus the Lasso coefficients to show which features it shrinks. | `python -m housing.ml.analysis.evaluate.compare_models` |
Outcome: the data is close to linear, so plain linear models do as well as or better than the tree model, and `train.py` uses `LinearRegression`.

### For learning / experimentation only

The scripts below are **not part of the price API**. Nothing in `train.py`, the registry, the API or the scheduler imports or uses them. They exist to practise hyperparameter tuning and see how the techniques compare.

| Script | Purpose | Run |
|---|---|---|
| `evaluate/grid_search_tune_hyperparam.py` | `GridSearchCV` (5-fold) over a log-scale alpha grid for Ridge and Lasso. Reports best params, CV RMSE, held-out test RMSE and R². | `python -m housing.ml.analysis.evaluate.grid_search_tune_hyperparam` |
| `evaluate/random_search_tune_hyperparam.py` | `RandomizedSearchCV` (30 iterations, 5-fold) over RandomForest `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`. | `python -m housing.ml.analysis.evaluate.random_search_tune_hyperparam` |
| `scratch_gradient_descent.py` | Standalone gradient descent exercise. | `python -m housing.ml.analysis.scratch_gradient_descent` |

---

## 3. ML: train, predict, registry

### Training: `ml/train.py`
`build_pipeline()` is the finalised model:

```text
numeric:     median impute → StandardScaler
categorical: most-frequent impute → OneHotEncoder(handle_unknown="ignore")
              → LinearRegression
```

`main()` loads and cleans the data, splits it 80/20 (`TEST_SIZE`, `RANDOM_SEED`), fits the pipeline, computes RMSE, MAE and R² on the test set, then saves the result as a new version and asks the registry whether to promote it. It returns `True` if the model was promoted. It also logs the feature coefficients and writes to `price-storage/logs/training.log`.

```bash
python -m housing.ml.train
```

### Prediction: `ml/predict.py`
`predict(records)` loads the production model and returns a list of prices. Running it as a script predicts three sample houses.

```bash
python -m housing.ml.predict
```

### Registry: `ml/model_registry.py`
Models are stored on disk under `price-storage/model/`:

- `versions/model_<id>.joblib` and `versions/metrics_<id>.json`: every trained candidate, with a timestamp id.
- `production_model.joblib` and `production_metrics.json`: the model currently served.

Logic:

- `save_version(pipeline, metrics)` writes a new version and returns its id.
- `promote_if_better(version_id, metrics)` is the quality gate. If there is no production model, the candidate is promoted. Otherwise it is **rejected if R² drops by more than `MAX_ACCEPTABLE_R2_DROP` (0.02)** compared with production, and promoted if not.
- `promote(version_id)` copies the version into the production paths atomically (copy to a `.tmp` file, then `os.replace`), so the API never sees a half-written file.
- `list_versions()` and `rollback_to(version_id)` let you review versions and go back to an older one.

```bash
python -m housing.ml.model_registry list                    # production metrics + every version (id, R², RMSE), newest first
python -m housing.ml.model_registry rollback <version_id>   # make an older version the production model
```

---

## 4. API and retraining scheduler

### API: `api/main.py` (FastAPI)

```bash
uvicorn housing.api.main:app --port 8000
```

| URL | What it is |
|---|---|
| http://127.0.0.1:8000/docs | Swagger UI: browse and try every endpoint |
| http://127.0.0.1:8000/redoc | ReDoc documentation |
| http://127.0.0.1:8000/openapi.json | OpenAPI schema |
| http://127.0.0.1:8000/health | Health check |

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check; reports whether the model is loaded. |
| `POST /predict` | Body is a `HouseFeatures` JSON; returns `{"predicted_price": ...}`. |
| `GET /new-listings?count=100` | **Messy data generation.** Returns `count` records from `messy_data_generator` (1–5000), standing in for a third-party listings feed. |

Notes:
- The production model is loaded on startup (the app refuses to start if none exists; run `train` or the scheduler's bootstrap first).
- The scheduler is a separate process that replaces the production model file. On every `/predict` request, `_reload_if_stale()` compares the production model file's modification time with the one loaded in memory. If the file changed (a new model was promoted), the API reloads it on that request, under a lock so concurrent requests don't reload twice. No API restart is needed, and until the next request arrives the old model stays in memory. If a reload fails, the previous model keeps serving.
- CORS allows `http://localhost:4200` (the Angular UI).

### Retraining scheduler: `jobs/retrain_scheduler.py` (APScheduler)

```bash
python -m housing.jobs.retrain_scheduler
```

1. **Bootstrap** (`bootstrap_if_needed`): if there is no production model, generate the clean dataset if needed and run `train.main()`. This doesn't need the API.
2. **Every 5 minutes** (and once immediately on start), `fetch_and_retrain_job` runs:
   1. **Fetch**: `fetch_new_data` calls `GET /new-listings` (base URL from `API_BASE_URL`, default `http://127.0.0.1:8000`, 10 s timeout), cleans the records with `validate_and_clean`, and appends them to `housing.csv`. On a connection error, timeout or HTTP error it logs the problem and exits non-zero, and the scheduler waits for the next tick.
   2. **Skip** retraining if no new rows were added.
   3. **Retrain**: `train.main()` fits on the grown dataset and saves a new version.
   4. **Promote**: the registry quality gate promotes it or rejects it (R² dropped too much). The job logs the R² change or that production is unchanged.
   5. Any exception in a cycle is logged and swallowed, so one bad run never stops the scheduler.

The API must be running for the fetch step to work, so start it first.

### Typical local run

```bash
python -m housing.data.synthetic.data_generator     # initial data
python -m housing.ml.train                          # first production model
uvicorn housing.api.main:app --port 8000            # terminal 1: API
python -m housing.jobs.retrain_scheduler            # terminal 2: scheduler
python -m housing.ml.model_registry list            # terminal 3: see versions and which R² is in production
```

### Docker

```bash
docker compose up --build
```

Starts bootstrap (runs once), then the API on port 8000, then the scheduler. All three share `../price-storage`, mounted at `/price-storage`.
