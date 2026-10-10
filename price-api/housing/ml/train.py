import json
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from housing import config
from housing.ml import model_registry
from housing.data.data_loader import load_raw_data


logger = logging.getLogger(__name__)

def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
         ("numeric",numeric_pipeline,config.NUMERIC_FEATURES),
         ("categorical",categorical_pipeline,config.CATEGORICAL_FEATURES),
    ])

    model = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("regressor",LinearRegression()),
    ])

    return model

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true,y_pred)))
    mae = float(mean_absolute_error(y_true,y_pred))
    r2  = float(r2_score(y_true,y_pred))
    return {"rmse": rmse, "mae":mae, "r2_score": r2}

def main() -> bool:
    df = load_raw_data()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=config.TEST_SIZE,random_state=config.RANDOM_SEED
    )

    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test.values,y_pred)
    logger.info("Test metrics: %s", metrics)

    version_id = model_registry.save_version(pipeline,metrics)
    promoted = model_registry.promote_if_better(version_id, metrics)

    if promoted:
       logger.info("Version %s is now the production model.", version_id)
    else:
       logger.warning(
           "Version %s was REJECTED -- production model unchanged. "
           "Run `python -m housing.ml.model_registry list` to review all versions.",
           version_id,
        )

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_
    coef_table = pd.Series(coefficients, index=feature_names).sort_values(
        key=abs, ascending=False
    )

    logger.info("Top feature coefficients:\n%s", coef_table.to_string())

    return promoted

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_PATH)],
    )
    main()
