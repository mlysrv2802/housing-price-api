import numpy as np
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from housing import config
from housing.data.data_loader import load_raw_data

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
        ("categorical",categorical_pipeline,config.CATEGORICAL_FEATURES)
    ])

def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse":float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true,y_pred)),
        "r2_score": float(r2_score(y_true,y_pred)),
       }

def main() -> None:
    df = load_raw_data()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED )

    candidates = {
        "LinearRegression": LinearRegression(),
        "Lasso (alpha=1.0)": Lasso(alpha=1.0, random_state=config.RANDOM_SEED),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=config.RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,random_state=config.RANDOM_SEED
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model)
        ])

        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test.values, y_pred)
        metrics["model"] = name 
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    results_df  = pd.DataFrame(results).set_index("model")
    results_df = results_df.sort_values("r2_score", ascending=False)

    print("=" * 60)
    print("MODEL COMPARISON (sorted by R^2, best first)")
    print("=" * 60)
    print(results_df.to_string())
    print()

    best_model_name = results_df.index[0]
    print(f"Best model on this data: {best_model_name}\n")

    print("lasso feature shrink print")
    lasso_pipeline = fitted_pipelines["Lasso (alpha=1.0)"]
    feature_names = lasso_pipeline.named_steps["preprocessor"].get_feature_names_out()
    lasso_coefs = pd.Series(
        lasso_pipeline.named_steps["regressor"].coef_, index=feature_names
    )
    print("Lasso coefficients (features it shrank toward/to zero are")
    print("effectively 'deselected'):")
    print(lasso_coefs.sort_values(key=abs, ascending=False).to_string())
    

if __name__ == "__main__":
    main()