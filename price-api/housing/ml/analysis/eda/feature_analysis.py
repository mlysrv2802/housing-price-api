import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from housing import config
from housing.data.data_loader import load_raw_data

def basic_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. BASIC OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} cols\n")
    print("Missing values per column:")
    missing = df.isna().sum()
    missing_only = missing[missing>0]
    print(missing_only if not missing_only.empty else " (none)")
    print()
    print("Numeric feature summary:")
    print(df.describe().T[["mean","std","min","max"]])



def correlation_with_target(df:pd.DataFrame) -> None:
    print("=" * 60)
    print("2. Correlation with target value")
    print("=" * 60)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    corr = corr.sort_values(key=abs,ascending=False)
    print(corr)
   

def _compute_variance_inflation_factor(numeric_df: pd.DataFrame) -> pd.DataFrame:
    result = []
    columns = numeric_df.columns
    for col in columns:
        y = numeric_df[col].values
        X = numeric_df.drop(columns=[col]).values
        r2 = LinearRegression().fit(X,y).score(X,y)
        vif = 1.0/(1.0-r2) if r2 < 0.999 else float("inf")
        result.append({"feature":col, "VIF":vif})
    return pd.DataFrame(result)


def multicollinearity_check(df:pd.DataFrame) -> None:
    print("=" * 60)
    print("3. MULTICOLLINEARITY CHECK (VIF)")
    print("=" * 60)
    numeric_df = df[config.NUMERIC_FEATURES].dropna()
    vif_data = _compute_variance_inflation_factor(numeric_df)
    print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))
    print()

def feature_importance_via_forest(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("4. FEATURE IMPORTANCE (RandomForest -- model-based ranking)")
    print("=" * 60)
    print("A quick, practical way to rank ALL features (numeric AND")
    print("categorical) together by how much they actually reduce")
    print("prediction error -- more reliable than raw correlation because")
    print("it captures non-linear effects and feature interactions.\n")

    df_clean = df.dropna()
    X_numeric = df_clean[config.NUMERIC_FEATURES].values
   
    encoder = OneHotEncoder(sparse_output=False)
    X_categorical = encoder.fit_transform(df_clean[config.CATEGORICAL_FEATURES])
    categorical_names = encoder.get_feature_names_out(config.CATEGORICAL_FEATURES)

    X = np.hstack([X_numeric,X_categorical])
    feature_names = config.NUMERIC_FEATURES + list(categorical_names)
    y = df_clean[config.TARGET_COLUMN].values

    forest = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_SEED)
    forest.fit(X,y)

    importance = pd.Series(forest.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    print(importance.to_string())
    print()

if __name__ == "__main__":
    df = load_raw_data()
    basic_overview(df)
    correlation_with_target(df)
    multicollinearity_check(df)
    feature_importance_via_forest(df)

