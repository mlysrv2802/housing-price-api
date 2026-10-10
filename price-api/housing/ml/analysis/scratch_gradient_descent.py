import numpy as np
import pandas as pd
from housing import config


df=pd.read_csv(config.RAW_DATA_PATH).dropna()

features = config.NUMERIC_FEATURES

X = df[features].values.astype(float)
y = df[config.TARGET_COLUMN].values.astype(float)

X_mean, X_std= X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std
n_samples, n_features = X_scaled.shape

w = np.zeros(n_features)
b =0.0

learning_rate = 0.5
n_iterations = 500

for iteration in range(n_iterations):
    y_pred = X_scaled @ w + b
    error = y_pred - y
    mse = np.mean(error**2)

    dw = (2/n_samples) * (X_scaled.T @ error)
    db = (2/n_samples) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if iteration % 50 == 0 or iteration == n_iterations - 1:
        print(f"Iteration {iteration:4d} | MSE: {mse:,.0f} | RMSE: {np.sqrt(mse):,.0f}")

for name, weight in zip(features, w):
    print(f"  {name:22s}: {weight:10.2f}")
print(f"  {'bias (intercept)':22s}: {b:10.2f}")

