import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from housing import config
from housing.data.data_loader import load_raw_data
from housing.ml.train import build_pipeline

IMG_DIR = config.IMG_DIR  # <storage>/outputs/img

def plot_feature_vs_target(df) -> None:
    """Step 1: does each feature LOOK linearly related to price?"""
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()

    for i, feature in enumerate(config.NUMERIC_FEATURES):
        axes[i].scatter(df[feature], df[config.TARGET_COLUMN], alpha=0.3, s=10)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(config.TARGET_COLUMN)
        axes[i].set_title(f"{feature} vs {config.TARGET_COLUMN}")

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_vs_target.png", dpi=100)
    print("Saved feature_vs_target.png")
    plt.close()

def plot_residual_diagnostics(df) -> None:
    """Step 2: the REAL check -- fit the model, then examine its errors."""
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    pipeline = build_pipeline()
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0,  color='red', linestyle= "--")
    axes[0].set_xlabel("Predicted price")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predicted\n(look for patterns -- there should be none)")

    # Distribution of residuals -- should look roughly bell-shaped/normal
    # if the linear model's error assumptions hold.
    axes[1].hist(residuals, bins=40, edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Distribution of residuals\n(should look roughly normal)")

    plt.tight_layout()
    plt.savefig(IMG_DIR / "residual_diagnostics.png", dpi=100)
    print("Saved residual_diagnostics.png")
    plt.close()

    # Quick numeric signal too, not just visual
    print(f"\nResidual mean: {residuals.mean():,.1f} (should be close to 0)")
    print(f"Residual std:  {residuals.std():,.1f}")

if __name__ == "__main__":
    df = load_raw_data()
    plot_feature_vs_target(df)
    plot_residual_diagnostics(df)
    print("\nOpen the two PNG files to visually inspect linearity assumptions.")