"""Extract coefficients from the saved linear model and plot them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_pipeline import config
from model_pipeline.predict import load_pipeline
from model_pipeline.preprocessing import load_data, split_data


def main() -> None:
    pipe, name = load_pipeline()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    features = pipe.named_steps["features"]
    encoder = pipe.named_steps["encode"]
    model = pipe.named_steps["model"]

    X_fe = features.transform(X_train)
    X_enc = encoder.transform(X_fe)
    feat_names = encoder.get_feature_names_out()

    std = X_enc.std(axis=0)
    coefs = model.coef_
    contrib = coefs * std  # standardized effect size (min ~= impact per 1 SD)

    importance = (
        pd.DataFrame({"feature": feat_names, "coef": coefs, "std_effect": contrib})
        .assign(abs_effect=lambda d: d["std_effect"].abs())
        .sort_values("abs_effect", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(config.REPORT_DIR / "feature_importance.csv", index=False)
    print(importance.to_string(index=False))

    # Plot
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    top = importance.head(15).iloc[::-1]
    colors = ["#2a7ae2" if v > 0 else "#d94f4f" for v in top["std_effect"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["std_effect"], color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Standardized effect on Delivery_Time_min  (coef × std)")
    ax.set_title("Feature importance — LinearRegression (top 15)")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "feature_importance.png", dpi=130)
    print(f"Saved: {config.FIGURE_DIR / 'feature_importance.png'}")

    # Residual plot
    y_pred = pipe.predict(X_test)
    resid = y_test.to_numpy() - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, resid, alpha=0.5, s=18)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xlabel("Predicted (min)")
    axes[0].set_ylabel("Residual (actual − pred)")
    axes[0].set_title("Residuals vs prediction")
    axes[0].grid(ls=":", alpha=0.5)

    axes[1].scatter(X_test["Distance_km"], resid, alpha=0.5, s=18)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Distance_km")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs distance")
    axes[1].grid(ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "residuals.png", dpi=130)
    print(f"Saved: {config.FIGURE_DIR / 'residuals.png'}")


if __name__ == "__main__":
    main()
