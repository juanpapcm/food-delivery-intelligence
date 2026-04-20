"""Data loading, feature engineering, and preprocessing.

The public surface is:

- ``load_data`` — read the CSV.
- ``split_data`` — train/test split.
- ``build_preprocessor`` — return a fitted-on-transform sklearn
  ``ColumnTransformer`` + FE block that plugs into any estimator Pipeline.

Feature engineering added here:
    * Missing-value handling (categoricals → "Unknown", experience → median).
    * Ordinal encoding of ``Traffic_Level`` (preserves monotonic signal).
    * ``Distance_x_Traffic`` interaction feature.
    * Bucketed ``Experience_Bin`` (new / mid / experienced).
    * One-hot encoding on categoricals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_pipeline import config

logger = logging.getLogger(__name__)


def load_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Load the raw delivery-times CSV and drop rows with no target."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing target", before - len(df))
    logger.info("Loaded %d rows, %d cols from %s", df.shape[0], df.shape[1], path)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratify-free train/test split on the target."""
    X = df[config.RAW_FEATURES].copy()
    y = df[config.TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer delivery-specific features.

    Learns the training-set median of ``Courier_Experience_yrs`` at ``fit``
    time and reuses it on ``transform``, so train/test leakage is avoided.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        self.exp_median_: float = float(X["Courier_Experience_yrs"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
            X[col] = X[col].fillna("Unknown")
        X["Courier_Experience_yrs"] = X["Courier_Experience_yrs"].fillna(self.exp_median_)

        X["Traffic_Ord"] = X["Traffic_Level"].map(config.TRAFFIC_ORDINAL).astype(float)
        X["Distance_x_Traffic"] = X["Distance_km"] * X["Traffic_Ord"]

        X["Experience_Bin"] = pd.cut(
            X["Courier_Experience_yrs"],
            bins=config.EXPERIENCE_BINS,
            labels=config.EXPERIENCE_LABELS,
        ).astype(str)

        return X


def build_preprocessor() -> Pipeline:
    """Build the feature-engineering + encoding pipeline (unfitted)."""
    numeric_out = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
        "Traffic_Ord",
        "Distance_x_Traffic",
    ]
    categorical_out = [
        "Weather",
        "Time_of_Day",
        "Vehicle_Type",
        "Experience_Bin",
    ]

    encoder = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_out),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_out),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("features", FeatureEngineer()), ("encode", encoder)])
