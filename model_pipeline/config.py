"""Central configuration for the delivery-time pipeline.

All paths, feature lists, and model hyperparameters live here so the rest of
the pipeline is environment-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = REPO_ROOT / "data" / "Food_Delivery_Times.csv"

OUTPUT_DIR: Path = REPO_ROOT / "outputs"
MODEL_DIR: Path = OUTPUT_DIR / "models"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
REPORT_DIR: Path = OUTPUT_DIR / "reports"

BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.pkl"

TARGET: str = "Delivery_Time_min"

NUMERIC_FEATURES: list[str] = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]
CATEGORICAL_FEATURES: list[str] = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]
RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Ordinal mapping for Traffic_Level (used to build the Distance × Traffic
# interaction). 'Unknown' is mapped to the middle bucket.
TRAFFIC_ORDINAL: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 2}

# Bucket edges for Courier_Experience_yrs → {new, mid, experienced}.
EXPERIENCE_BINS: list[float] = [-0.01, 1.0, 3.0, 100.0]
EXPERIENCE_LABELS: list[str] = ["new", "mid", "experienced"]

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# n_jobs=1 everywhere: dataset is tiny (n=1000), so parallel trees cost more
# in process overhead / RAM than they save. The outer CV is also serial to
# avoid double-parallelization (K folds × all cores each = memory blowup).
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
    },
}
