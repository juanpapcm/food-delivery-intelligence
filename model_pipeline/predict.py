"""Load the persisted pipeline and predict delivery time."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from model_pipeline import config

logger = logging.getLogger(__name__)


def load_pipeline(path: Path = config.BEST_MODEL_PATH) -> tuple[Pipeline, str]:
    """Load the saved pipeline; return (pipeline, model_name)."""
    artifact = joblib.load(path)
    logger.info("Loaded %s from %s (CV MAE=%.3f)",
                artifact["model_name"], path, artifact["cv_mae"])
    return artifact["pipeline"], artifact["model_name"]


def _to_frame(payload: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame(payload)


def predict(
    payload: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    pipeline: Pipeline | None = None,
) -> np.ndarray:
    """Predict delivery time (minutes) for one or many rows.

    ``payload`` may be a DataFrame, a single dict, or a list of dicts whose
    keys match ``config.RAW_FEATURES``. Missing columns are tolerated — the
    preprocessing step handles NaNs the same way it did at training time.
    """
    if pipeline is None:
        pipeline, _ = load_pipeline()

    X = _to_frame(payload)
    for col in config.RAW_FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    X = X[config.RAW_FEATURES]

    preds = pipeline.predict(X)
    return np.asarray(preds)
