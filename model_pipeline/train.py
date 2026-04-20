"""Model training: CV across candidates, select best by MAE, refit, save."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from model_pipeline import config
from model_pipeline.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    name: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    r2_mean: float


def _build_candidates() -> dict[str, Pipeline]:
    """Return {name: full Pipeline (preprocessing + estimator)}."""
    estimators = {
        "linear_regression": LinearRegression(**config.MODEL_PARAMS["linear_regression"]),
        "random_forest": RandomForestRegressor(**config.MODEL_PARAMS["random_forest"]),
        "xgboost": XGBRegressor(**config.MODEL_PARAMS["xgboost"]),
    }
    return {
        name: Pipeline(steps=[*build_preprocessor().steps, ("model", est)])
        for name, est in estimators.items()
    }


def cross_validate_models(
    X: pd.DataFrame, y: pd.Series, folds: int = config.CV_FOLDS
) -> tuple[Mapping[str, Pipeline], list[CVResult]]:
    """Run ``folds``-fold CV for every candidate model.

    Returns the candidate dictionary (unfitted) plus a sorted-by-MAE list of
    results.
    """
    candidates = _build_candidates()
    kf = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results: list[CVResult] = []
    for name, pipe in candidates.items():
        logger.info("CV: %s (%d folds)", name, folds)
        cv = cross_validate(pipe, X, y, cv=kf, scoring=scoring, n_jobs=1)
        res = CVResult(
            name=name,
            mae_mean=float(-np.mean(cv["test_mae"])),
            mae_std=float(np.std(cv["test_mae"])),
            rmse_mean=float(-np.mean(cv["test_rmse"])),
            r2_mean=float(np.mean(cv["test_r2"])),
        )
        logger.info(
            "  %s → MAE=%.3f ± %.3f  RMSE=%.3f  R²=%.3f",
            name, res.mae_mean, res.mae_std, res.rmse_mean, res.r2_mean,
        )
        results.append(res)

    results.sort(key=lambda r: r.mae_mean)
    return candidates, results


def train_and_save_best(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path = config.BEST_MODEL_PATH,
) -> tuple[Pipeline, CVResult, list[CVResult]]:
    """Fit the best candidate on the full training set and persist it."""
    candidates, results = cross_validate_models(X_train, y_train)
    best = results[0]
    logger.info("Best model by MAE: %s (%.3f)", best.name, best.mae_mean)

    pipe = candidates[best.name]
    pipe.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "model_name": best.name, "cv_mae": best.mae_mean}, output_path)
    logger.info("Saved best model to %s", output_path)

    return pipe, best, results
