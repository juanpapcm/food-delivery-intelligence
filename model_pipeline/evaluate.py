"""Metrics and error-analysis helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}

    def __str__(self) -> str:
        return f"MAE={self.mae:.3f}  RMSE={self.rmse:.3f}  R²={self.r2:.3f}"


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> Metrics:
    """Return MAE / RMSE / R² bundled as a :class:`Metrics`."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(mae=mae, rmse=rmse, r2=r2)


def error_analysis(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    by: list[str] | None = None,
) -> Mapping[str, pd.DataFrame]:
    """Breakdown of absolute error by categorical / binned feature.

    Returns one table per feature in ``by``. Each table has mean absolute
    error, bias (signed mean error), and count per group. Useful for
    identifying cohorts where the model fails.
    """
    by = by or ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
    residuals = y_true.to_numpy() - y_pred
    abs_err = np.abs(residuals)

    frame = X.copy()
    frame["_abs_err"] = abs_err
    frame["_bias"] = residuals

    report: dict[str, pd.DataFrame] = {}
    for col in by:
        if col not in frame.columns:
            continue
        tbl = (
            frame.groupby(col, dropna=False)
            .agg(n=("_abs_err", "size"), mae=("_abs_err", "mean"), bias=("_bias", "mean"))
            .sort_values("mae", ascending=False)
            .round(3)
        )
        report[col] = tbl

    # Distance bucket — the dominant driver, worth breaking out separately.
    if "Distance_km" in frame.columns:
        frame["_dist_bucket"] = pd.cut(
            frame["Distance_km"],
            bins=[0, 3, 7, 12, np.inf],
            labels=["<3km", "3-7km", "7-12km", "12+km"],
        )
        tbl = (
            frame.groupby("_dist_bucket", observed=True)
            .agg(n=("_abs_err", "size"), mae=("_abs_err", "mean"), bias=("_bias", "mean"))
            .round(3)
        )
        report["Distance_bucket"] = tbl

    return report


def log_error_report(report: Mapping[str, pd.DataFrame]) -> None:
    """Log each error-analysis table at INFO level."""
    for name, tbl in report.items():
        logger.info("Error breakdown by %s:\n%s", name, tbl.to_string())
