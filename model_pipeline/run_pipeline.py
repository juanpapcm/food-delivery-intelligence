"""End-to-end delivery-time pipeline.

Run from the repo root:

    python -m model_pipeline.run_pipeline

Steps: load → split → CV across candidates → fit & save best → evaluate on
held-out test → print per-cohort error breakdown.
"""

from __future__ import annotations

import logging

from model_pipeline import config
from model_pipeline.evaluate import compute_metrics, error_analysis, log_error_report
from model_pipeline.preprocessing import load_data, split_data
from model_pipeline.train import train_and_save_best


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    configure_logging()
    log = logging.getLogger("pipeline")

    log.info("=== 1/4  Load & split ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    log.info("=== 2/4  Cross-validate candidates ===")
    pipe, best, all_results = train_and_save_best(X_train, y_train)

    log.info("=== 3/4  Held-out evaluation ===")
    y_pred = pipe.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    log.info("Test set (%s): %s", best.name, test_metrics)

    log.info("=== 4/4  Error analysis ===")
    report = error_analysis(X_test, y_test, y_pred)
    log_error_report(report)

    log.info("Leaderboard:")
    for r in all_results:
        log.info(
            "  %-20s MAE=%.3f ± %.3f   RMSE=%.3f   R²=%.3f",
            r.name, r.mae_mean, r.mae_std, r.rmse_mean, r.r2_mean,
        )
    log.info("Artifact: %s", config.BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
