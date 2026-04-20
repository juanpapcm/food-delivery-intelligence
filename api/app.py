"""FastAPI service for delivery-time predictions.

Run locally:

    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from model_pipeline import config
from model_pipeline.predict import load_pipeline
from model_pipeline.preprocessing import load_data, split_data

MODEL_VERSION = "1.0.0"
CI_Z = 1.28  # ~80% interval from a normal residual assumption

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and estimate residual std once at startup."""
    try:
        pipeline, model_name = load_pipeline()
    except FileNotFoundError as exc:
        logger.error("Model artifact missing at %s: %s", config.BEST_MODEL_PATH, exc)
        raise

    df = load_data()
    X_train, _, y_train, _ = split_data(df)
    resid = y_train.to_numpy() - pipeline.predict(X_train)
    resid_std = float(np.std(resid))
    logger.info("Residual std (train) = %.3f min → ±%.2f for CI", resid_std, CI_Z * resid_std)

    state["pipeline"] = pipeline
    state["model_name"] = model_name
    state["resid_std"] = resid_std
    yield
    state.clear()


app = FastAPI(
    title="Food-delivery ETA",
    version=MODEL_VERSION,
    description="Predicts delivery time in minutes given trip context.",
    lifespan=lifespan,
)


def _request_to_frame(req: PredictRequest) -> pd.DataFrame:
    """Map the API schema to the raw column names the pipeline expects."""
    return pd.DataFrame([{
        "Distance_km": req.distance_km,
        "Weather": req.weather,
        "Traffic_Level": req.traffic_level,
        "Time_of_Day": req.time_of_day,
        "Vehicle_Type": req.vehicle_type,
        "Preparation_Time_min": req.preparation_time_min,
        "Courier_Experience_yrs": req.courier_experience_yrs,
    }])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in state,
        model_name=str(state.get("model_name", "")),
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = _request_to_frame(req)
        y_pred = float(pipeline.predict(X)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    if y_pred < 0 or not np.isfinite(y_pred):
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction")

    half_width = CI_Z * float(state["resid_std"])
    lo = max(0.0, y_pred - half_width)
    hi = y_pred + half_width

    return PredictResponse(
        estimated_delivery_time_min=round(y_pred, 2),
        confidence_interval=(round(lo, 2), round(hi, 2)),
        model_version=MODEL_VERSION,
    )


@app.exception_handler(ValueError)
async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
