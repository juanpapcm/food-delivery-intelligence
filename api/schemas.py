"""Request / response schemas for the delivery-time API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Weather = Literal["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
Traffic = Literal["Low", "Medium", "High"]
TimeOfDay = Literal["Morning", "Afternoon", "Evening", "Night"]
Vehicle = Literal["Bike", "Scooter", "Car"]


class PredictRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=100, description="Trip distance in km.")
    weather: Weather
    traffic_level: Traffic
    time_of_day: TimeOfDay
    vehicle_type: Vehicle
    preparation_time_min: float = Field(..., ge=0, le=180)
    courier_experience_yrs: float = Field(..., ge=0, le=50)

    @field_validator("distance_km", "preparation_time_min", "courier_experience_yrs")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be finite")
        return v


class PredictResponse(BaseModel):
    estimated_delivery_time_min: float
    confidence_interval: tuple[float, float]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str
    model_version: str
