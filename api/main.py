from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.feature_engineering import get_model_features

BASE_DIR = Path(__file__).resolve().parents[1]
BUNDLE_PATH = BASE_DIR / "artifacts" / "aeroshield_model.joblib"

app = FastAPI(
    title="AeroShield Cameroon API",
    description="Forecast API for climate-driven air quality risk in Cameroon.",
    version="1.0.0",
)

bundle = None
if BUNDLE_PATH.exists():
    bundle = joblib.load(BUNDLE_PATH)


class PredictionRequest(BaseModel):
    city: str
    region: str
    latitude: float
    longitude: float
    month: int
    quarter: int
    dayofyear: int
    weekofyear: int
    weather_code: float
    temperature_2m_max: float
    temperature_2m_min: float
    temperature_2m_mean: float
    apparent_temperature_mean: float
    precipitation_sum: float
    precipitation_hours: float
    wind_speed_10m_max: float
    wind_gusts_10m_max: float
    wind_direction_10m_dominant: float
    shortwave_radiation_sum: float
    sunshine_duration: float
    et0_fao_evapotranspiration: float
    is_dry_day: int
    is_hot_day: int
    temperature_2m_mean_lag1: float
    wind_speed_10m_max_lag1: float
    precipitation_sum_lag1: float
    shortwave_radiation_sum_lag1: float
    et0_fao_evapotranspiration_lag1: float
    aqri_current_lag1: float
    temperature_2m_mean_roll3: float
    wind_speed_10m_max_roll3: float
    precipitation_sum_roll3: float
    shortwave_radiation_sum_roll3: float
    et0_fao_evapotranspiration_roll3: float
    aqri_current_roll3: float
    temperature_2m_mean_roll7: float
    wind_speed_10m_max_roll7: float
    precipitation_sum_roll7: float
    shortwave_radiation_sum_roll7: float
    et0_fao_evapotranspiration_roll7: float
    aqri_current_roll7: float
    dry_streak_7d: float


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "AeroShield Cameroon API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "ok": bool(bundle is not None),
        "model_loaded": bool(bundle is not None),
    }


@app.get("/metrics")
def metrics() -> dict:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Train the model first.")
    return bundle["metrics"]


@app.get("/cities")
def cities() -> list[dict[str, str | float]]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Train the model first.")
    latest = bundle["latest_data"]
    deduped = latest[["city", "region", "latitude", "longitude"]].drop_duplicates().sort_values(["region", "city"])
    return deduped.to_dict(orient="records")


@app.get("/latest-risk")
def latest_risk() -> list[dict]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Train the model first.")
    latest = bundle["latest_data"].sort_values("time").groupby("city", as_index=False).tail(1).copy()
    features = bundle["features"]
    latest["predicted_next_day_aqri"] = bundle["model"].predict(latest[features])
    latest["predicted_alert_level"] = pd.cut(
        latest["predicted_next_day_aqri"],
        bins=[-float("inf"), 33, 66, float("inf")],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    cols = [
        "time",
        "city",
        "region",
        "latitude",
        "longitude",
        "aqri_current",
        "predicted_next_day_aqri",
        "predicted_alert_level",
    ]
    return latest[cols].sort_values("predicted_next_day_aqri", ascending=False).to_dict(orient="records")


@app.get("/city-history/{city_name}")
def city_history(city_name: str) -> list[dict]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Train the model first.")
    data = bundle["latest_data"]
    city_df = data.loc[data["city"].str.lower() == city_name.lower()].copy()
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

    features = bundle["features"]
    city_df["predicted_next_day_aqri"] = bundle["model"].predict(city_df[features])
    result = city_df[[
        "time",
        "city",
        "region",
        "aqri_current",
        "target_next_day_aqri",
        "predicted_next_day_aqri",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
    ]].sort_values("time")
    return result.to_dict(orient="records")


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, float | str]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Train the model first.")
    feature_order = get_model_features()
    df = pd.DataFrame([payload.model_dump()])[feature_order]
    score = float(bundle["model"].predict(df)[0])
    if score <= 33:
        level = "Low"
    elif score <= 66:
        level = "Medium"
    else:
        level = "High"
    return {
        "predicted_next_day_aqri": round(score, 2),
        "alert_level": level,
    }
