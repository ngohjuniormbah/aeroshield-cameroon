from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_STAGNATION_CODES = {3, 45, 48}
DUSTY_OR_WINDY_CODES = {55, 56, 57, 66, 67}



def build_virtual_sensor_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a transparent air-exposure proxy from meteorological conditions.

    This is the key innovation of the project:
    we simulate a low-cost virtual air-quality sensing layer using weather-only data,
    because the baseline dataset does not include measured PM2.5.
    """
    work = df.copy()

    wind_factor = np.clip((15 - work["wind_speed_10m_max"]) / 15, 0, 1)
    rain_factor = np.clip((3 - work["precipitation_sum"]) / 3, 0, 1)
    heat_factor = np.clip((work["temperature_2m_mean"] - 24) / 10, 0, 1)
    solar_factor = np.clip((work["shortwave_radiation_sum"] - 15) / 12, 0, 1)
    dryness_factor = np.clip((work["et0_fao_evapotranspiration"] - 3) / 4, 0, 1)
    stagnation_factor = work["weather_code"].isin(WEATHER_STAGNATION_CODES).astype(float)
    dust_factor = work["weather_code"].isin(DUSTY_OR_WINDY_CODES).astype(float)

    work["aqri_current"] = 100 * (
        0.24 * wind_factor
        + 0.18 * rain_factor
        + 0.18 * heat_factor
        + 0.14 * solar_factor
        + 0.12 * dryness_factor
        + 0.08 * stagnation_factor
        + 0.06 * dust_factor
    )

    work["aqri_level_current"] = pd.cut(
        work["aqri_current"],
        bins=[-np.inf, 33, 66, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return work



def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values(["city", "time"]).reset_index(drop=True)

    work["month"] = work["time"].dt.month
    work["quarter"] = work["time"].dt.quarter
    work["dayofyear"] = work["time"].dt.dayofyear
    work["weekofyear"] = work["time"].dt.isocalendar().week.astype(int)
    work["is_dry_day"] = (work["precipitation_sum"] <= 1).astype(int)
    work["is_hot_day"] = (work["temperature_2m_mean"] >= 28).astype(int)

    lag_columns = [
        "temperature_2m_mean",
        "wind_speed_10m_max",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "aqri_current",
    ]

    for col in lag_columns:
        work[f"{col}_lag1"] = work.groupby("city")[col].shift(1)
        work[f"{col}_roll3"] = (
            work.groupby("city")[col].rolling(3).mean().reset_index(level=0, drop=True)
        )
        work[f"{col}_roll7"] = (
            work.groupby("city")[col].rolling(7).mean().reset_index(level=0, drop=True)
        )

    work["dry_streak_7d"] = (
        work.groupby("city")["is_dry_day"].rolling(7).sum().reset_index(level=0, drop=True)
    )

    work["target_next_day_aqri"] = work.groupby("city")["aqri_current"].shift(-1)
    work["target_next_day_level"] = pd.cut(
        work["target_next_day_aqri"],
        bins=[-np.inf, 33, 66, np.inf],
        labels=["Low", "Medium", "High"],
    )

    return work.dropna().reset_index(drop=True)



def get_model_features() -> list[str]:
    return [
        "city",
        "region",
        "latitude",
        "longitude",
        "month",
        "quarter",
        "dayofyear",
        "weekofyear",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "precipitation_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "sunshine_duration",
        "et0_fao_evapotranspiration",
        "is_dry_day",
        "is_hot_day",
        "temperature_2m_mean_lag1",
        "wind_speed_10m_max_lag1",
        "precipitation_sum_lag1",
        "shortwave_radiation_sum_lag1",
        "et0_fao_evapotranspiration_lag1",
        "aqri_current_lag1",
        "temperature_2m_mean_roll3",
        "wind_speed_10m_max_roll3",
        "precipitation_sum_roll3",
        "shortwave_radiation_sum_roll3",
        "et0_fao_evapotranspiration_roll3",
        "aqri_current_roll3",
        "temperature_2m_mean_roll7",
        "wind_speed_10m_max_roll7",
        "precipitation_sum_roll7",
        "shortwave_radiation_sum_roll7",
        "et0_fao_evapotranspiration_roll7",
        "aqri_current_roll7",
        "dry_streak_7d",
    ]
