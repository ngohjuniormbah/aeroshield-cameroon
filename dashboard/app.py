from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("AEROSHIELD_API", "http://127.0.0.1:8001")

st.set_page_config(page_title="AeroShield Cameroon", layout="wide")

st.title("AeroShield Cameroon")
st.caption(
    "Climate-driven air-quality risk forecasting for Cameroon using a virtual sensor network.")


@st.cache_data(ttl=60)
def fetch_json(path: str):
    response = requests.get(f"{API_BASE}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


try:
    metrics = fetch_json("/metrics")
    latest_risk = pd.DataFrame(fetch_json("/latest-risk"))
    cities = pd.DataFrame(fetch_json("/cities"))
except Exception as exc:
    st.error(
        "API not reachable or model not trained yet. Start the backend after training the model.\n\n"
        f"Error: {exc}"
    )
    st.stop()

latest_risk["time"] = pd.to_datetime(latest_risk["time"])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Model MAE", f"{metrics['mae']:.2f}")
kpi2.metric("Model RMSE", f"{metrics['rmse']:.2f}")
kpi3.metric("Model R²", f"{metrics['r2']:.2f}")
kpi4.metric("Cities Covered", f"{cities['city'].nunique()}")

st.subheader("National risk heatmap")
map_fig = px.scatter_map(
    latest_risk,
    lat="latitude",
    lon="longitude",
    size="predicted_next_day_aqri",
    color="predicted_alert_level",
    hover_name="city",
    hover_data={
        "region": True,
        "predicted_next_day_aqri": ":.2f",
        "aqri_current": ":.2f",
        "latitude": False,
        "longitude": False,
    },
    zoom=4.6,
    height=600,
)
map_fig.update_layout(map_style="open-street-map",
                      margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(map_fig, use_container_width=True)

st.subheader("Top cities by next-day risk")
st.dataframe(
    latest_risk[["city", "region", "aqri_current",
                 "predicted_next_day_aqri", "predicted_alert_level"]]
    .sort_values("predicted_next_day_aqri", ascending=False)
    .rename(columns={
        "aqri_current": "Current AQRI",
        "predicted_next_day_aqri": "Predicted Next-Day AQRI",
        "predicted_alert_level": "Alert",
    }),
    use_container_width=True,
)

st.subheader("Climate vs risk explorer")
selected_city = st.selectbox(
    "Choose a city", options=sorted(latest_risk["city"].unique()))
city_history = pd.DataFrame(fetch_json(f"/city-history/{selected_city}"))
city_history["time"] = pd.to_datetime(city_history["time"])

left, right = st.columns(2)
with left:
    risk_fig = px.line(
        city_history,
        x="time",
        y=["aqri_current", "predicted_next_day_aqri"],
        title=f"Risk trajectory for {selected_city}",
    )
    st.plotly_chart(risk_fig, use_container_width=True)
with right:
    scatter_fig = px.scatter(
        city_history,
        x="temperature_2m_mean",
        y="predicted_next_day_aqri",
        size="shortwave_radiation_sum",
        color="precipitation_sum",
        title="Heat, radiation, rainfall and risk",
        hover_data=["time", "wind_speed_10m_max"],
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("Decision support")
current_top = latest_risk.sort_values(
    "predicted_next_day_aqri", ascending=False).head(5)
for _, row in current_top.iterrows():
    alert = row["predicted_alert_level"]
    if alert == "High":
        st.error(
            f"{row['city']} ({row['region']}) is forecast as HIGH risk tomorrow with AQRI {row['predicted_next_day_aqri']:.1f}.")
    elif alert == "Medium":
        st.warning(
            f"{row['city']} ({row['region']}) is forecast as MEDIUM risk tomorrow with AQRI {row['predicted_next_day_aqri']:.1f}.")
    else:
        st.success(
            f"{row['city']} ({row['region']}) is forecast as LOW risk tomorrow with AQRI {row['predicted_next_day_aqri']:.1f}.")

st.markdown("""
### How to explain the dashboard
- The map acts like a virtual sensor network across Cameroon.
- Each point estimates tomorrow's city-level air-quality risk.
- The charts reveal which climate drivers are associated with higher risk.
- The alert area converts analytics into action for health planners, schools, and communities.
""")
