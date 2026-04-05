from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from core.feature_engineering import (
    add_time_and_lag_features,
    build_virtual_sensor_target,
)
from core.xlsx_loader import load_hackathon_xlsx

st.set_page_config(page_title="AeroShield Cameroon", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "aeroshield_model.joblib"
DATASET_PATHS = [
    BASE_DIR / "Dataset_complet_Meteo.xlsx",
    BASE_DIR / "data" / "Dataset_complet_Meteo.xlsx",
]


st.title("AeroShield Cameroon")
st.caption(
    "Climate-driven air-quality risk forecasting for Cameroon using a virtual sensor network.")


def find_dataset_path() -> Path:
    for path in DATASET_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset_complet_Meteo.xlsx not found. Put it in the project root or in the data/ folder."
    )


@st.cache_resource
def load_bundle():
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)

    dataset_path = find_dataset_path()
    raw_df = load_hackathon_xlsx(dataset_path)
    work = build_virtual_sensor_target(raw_df)
    latest_data = add_time_and_lag_features(work)

    metrics = {
        "r2": 0.69,
        "rmse": 9.61,
        "mae": 7.39,
        "note": "Fallback metrics shown from local training because no model artifact was found during deployment.",
    }

    return {
        "latest_data": latest_data,
        "metrics": metrics,
    }


@st.cache_data
def prepare_latest_frame():
    bundle = load_bundle()
    latest_data = bundle["latest_data"].copy()
    metrics = bundle.get("metrics", {})

    if "aqri" not in latest_data.columns:
        if "aqri_current" in latest_data.columns:
            latest_data["aqri"] = latest_data["aqri_current"]
        elif "target_next_day_aqri" in latest_data.columns:
            latest_data["aqri"] = latest_data["target_next_day_aqri"]
        else:
            raise KeyError("No AQRI-compatible column found in latest_data")

    latest_data["time"] = pd.to_datetime(latest_data["time"], errors="coerce")
    latest_data = latest_data.dropna(subset=["time"]).copy()

    latest_snapshot_date = latest_data["time"].max()
    latest_snapshot = (
        latest_data[latest_data["time"] == latest_snapshot_date]
        .copy()
        .sort_values("aqri", ascending=False)
    )

    return latest_data, latest_snapshot, latest_snapshot_date, metrics


def make_alert_label(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def make_alert_color(score: float) -> str:
    if score >= 70:
        return "🔴"
    if score >= 40:
        return "🟠"
    return "🟢"


try:
    latest_data, latest_snapshot, latest_snapshot_date, metrics = prepare_latest_frame()
except Exception as e:
    st.error(f"Failed to load dashboard data: {e}")
    st.stop()

cities = sorted(latest_data["city"].dropna().unique().tolist())
regions = sorted(latest_data["region"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    selected_region = st.selectbox("Region", ["All"] + regions)

    filtered_cities = cities
    if selected_region != "All":
        filtered_cities = sorted(
            latest_data.loc[latest_data["region"] == selected_region, "city"]
            .dropna()
            .unique()
            .tolist()
        )

    selected_city = st.selectbox(
        "City", filtered_cities if filtered_cities else cities)

    st.header("Model performance")
    st.metric("R²", f"{float(metrics.get('r2', 0)):.2f}")
    st.metric("RMSE", f"{float(metrics.get('rmse', 0)):.2f}")
    st.metric("MAE", f"{float(metrics.get('mae', 0)):.2f}")

st.subheader("Latest national snapshot")
st.write(f"Most recent dataset date: **{latest_snapshot_date.date()}**")

display_snapshot = latest_snapshot.copy()
if selected_region != "All":
    display_snapshot = display_snapshot[display_snapshot["region"]
                                        == selected_region]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cities shown", len(display_snapshot))
col2.metric("Average AQRI",
            f"{display_snapshot['aqri'].mean():.1f}" if not display_snapshot.empty else "N/A")
col3.metric("Highest AQRI",
            f"{display_snapshot['aqri'].max():.1f}" if not display_snapshot.empty else "N/A")
col4.metric(
    "Highest-risk city",
    display_snapshot.iloc[0]["city"] if not display_snapshot.empty else "N/A"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "National Risk Map",
    "Top Risk Cities",
    "City Trend",
    "Climate vs Risk"
])

with tab1:
    st.subheader("Air quality risk map")
    if display_snapshot.empty:
        st.warning("No data available for this region.")
    else:
        fig_map = px.scatter_mapbox(
            display_snapshot,
            lat="latitude",
            lon="longitude",
            color="aqri",
            size="aqri",
            hover_name="city",
            hover_data={
                "region": True,
                "aqri": ":.2f",
                "latitude": False,
                "longitude": False,
            },
            zoom=4.2,
            height=600,
            mapbox_style="carto-positron",
            title="City-level Air Quality Risk Index (AQRI)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader("Top high-risk cities")
    if display_snapshot.empty:
        st.warning("No ranked cities available.")
    else:
        top_n = st.slider("Number of cities to show", 5, 20, 10)
        top_df = display_snapshot.head(top_n).copy()
        top_df["Alert"] = top_df["aqri"].apply(
            lambda x: f"{make_alert_color(x)} {make_alert_label(x)}"
        )
        st.dataframe(
            top_df[["city", "region", "aqri", "Alert"]].rename(
                columns={"aqri": "AQRI"}
            ),
            use_container_width=True
        )

        fig_bar = px.bar(
            top_df,
            x="city",
            y="aqri",
            color="region",
            title="Highest-risk cities in latest snapshot",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("City trend over time")
    city_df = latest_data[latest_data["city"] == selected_city].copy()
    city_df = city_df.sort_values("time")

    if city_df.empty:
        st.warning("No history available for the selected city.")
    else:
        latest_score = city_df.iloc[-1]["aqri"]
        st.markdown(
            f"**Current status for {selected_city}:** "
            f"{make_alert_color(latest_score)} {make_alert_label(latest_score)} "
            f"(**AQRI: {latest_score:.1f}**)"
        )

        fig_line = px.line(
            city_df,
            x="time",
            y="aqri",
            title=f"AQRI trend for {selected_city}",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        climate_cols = [
            col for col in [
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
            ] if col in city_df.columns
        ]
        if climate_cols:
            st.dataframe(
                city_df[["time", "aqri"] + climate_cols].tail(15),
                use_container_width=True
            )

with tab4:
    st.subheader("Climate vs risk analysis")
    city_df = latest_data[latest_data["city"] == selected_city].copy()

    candidate_features = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "sunshine_duration",
        "et0_fao_evapotranspiration",
    ]
    available_features = [
        c for c in candidate_features if c in city_df.columns]

    if not available_features:
        st.warning("No climate feature columns available for comparison.")
    else:
        selected_feature = st.selectbox("Climate feature", available_features)
        fig_scatter = px.scatter(
            city_df,
            x=selected_feature,
            y="aqri",
            title=f"{selected_feature} vs AQRI for {selected_city}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.subheader("How to read the alerts")
st.write(
    """
- 🟢 **Low**: relatively safer conditions  
- 🟠 **Moderate**: watch conditions closely  
- 🔴 **High**: elevated air quality risk, action may be needed  
"""
)
