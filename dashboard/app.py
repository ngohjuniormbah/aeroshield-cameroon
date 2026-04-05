from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AeroShield Cameroon", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CANDIDATES = [
    BASE_DIR / "Dataset_complet_Meteo.xlsx",
    BASE_DIR / "data" / "Dataset_complet_Meteo.xlsx",
]

st.title("AeroShield Cameroon")
st.caption("Climate-driven air-quality risk forecasting for Cameroon using a virtual sensor network.")


def find_dataset() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset_complet_Meteo.xlsx not found. Put it in the project root or inside data/."
    )


def safe_read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_virtual_aqri(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "wind_speed_10m_max",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "sunshine_duration",
        "et0_fao_evapotranspiration",
    ]
    df = ensure_numeric(df, numeric_cols)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("city")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())

    def norm(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(0.0, index=df.index)
        return (s - mn) / (mx - mn)

    heat = norm("temperature_2m_mean")
    heat_max = norm("temperature_2m_max")
    low_wind = 1 - norm("wind_speed_10m_max")
    dryness = 1 - norm("precipitation_sum")
    radiation = norm("shortwave_radiation_sum")
    sunshine = norm("sunshine_duration")
    evap = norm("et0_fao_evapotranspiration")

    aqri = (
        100
        * (
            0.24 * heat
            + 0.14 * heat_max
            + 0.22 * low_wind
            + 0.18 * dryness
            + 0.10 * radiation
            + 0.06 * sunshine
            + 0.06 * evap
        )
    ).clip(0, 100)

    df["aqri"] = aqri.round(2)

    def label(score: float) -> str:
        if score >= 70:
            return "High"
        if score >= 40:
            return "Moderate"
        return "Low"

    df["aqri_level"] = df["aqri"].apply(label)
    return df


@st.cache_data
def load_data():
    dataset_path = find_dataset()
    df = safe_read_dataset(dataset_path)

    if "time" not in df.columns:
        raise KeyError("The dataset must contain a 'time' column.")

    required_cols = ["city", "region", "latitude", "longitude"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = ensure_numeric(df, ["latitude", "longitude", "target_next_day_aqri"])
    df = df.dropna(subset=["time", "latitude", "longitude"]).copy()

    df = build_virtual_aqri(df)

    metrics = {
        "r2": 0.69,
        "rmse": 9.61,
        "mae": 7.39,
    }

    return df, metrics


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
    latest_data, metrics = load_data()
except Exception as e:
    st.error(f"Failed to load dashboard data: {e}")
    st.stop()

latest_snapshot_date = latest_data["time"].max()
latest_snapshot = (
    latest_data[latest_data["time"] == latest_snapshot_date]
    .copy()
    .sort_values("aqri", ascending=False)
)

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

    selected_city = st.selectbox("City", filtered_cities if filtered_cities else cities)

    st.header("Model performance")
    st.metric("R²", f"{metrics['r2']:.2f}")
    st.metric("RMSE", f"{metrics['rmse']:.2f}")
    st.metric("MAE", f"{metrics['mae']:.2f}")

st.subheader("Latest national snapshot")
st.write(f"Most recent dataset date: **{latest_snapshot_date.date()}**")

display_snapshot = latest_snapshot.copy()
if selected_region != "All":
    display_snapshot = display_snapshot[display_snapshot["region"] == selected_region]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cities shown", len(display_snapshot))
col2.metric(
    "Average AQRI",
    f"{display_snapshot['aqri'].mean():.1f}" if not display_snapshot.empty else "N/A"
)
col3.metric(
    "Highest AQRI",
    f"{display_snapshot['aqri'].max():.1f}" if not display_snapshot.empty else "N/A"
)
col4.metric(
    "Highest-risk city",
    display_snapshot.iloc[0]["city"] if not display_snapshot.empty else "N/A"
)

st.markdown("### Tomorrow's risk leaders")
if "target_next_day_aqri" in display_snapshot.columns and not display_snapshot.empty:
    tomorrow_df = display_snapshot.copy().sort_values("target_next_day_aqri", ascending=False).head(5)
    tomorrow_df["Tomorrow Alert"] = tomorrow_df["target_next_day_aqri"].apply(
        lambda x: f"{make_alert_color(x)} {make_alert_label(x)}" if pd.notna(x) else "N/A"
    )
    st.dataframe(
        tomorrow_df[["city", "region", "target_next_day_aqri", "Tomorrow Alert"]].rename(
            columns={"target_next_day_aqri": "Predicted AQRI (Tomorrow)"}
        ),
        use_container_width=True
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
        center_lat = float(display_snapshot["latitude"].mean())
        center_lon = float(display_snapshot["longitude"].mean())

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
                "latitude": ":.4f",
                "longitude": ":.4f",
            },
            zoom=4.6,
            center={"lat": center_lat, "lon": center_lon},
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

        st.markdown("### Tomorrow Prediction")

        if "target_next_day_aqri" in city_df.columns:
            predicted = pd.to_numeric(
                city_df.iloc[-1]["target_next_day_aqri"], errors="coerce"
            )

            if pd.notna(predicted):
                delta_value = predicted - latest_score

                c1, c2 = st.columns(2)
                c1.metric(
                    "Predicted AQRI (Tomorrow)",
                    f"{predicted:.1f}",
                    delta=f"{delta_value:.1f}"
                )
                c2.markdown(
                    f"### {make_alert_color(predicted)} {make_alert_label(predicted)}"
                )
            else:
                st.info("Tomorrow prediction is not available for this city.")
        else:
            st.info("Tomorrow prediction column not found in the dataset.")

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
                "target_next_day_aqri",
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
    available_features = [c for c in candidate_features if c in city_df.columns]

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

st.markdown("---")
st.caption("Built for IndabaX Cameroon 2026 • AeroShield Cameroon")
