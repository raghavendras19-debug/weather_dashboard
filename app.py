import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Advanced Weather Dashboard",
    page_icon="🌦",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.metric-box {
    background: linear-gradient(135deg, #74ebd5, #ACB6E5);
    padding: 15px;
    border-radius: 12px;
    color: black;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("weather.csv")

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Dashboard Controls")

states = sorted(df["Station.State"].dropna().unique())
selected_state = st.sidebar.selectbox("Select State", states)

state_df = df[df["Station.State"] == selected_state]
cities = sorted(state_df["Station.City"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities)

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Date.Year"].min()),
    int(df["Date.Year"].max()),
    (int(df["Date.Year"].min()), int(df["Date.Year"].max()))
)

filtered_df = df[
    (df["Station.State"] == selected_state) &
    (df["Station.City"] == selected_city) &
    (df["Date.Year"].between(year_range[0], year_range[1]))
]

if filtered_df.empty:
    st.warning("⚠ No data available for the selected filters.")
    st.stop()

st.title("🌦 Advanced Weather Analysis Dashboard")
st.markdown("### Historical Trends • Seasonal Patterns • Climate Insights")

# ------------------ KPI METRICS ------------------
st.subheader("📌 Climate Summary")

def safe_mean(col):
    return filtered_df[col].dropna().mean()

c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(f"<div class='metric-box'>🌡 Avg Temp<br><b>{safe_mean('Data.Temperature.Avg Temp'):.2f} °C</b></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-box'>🔥 Max Temp<br><b>{safe_mean('Data.Temperature.Max Temp'):.2f} °C</b></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-box'>❄ Min Temp<br><b>{safe_mean('Data.Temperature.Min Temp'):.2f} °C</b></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-box'>🌧 Precipitation<br><b>{safe_mean('Data.Precipitation'):.2f} mm</b></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='metric-box'>💨 Wind Speed<br><b>{safe_mean('Data.Wind.Speed'):.2f} km/h</b></div>", unsafe_allow_html=True)

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends",
    "🍂 Seasonal Patterns",
    "🚨 Anomalies",
    "🌍 Location Comparison",
    "📜 Climate Insights"
])

# ================= TAB 1: TRENDS =================
with tab1:
    st.subheader("Temperature Trends Over Time")

    fig = px.line(
        filtered_df,
        x="Date.Year",
        y=[
            "Data.Temperature.Min Temp",
            "Data.Temperature.Avg Temp",
            "Data.Temperature.Max Temp"
        ],
        markers=True,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Precipitation Trend")

    fig = px.bar(
        filtered_df,
        x="Date.Year",
        y="Data.Precipitation",
        color="Data.Precipitation",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, width="stretch")

# ================= TAB 2: SEASONAL =================
with tab2:
    st.subheader("Seasonal Temperature Heatmap")

    seasonal = (
        filtered_df
        .dropna(subset=["Date.Month", "Data.Temperature.Avg Temp"])
        .groupby("Date.Month", as_index=False)
        .mean(numeric_only=True)
    )

    fig = px.imshow(
        seasonal.set_index("Date.Month")[["Data.Temperature.Avg Temp"]],
        color_continuous_scale="RdYlBu_r",
        labels=dict(x="Metric", y="Month", color="Temperature")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Wind Direction Distribution")

    fig = px.histogram(
        filtered_df.dropna(subset=["Data.Wind.Direction"]),
        x="Data.Wind.Direction",
        nbins=36,
        color_discrete_sequence=["orange"]
    )
    st.plotly_chart(fig, width="stretch")

# ================= TAB 3: ANOMALIES =================
with tab3:
    st.subheader("🚨 Weather Anomaly Detection")

    filtered_df = filtered_df.copy()

    anomaly_features = [
        "Data.Temperature.Avg Temp",
        "Data.Precipitation"
    ]

    anomaly_data = filtered_df[anomaly_features].dropna()

    if anomaly_data.shape[0] < 5:
        st.warning("⚠ Not enough data to perform anomaly detection.")
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        filtered_df.loc[anomaly_data.index, "Anomaly"] = model.fit_predict(anomaly_data)

        fig = px.scatter(
            filtered_df.loc[anomaly_data.index],
            x="Data.Temperature.Avg Temp",
            y="Data.Precipitation",
            color=filtered_df.loc[anomaly_data.index, "Anomaly"].astype(str),
            color_discrete_map={"1": "green", "-1": "red"},
            labels={"color": "Anomaly"}
        )
        st.plotly_chart(fig, width="stretch")
        st.error(
            f"🚨 Detected {len(filtered_df[filtered_df['Anomaly'] == -1])} anomalous records"
        )

# ================= TAB 4: LOCATION =================
with tab4:
    st.subheader("City-wise Climate Comparison")

    compare_cities = st.multiselect(
        "Select Cities",
        df["Station.City"].dropna().unique(),
        default=df["Station.City"].dropna().unique()[:4]
    )

    comp_df = df[df["Station.City"].isin(compare_cities)]

    fig = px.box(
        comp_df,
        x="Station.City",
        y="Data.Temperature.Avg Temp",
        color="Station.City",
        title="Average Temperature Comparison"
    )
    st.plotly_chart(fig, width="stretch")

# ================= TAB 5: INSIGHTS =================
with tab5:
    st.subheader("Long-Term Climate Insights")

    st.markdown("""
    **Key Observations:**
    - Increasing average temperature trends indicate potential climate change.
    - Seasonal cycles show predictable warming and cooling phases.
    - Anomalies highlight extreme weather events.
    - Wind and precipitation patterns vary significantly across regions.
    - Urban locations tend to show higher temperature variability.
    """)

    fig = px.scatter_matrix(
        filtered_df.dropna(),
        dimensions=[
            "Data.Temperature.Avg Temp",
            "Data.Precipitation",
            "Data.Wind.Speed"
        ],
        color="Date.Month"
    )
    st.plotly_chart(fig, width="stretch")

st.success("✅ Interactive Weather Dashboard Loaded Successfully")
