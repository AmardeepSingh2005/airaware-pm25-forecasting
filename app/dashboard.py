from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "pollution": ROOT / "data/raw/pm25_new_delhi_recent_90d.csv",
    "combined": ROOT / "data/processed/pm25_weather_combined.csv",
    "predictions": ROOT / "reports/hourly_model_predictions.csv",
    "importance": ROOT / "reports/feature_importance.csv",
}

st.set_page_config(
    page_title="AirAware | Air Quality",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ AirAware")
st.caption("Air-quality exploration and one-hour-ahead forecasting | New Delhi, India")

missing_files = [
    str(path) for path in FILES.values() if not path.exists()
]

if missing_files:
    st.error("Required project files are missing:")
    for filename in missing_files:
        st.write(filename)
    st.stop()

pollution = pd.read_csv(FILES["pollution"])
combined = pd.read_csv(FILES["combined"])
predictions = pd.read_csv(FILES["predictions"])
importance = pd.read_csv(FILES["importance"])

pollution["datetime_utc"] = pd.to_datetime(
    pollution["datetime_utc"], utc=True, errors="coerce"
)
pollution["pm25"] = pd.to_numeric(
    pollution["pm25"], errors="coerce"
)
pollution = (
    pollution.dropna(subset=["datetime_utc", "pm25"])
    .sort_values("datetime_utc")
    .drop_duplicates(subset="datetime_utc")
)

combined["time_local"] = pd.to_datetime(
    combined["time_local"], errors="coerce"
)

numeric_columns = [
    "pm25", "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m", "wind_direction_10m",
]
for column in numeric_columns:
    if column in combined.columns:
        combined[column] = pd.to_numeric(
            combined[column], errors="coerce"
        )

tabs = st.tabs([
    "Overview",
    "Weather analysis",
    "Model evaluation",
    "Data quality",
])

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
with tabs[0]:
    latest = pollution.iloc[-1]

    a, b, c, d = st.columns(4)
    a.metric("PM2.5 readings", f"{len(pollution):,}")
    b.metric("Latest recorded PM2.5", f"{latest['pm25']:.1f} µg/m³")
    c.metric("Median PM2.5", f"{pollution['pm25'].median():.1f} µg/m³")
    d.metric("Observed period", "Approx. 90 days")

    st.subheader("PM2.5 over time")
    st.caption(
        "Gaps are preserved: lines break where no measurement exists."
    )

    series = (
        pollution.set_index("datetime_utc")["pm25"]
        .sort_index()
    )
    hourly_index = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq="h",
    )
    hourly = series.reindex(hourly_index)
    rolling = hourly.rolling(
        window=24, min_periods=12
    ).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly.index, hourly, linewidth=0.8, label="Observed PM2.5")
    ax.plot(
        rolling.index, rolling,
        linewidth=2, label="24-hour rolling average"
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Latest available observations")
    st.dataframe(
        pollution[["datetime_utc", "pm25", "unit"]].tail(15),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# WEATHER ANALYSIS
# --------------------------------------------------
with tabs[1]:
    st.subheader("Weather versus PM2.5")

    choices = {
        "Temperature (°C)": "temperature_2m",
        "Relative humidity (%)": "relative_humidity_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (km/h)": "wind_speed_10m",
    }
    label = st.selectbox("Select a weather variable", list(choices))
    variable = choices[label]

    valid = combined[["pm25", variable]].dropna()
    correlation = valid["pm25"].corr(valid[variable])

    left, right = st.columns([2, 1])

    with left:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(valid[variable], valid["pm25"], alpha=0.35, s=12)
        ax.set_xlabel(label)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"PM2.5 versus {label}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.metric("Pearson correlation", f"{correlation:.3f}")
        st.write(f"Matched observations: {len(valid):,}")
        st.info(
            "Correlation describes association, not causation. "
            "Weather values are model-based historical estimates."
        )

    st.subheader("Correlation summary")
    weather_columns = list(choices.values())
    corr = combined[["pm25"] + weather_columns].corr()["pm25"].drop("pm25")
    corr.index = [name.replace("_", " ").title() for name in corr.index]
    st.dataframe(
        corr.rename("Pearson correlation").to_frame().round(3),
        use_container_width=True,
    )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
with tabs[2]:
    st.subheader("One-hour-ahead model evaluation")

    actual_col = "target_pm25_next_hour"
    model_columns = {
        "Persistence baseline": "baseline_prediction",
        "Random Forest": "random_forest_prediction",
    }

    metrics = []
    valid_predictions = predictions.dropna(
        subset=[actual_col] + list(model_columns.values())
    )

    for model_name, prediction_col in model_columns.items():
        errors = (
            valid_predictions[actual_col]
            - valid_predictions[prediction_col]
        )
        metrics.append({
            "Model": model_name,
            "MAE (µg/m³)": errors.abs().mean(),
            "RMSE (µg/m³)": (errors.pow(2).mean()) ** 0.5,
        })

    metrics_df = pd.DataFrame(metrics).set_index("Model")
    st.dataframe(
        metrics_df.round(3),
        use_container_width=True,
    )
    st.caption(
        "Lower MAE and RMSE are better. Results apply to this "
        "chronological test set and are not a guarantee of future accuracy."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df[["MAE (µg/m³)", "RMSE (µg/m³)"]].plot(
        kind="bar", ax=ax
    )
    ax.set_ylabel("Error (µg/m³)")
    ax.set_title("Forecast error comparison")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature importance")
    top_features = importance.nlargest(10, "importance").sort_values(
        "importance"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_xlabel("Relative importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Test predictions")
    st.dataframe(
        valid_predictions.tail(50),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# DATA QUALITY
# --------------------------------------------------
with tabs[3]:
    st.subheader("Data-quality checks")

    timestamps = pollution["datetime_utc"].dropna().sort_values()
    expected = int(
        (timestamps.max() - timestamps.min()).total_seconds() / 3600
    ) + 1
    coverage = len(timestamps) / expected * 100 if expected else 0
    largest_gap = (
        timestamps.diff().dt.total_seconds().div(3600).max()
    )

    x, y, z = st.columns(3)
    x.metric("Valid readings", f"{len(pollution):,}")
    y.metric("Approx. hourly coverage", f"{coverage:.1f}%")
    z.metric("Largest observed gap", f"{largest_gap:.0f} hours")

    st.write("Missing values in the pollution data:")
    st.dataframe(
        pollution.isna().sum().rename("Missing count").to_frame(),
        use_container_width=True,
    )

    st.warning(
        "AirAware is a student research prototype using one sensor "
        "and a limited historical period. Historical weather is estimated. "
        "Forecasts are experimental and should not be used for health "
        "or official environmental decisions."
    )

st.divider()
st.caption("AirAware | Exploratory Data Science Project")
