import pandas as pd
from pathlib import Path

# Load the combined dataset
pollution = pd.read_csv(
    "data/processed/pm25_weather_combined.csv"
)
weather = pd.read_csv(
    "data/raw/weather_new_delhi.csv"
)

# Parse local timestamps and numeric measurements
pollution["time_local"] = pd.to_datetime(
    pollution["time_local"], errors="coerce"
)
pollution["pm25"] = pd.to_numeric(
    pollution["pm25"], errors="coerce"
)

weather["time"] = pd.to_datetime(
    weather["time"], errors="coerce"
)

# Assign each observation to its local calendar day
pollution["date"] = pollution["time_local"].dt.normalize()
weather["date"] = weather["time"].dt.normalize()

# Aggregate pollution measurements by day
daily_pollution = (
    pollution.dropna(subset=["date"])
    .groupby("date")
    .agg(
        readings=("pm25", "count"),
        pm25_daily_mean_observed=("pm25", "mean"),
        pm25_daily_median=("pm25", "median")
    )
    .reset_index()
)

# Aggregate weather independently, using all available hours
daily_weather = (
    weather.dropna(subset=["date"])
    .groupby("date")
    .agg(
        temperature_2m=("temperature_2m", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        precipitation=("precipitation", "sum"),
        wind_speed_10m=("wind_speed_10m", "mean")
    )
    .reset_index()
)

# Create a complete daily calendar, including missing days
start_date = pollution["date"].min()
end_date = pollution["date"].max()

calendar = pd.DataFrame({
    "date": pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )
})

daily = (
    calendar
    .merge(daily_pollution, on="date", how="left")
    .merge(daily_weather, on="date", how="left")
)

# Distinguish days with no data from days with partial coverage
daily["readings"] = daily["readings"].fillna(0).astype(int)

daily["coverage_percent"] = (
    daily["readings"] / 24 * 100
).round(1)

daily["eligible_for_initial_analysis"] = (
    daily["readings"] >= 18
)

# Keep observed means for transparency, but only use well-covered
# days for our initial daily forecasting analysis.
daily["pm25_daily_mean"] = (
    daily["pm25_daily_mean_observed"]
    .where(daily["eligible_for_initial_analysis"])
)

# Save the prepared dataset
output = Path("data/processed/daily_pm25_weather.csv")
output.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(output, index=False)

print("===== DAILY DATASET REPORT =====")
print("Calendar days:", len(daily))
print("Days with any pollution readings:", (daily["readings"] > 0).sum())
print("Days with at least 18 readings:", daily["eligible_for_initial_analysis"].sum())
print("Days with 1-17 readings:", ((daily["readings"] > 0) & (daily["readings"] < 18)).sum())
print("Days with zero readings:", (daily["readings"] == 0).sum())

print("\nFirst five days:")
print(
    daily[[
        "date", "readings", "coverage_percent",
        "pm25_daily_mean", "temperature_2m",
        "relative_humidity_2m", "wind_speed_10m"
    ]].head().to_string(index=False)
)

print("\nSaved to:", output)
