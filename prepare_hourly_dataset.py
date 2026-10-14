import pandas as pd
from pathlib import Path

POLLUTION_FILE = "data/raw/pm25_new_delhi_recent_90d.csv"
WEATHER_FILE = "data/raw/weather_new_delhi.csv"

# Load pollution observations
pollution = pd.read_csv(POLLUTION_FILE)
pollution["datetime_utc"] = pd.to_datetime(
    pollution["datetime_utc"], utc=True, errors="coerce"
)
pollution["pm25"] = pd.to_numeric(
    pollution["pm25"], errors="coerce"
)

pollution = (
    pollution.dropna(subset=["datetime_utc", "pm25"])
    .drop_duplicates(subset="datetime_utc")
    .set_index("datetime_utc")
    .sort_index()
)

# Load weather records; source timestamps are New Delhi local time
weather = pd.read_csv(WEATHER_FILE)
weather["time"] = pd.to_datetime(
    weather["time"], errors="coerce"
)

weather["time_utc"] = (
    weather["time"]
    .dt.tz_localize("Asia/Kolkata")
    .dt.tz_convert("UTC")
)

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

weather = (
    weather.dropna(subset=["time_utc"])
    .drop_duplicates(subset="time_utc")
    .set_index("time_utc")
    .sort_index()
)

# Build a complete hourly timeline that preserves the observed
# timestamp alignment, including the half-hour UTC offset.
hourly_index = pd.date_range(
    start=pollution.index.min(),
    end=pollution.index.max(),
    freq="h"
)

pm25 = pollution["pm25"].reindex(hourly_index)
hourly_weather = weather[weather_columns].reindex(hourly_index)

# Predict the next hour using only information available now.
dataset = pd.DataFrame(index=hourly_index)
dataset["pm25_current"] = pm25
dataset["pm25_previous_hour"] = pm25.shift(1)
dataset["pm25_mean_last_3h"] = pm25.rolling(
    window=3, min_periods=3
).mean()

for column in weather_columns:
    dataset[column] = hourly_weather[column]

target_time_local = (
    hourly_index + pd.Timedelta(hours=1)
).tz_convert("Asia/Kolkata")

dataset["forecast_hour_local"] = target_time_local.hour
dataset["forecast_day_of_week_local"] = target_time_local.dayofweek
dataset["target_pm25_next_hour"] = pm25.shift(-1)

# Keep only rows with complete features and a known next-hour target.
required_columns = [
    "pm25_current",
    "pm25_previous_hour",
    "pm25_mean_last_3h",
    *weather_columns,
    "forecast_hour_local",
    "forecast_day_of_week_local",
    "target_pm25_next_hour",
]

ready = dataset.dropna(subset=required_columns).copy()

ready.insert(0, "origin_time_utc", ready.index.astype(str))
ready.insert(
    1, "target_time_utc",
    (ready.index + pd.Timedelta(hours=1)).astype(str)
)

output = Path("data/processed/hourly_forecast_dataset.csv")
output.parent.mkdir(parents=True, exist_ok=True)
ready.to_csv(output, index=False)

print("===== HOURLY FORECAST DATASET =====")
print("Possible hourly slots:", len(hourly_index))
print("Observed pollution readings:", pm25.notna().sum())
print("Usable next-hour examples:", len(ready))
print("Missing-hour gaps are preserved.")
print("Saved to:", output)

print("\nFirst five examples:")
print(ready.head().to_string(index=False))
