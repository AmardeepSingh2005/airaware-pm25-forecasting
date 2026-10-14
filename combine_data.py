
import pandas as pd
from pathlib import Path

# Load both datasets
pollution = pd.read_csv(
    "data/raw/pm25_new_delhi_recent_90d.csv"
)

weather = pd.read_csv(
    "data/raw/weather_new_delhi.csv"
)

# Convert pollution timestamps to New Delhi local time
pollution["time_local"] = (
    pd.to_datetime(
        pollution["datetime_local"],
        utc=True,
        errors="coerce"
    )
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
    .dt.floor("h")
)

# Convert weather timestamps to local datetime
weather["time_local"] = pd.to_datetime(
    weather["time"],
    errors="coerce"
).dt.floor("h")

# Ensure one weather record per timestamp
weather = weather.drop_duplicates(
    subset="time_local",
    keep="first"
)

# Merge weather into pollution data
combined = pollution.merge(
    weather.drop(columns=["time"]),
    on="time_local",
    how="left",
    validate="many_to_one",
    indicator=True
)

# Save the combined dataset
output = Path("data/processed/pm25_weather_combined.csv")
output.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output, index=False)

# Report the results
print("===== DATA MERGE REPORT =====")
print("Pollution records:", len(pollution))
print("Weather records:", len(weather))
print("Combined records:", len(combined))

print("\nWeather match status:")
print(combined["_merge"].value_counts().to_string())

print("\nMissing values:")
print(
    combined.drop(columns=["_merge"])
    .isna()
    .sum()
    .to_string()
)

print("\nFirst five combined records:")
print(combined.head().to_string(index=False))

print("\nSaved to:", output)
