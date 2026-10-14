
import pandas as pd
from pathlib import Path

# Load the combined pollution and weather dataset
df = pd.read_csv(
    "data/processed/pm25_weather_combined.csv"
)

# Parse local timestamps and PM2.5 values
df["time_local"] = pd.to_datetime(
    df["time_local"], errors="coerce"
)

df["pm25"] = pd.to_numeric(
    df["pm25"], errors="coerce"
)

df = df.dropna(subset=["time_local", "pm25"]).copy()

# Count actual PM2.5 readings per local calendar day
df["date"] = df["time_local"].dt.date

daily = (
    df.groupby("date")
    .agg(
        readings=("pm25", "count"),
        daily_mean_pm25=("pm25", "mean")
    )
    .reset_index()
)

# A day with at least 18 of 24 expected hourly readings
# is provisionally considered suitable for daily analysis.
daily["coverage_percent"] = (
    daily["readings"] / 24 * 100
).round(1)

daily["at_least_18_readings"] = daily["readings"] >= 18

# Save the report without modifying the original data
output = Path("reports/daily_data_coverage.csv")
output.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(output, index=False)

print("===== DAILY DATA COVERAGE =====")
print("Calendar days with readings:", len(daily))
print(
    "Days with at least 18 readings:",
    daily["at_least_18_readings"].sum()
)
print(
    "Days with fewer than 12 readings:",
    (daily["readings"] < 12).sum()
)

print("\nReadings per day:")
print(daily["readings"].describe().round(2))

print("\nDays with the fewest readings:")
print(
    daily.nsmallest(10, "readings")[
        ["date", "readings", "coverage_percent"]
    ].to_string(index=False)
)

print("\nSaved report to:", output)

