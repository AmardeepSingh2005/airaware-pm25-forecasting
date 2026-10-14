
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load recent PM2.5 data
file_path = "data/raw/pm25_new_delhi_recent_90d.csv"
df = pd.read_csv(file_path)

# Clean data types
df["datetime_utc"] = pd.to_datetime(
    df["datetime_utc"], utc=True, errors="coerce"
)
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

df = df.dropna(subset=["datetime_utc", "pm25"])
df = df.sort_values("datetime_utc")
df = df.drop_duplicates(subset="datetime_utc", keep="first")

# Set timestamps as the index
df = df.set_index("datetime_utc")

# Create a complete hourly timeline.
# Missing hours remain NaN; we do not invent readings.
hourly_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="h"
)

hourly_pm25 = df["pm25"].reindex(hourly_index)

# Calculate a rolling average using available readings
rolling_avg = hourly_pm25.rolling(
    window=24,
    min_periods=12
).mean()

# Plot the data
plt.figure(figsize=(14, 6))

plt.plot(
    hourly_pm25.index,
    hourly_pm25,
    label="Observed PM2.5",
    linewidth=0.8,
    alpha=0.65
)

plt.plot(
    rolling_avg.index,
    rolling_avg,
    label="24-hour rolling average",
    linewidth=2
)

plt.title("New Delhi PM2.5: Recent 90-Day Overview")
plt.xlabel("Date (UTC)")
plt.ylabel("PM2.5 concentration (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

# Save the figure
output = Path("reports/pm25_time_series.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200, bbox_inches="tight")

print("Graph saved to:", output)
print("Hourly slots:", len(hourly_pm25))
print("Missing hourly slots:", hourly_pm25.isna().sum())

plt.show()
