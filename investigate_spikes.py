
import pandas as pd

file_path = "data/raw/pm25_new_delhi_recent_90d.csv"

df = pd.read_csv(file_path)

df["datetime_utc"] = pd.to_datetime(
    df["datetime_utc"], utc=True, errors="coerce"
)

df["pm25"] = pd.to_numeric(
    df["pm25"], errors="coerce"
)

df = df.dropna(subset=["datetime_utc", "pm25"])

# Display the 10 highest measurements
top_readings = df.nlargest(10, "pm25")[
    ["datetime_utc", "pm25", "unit"]
]

print("===== TOP 10 PM2.5 READINGS =====")
print(top_readings.to_string(index=False))

# Calculate useful distribution statistics
print("\n===== DISTRIBUTION CHECK =====")
print("Median:", round(df["pm25"].median(), 2))
print("95th percentile:", round(df["pm25"].quantile(0.95), 2))
print("99th percentile:", round(df["pm25"].quantile(0.99), 2))
print("Maximum:", round(df["pm25"].max(), 2))

# Count observations above the dataset's 95th percentile
threshold = df["pm25"].quantile(0.95)

print(
    "\nReadings at or above the 95th percentile:",
    (df["pm25"] >= threshold).sum()
)
