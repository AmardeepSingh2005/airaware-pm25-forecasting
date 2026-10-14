
import pandas as pd

file_path = "data/raw/pm25_new_delhi_recent_90d.csv"

df = pd.read_csv(file_path)

df["datetime_utc"] = pd.to_datetime(
    df["datetime_utc"],
    utc=True,
    errors="coerce"
)

df["pm25"] = pd.to_numeric(
    df["pm25"],
    errors="coerce"
)

df = (
    df.dropna(subset=["datetime_utc", "pm25"])
      .sort_values("datetime_utc")
      .reset_index(drop=True)
)

# Find the highest recorded PM2.5 value
peak_index = df["pm25"].idxmax()
peak = df.loc[peak_index]
peak_time = peak["datetime_utc"]

# Select observations six hours before and after the peak
start_time = peak_time - pd.Timedelta(hours=6)
end_time = peak_time + pd.Timedelta(hours=6)

window = df[
    (df["datetime_utc"] >= start_time) &
    (df["datetime_utc"] <= end_time)
]

print("===== PEAK PERIOD INVESTIGATION =====")
print("Peak timestamp (UTC):", peak_time)
print("Peak PM2.5:", peak["pm25"], "µg/m³")
print("Observations in 12-hour window:", len(window))

print("\n===== NEARBY OBSERVATIONS =====")
print(
    window[["datetime_utc", "pm25"]]
    .to_string(index=False)
)

print("\nMean PM2.5 in window:", round(window["pm25"].mean(), 2))
print("Median PM2.5 in window:", round(window["pm25"].median(), 2))
