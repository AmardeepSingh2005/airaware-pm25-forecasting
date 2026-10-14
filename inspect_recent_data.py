import pandas as pd

file_path = "data/raw/pm25_new_delhi_recent_90d.csv"

df = pd.read_csv(file_path)

# Convert timestamps and measurements into usable data types
dates = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
pm25 = pd.to_numeric(df["pm25"], errors="coerce")

# Sort timestamps from oldest to newest
valid_dates = dates.dropna().sort_values().reset_index(drop=True)

# Calculate the time between consecutive measurements
intervals = valid_dates.diff().dropna().dt.total_seconds() / 3600

print("===== RECENT DATA QUALITY REPORT =====")
print("Total rows:", len(df))
print("Invalid timestamps:", dates.isna().sum())
print("Duplicate timestamps:", dates.dropna().duplicated().sum())
print("Missing PM2.5 values:", pm25.isna().sum())

print("\n===== TIME COVERAGE =====")
print("First timestamp:", valid_dates.min())
print("Last timestamp:", valid_dates.max())

span_hours = (
    valid_dates.max() - valid_dates.min()
).total_seconds() / 3600

expected_hourly_slots = int(span_hours) + 1
unique_timestamps = valid_dates.nunique()

coverage = (
    unique_timestamps / expected_hourly_slots * 100
    if expected_hourly_slots > 0 else 0
)

print("Approximate hourly slots:", expected_hourly_slots)
print("Unique timestamps:", unique_timestamps)
print("Approximate hourly coverage:", round(coverage, 2), "%")

print("\n===== TIME INTERVALS =====")
print("Median interval (hours):", intervals.median())
print("Largest gap (hours):", intervals.max())
print("Gaps over 1.5 hours:", (intervals > 1.5).sum())

print("\n===== PM2.5 STATISTICS =====")
print(pm25.describe())