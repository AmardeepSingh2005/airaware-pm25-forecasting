import pandas as pd

file_path = "data/raw/pm25_new_delhi.csv"

df = pd.read_csv(file_path)

# Convert timestamps and pollutant values
dates = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
pm25 = pd.to_numeric(df["pm25"], errors="coerce")

print("----- DATA QUALITY REPORT -----")

print("Total records:", len(df))
print("Missing PM2.5 values:", pm25.isna().sum())
print("Duplicate timestamps:", dates.duplicated().sum())

print("\n----- DATE RANGE -----")
print("First timestamp:", dates.min())
print("Last timestamp:", dates.max())

print("\n----- PM2.5 STATISTICS -----")
print(pm25.describe())

print("\n----- MISSING VALUES BY COLUMN -----")
print(df.isna().sum())