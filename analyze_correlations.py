
import pandas as pd

# Load the combined dataset
df = pd.read_csv(
    "data/processed/pm25_weather_combined.csv"
)

# Select the variables for our first analysis
columns = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]

analysis = df[columns].apply(
    pd.to_numeric, errors="coerce"
).dropna()

# Pearson measures linear relationships
pearson = analysis.corr(method="pearson")

# Spearman measures monotonic relationships
spearman = analysis.corr(method="spearman")

print("===== PEARSON CORRELATION =====")
print(pearson["pm25"].drop("pm25").round(3))

print("\n===== SPEARMAN CORRELATION =====")
print(spearman["pm25"].drop("pm25").round(3))

print("\nObservations analysed:", len(analysis))
