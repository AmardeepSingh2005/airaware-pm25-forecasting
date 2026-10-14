
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the combined dataset
df = pd.read_csv(
    "data/processed/pm25_weather_combined.csv"
)

# Ensure values are numeric
columns = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=columns)

# Weather variables to compare against PM2.5
weather_variables = [
    ("temperature_2m", "Temperature (°C)"),
    ("relative_humidity_2m", "Relative humidity (%)"),
    ("precipitation", "Precipitation (mm)"),
    ("wind_speed_10m", "Wind speed (km/h)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (column, label) in zip(axes.flat, weather_variables):
    ax.scatter(
        df[column],
        df["pm25"],
        alpha=0.3,
        s=12
    )

    correlation = df["pm25"].corr(df[column])

    ax.set_title(f"{label}\nPearson r = {correlation:.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.grid(True, alpha=0.2)

fig.suptitle(
    "PM2.5 and Weather Relationships — New Delhi",
    fontsize=14
)

fig.tight_layout()

output = Path("reports/pm25_weather_relationships.png")
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=200, bbox_inches="tight")

print("Charts saved to:", output)
print("Observations plotted:", len(df))

plt.show()
