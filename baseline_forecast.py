import pandas as pd
from pathlib import Path

# Load the daily dataset
df = pd.read_csv(
    "data/processed/daily_pm25_weather.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

df["pm25_daily_mean"] = pd.to_numeric(
    df["pm25_daily_mean"], errors="coerce"
)

# Yesterday's observed daily average predicts today's average.
# Missing calendar days stay missing; we do not fill them.
df["previous_day_pm25"] = (
    df["pm25_daily_mean"].shift(1)
)

# Keep only cases where today and yesterday both have
# adequate observed pollution data.
pairs = df.dropna(
    subset=["pm25_daily_mean", "previous_day_pm25"]
).copy()

pairs["forecast_pm25"] = pairs["previous_day_pm25"]

if len(pairs) < 10:
    print("Not enough consecutive eligible days yet.")
    print("Available forecast cases:", len(pairs))
    raise SystemExit(0)

# Chronological split: earliest 80% for development,
# latest 20% for the final baseline evaluation.
split_point = int(len(pairs) * 0.8)

development = pairs.iloc[:split_point].copy()
test = pairs.iloc[split_point:].copy()

# Calculate test-set errors
errors = test["pm25_daily_mean"] - test["forecast_pm25"]

mae = errors.abs().mean()
rmse = (errors.pow(2).mean()) ** 0.5

print("===== BASELINE FORECAST REPORT =====")
print("Eligible consecutive-day cases:", len(pairs))
print("Development cases:", len(development))
print("Test cases:", len(test))

print("\nTest period:")
print("From:", test["date"].min().date())
print("To:", test["date"].max().date())

print("\nTest performance:")
print("MAE:", round(mae, 2), "µg/m³")
print("RMSE:", round(rmse, 2), "µg/m³")

print("\nFirst test predictions:")
print(
    test[["date", "pm25_daily_mean", "forecast_pm25"]]
    .rename(columns={
        "date": "date",
        "pm25_daily_mean": "actual_pm25",
        "forecast_pm25": "predicted_pm25"
    })
    .head(10)
    .round(2)
    .to_string(index=False)
)

# Save test predictions
output = Path("reports/baseline_predictions.csv")
output.parent.mkdir(parents=True, exist_ok=True)

test[[
    "date", "pm25_daily_mean", "forecast_pm25"
]].to_csv(output, index=False)

print("\nPredictions saved to:", output)
