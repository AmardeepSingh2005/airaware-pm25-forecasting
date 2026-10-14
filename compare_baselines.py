import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data/processed/daily_pm25_weather.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

df["pm25_daily_mean"] = pd.to_numeric(
    df["pm25_daily_mean"], errors="coerce"
)

# Keep only consecutive days with valid daily averages
df["previous_day_pm25"] = df["pm25_daily_mean"].shift(1)

pairs = df.dropna(
    subset=["pm25_daily_mean", "previous_day_pm25"]
).copy()

# Chronological 80/20 split
split = int(len(pairs) * 0.8)
development = pairs.iloc[:split].copy()
test = pairs.iloc[split:].copy()

if development.empty or test.empty:
    raise SystemExit("Not enough data for comparison.")

# Model A: persistence forecast
test["persistence_prediction"] = test["previous_day_pm25"]

# Model B: mean estimated from development data only
training_mean = development["pm25_daily_mean"].mean()
test["mean_prediction"] = training_mean

def calculate_metrics(actual, predicted):
    errors = actual - predicted
    return {
        "MAE": errors.abs().mean(),
        "RMSE": (errors.pow(2).mean()) ** 0.5
    }

persistence = calculate_metrics(
    test["pm25_daily_mean"],
    test["persistence_prediction"]
)

mean_model = calculate_metrics(
    test["pm25_daily_mean"],
    test["mean_prediction"]
)

comparison = pd.DataFrame(
    [persistence, mean_model],
    index=["Persistence", "Training mean"]
)

print("===== BASELINE COMPARISON =====")
print("Development cases:", len(development))
print("Test cases:", len(test))
print("Test period:", test["date"].min().date(),
      "to", test["date"].max().date())

print("\nTraining-set mean PM2.5:", round(training_mean, 2))
print("\nTest metrics (µg/m³):")
print(comparison.round(2).to_string())

if persistence["MAE"] < mean_model["MAE"]:
    print("\nPersistence has lower MAE on this test set.")
elif persistence["MAE"] > mean_model["MAE"]:
    print("\nTraining mean has lower MAE on this test set.")
else:
    print("\nBoth baselines have equal MAE on this test set.")

output = Path("reports/baseline_comparison.csv")
output.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output)

print("\nSaved comparison to:", output)
