import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "data/processed/hourly_forecast_dataset.csv"

df = pd.read_csv(DATA_FILE)
df["origin_time_utc"] = pd.to_datetime(
    df["origin_time_utc"], utc=True, errors="coerce"
)

features = [
    "pm25_current",
    "pm25_previous_hour",
    "pm25_mean_last_3h",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "forecast_hour_local",
    "forecast_day_of_week_local",
]
target = "target_pm25_next_hour"

df = df.dropna(
    subset=["origin_time_utc", target] + features
).sort_values("origin_time_utc").reset_index(drop=True)

if len(df) < 100:
    raise SystemExit(f"Not enough examples: {len(df)}")

# Preserve chronological order: oldest 80% for training,
# newest 20% for testing.
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Baseline: current pollution predicts the next hour.
baseline_predictions = X_test["pm25_current"]

# Train Random Forest.
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
rf_predictions = model.predict(X_test)

def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

print("===== AIR AWARE: HOURLY FORECAST =====")
print("Usable examples:", len(df))
print("Training examples:", len(train))
print("Test examples:", len(test))
print("Test start:", test["origin_time_utc"].min())
print("Test end:", test["origin_time_utc"].max())

print("\nMetrics (µg/m³; lower is better):")
report("Persistence baseline", y_test, baseline_predictions)
report("Random Forest", y_test, rf_predictions)

# Save test predictions.
output = Path("reports/hourly_model_predictions.csv")
output.parent.mkdir(parents=True, exist_ok=True)

predictions = test[
    ["origin_time_utc", "target_time_utc", target]
].copy()
predictions["baseline_prediction"] = baseline_predictions.to_numpy()
predictions["random_forest_prediction"] = rf_predictions
predictions.to_csv(output, index=False)

# Save model feature importance.
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv("reports/feature_importance.csv", index=False)

print("\nTop five features:")
print(importance.head().to_string(index=False))
print("\nPredictions saved to:", output)
print("Feature importance saved to: reports/feature_importance.csv")
