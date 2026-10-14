import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the prepared hourly forecasting dataset
df = pd.read_csv(
    "data/processed/hourly_forecast_dataset.csv"
)

df["origin_time_utc"] = pd.to_datetime(
    df["origin_time_utc"], utc=True, errors="coerce"
)

pollution_features = [
    "pm25_current",
    "pm25_previous_hour",
    "pm25_mean_last_3h",
    "forecast_hour_local",
    "forecast_day_of_week_local",
]

weather_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

target = "target_pm25_next_hour"
all_features = pollution_features + weather_features

# Use exactly the same complete cases for both models
df = df.dropna(
    subset=["origin_time_utc", target] + all_features
)
df = df.sort_values("origin_time_utc").reset_index(drop=True)

# Chronological 80/20 split
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

y_train = train[target]
y_test = test[target]

print("===== CONTROLLED WEATHER ABLATION =====")
print("Total examples:", len(df))
print("Training examples:", len(train))
print("Test examples:", len(test))
print("Test period:", test["origin_time_utc"].min())
print("To:", test["origin_time_utc"].max())

def evaluate(name, features):
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[features], y_train)
    predicted = model.predict(test[features])

    mae = mean_absolute_error(y_test, predicted)
    rmse = mean_squared_error(y_test, predicted) ** 0.5

    print(f"\n{name}")
    print("Features:", len(features))
    print(f"MAE:  {mae:.3f} µg/m³")
    print(f"RMSE: {rmse:.3f} µg/m³")

    return {
        "model": name,
        "feature_count": len(features),
        "MAE": mae,
        "RMSE": rmse,
    }

results = [
    evaluate("Random Forest without weather", pollution_features),
    evaluate("Random Forest with weather", all_features),
]

comparison = pd.DataFrame(results)

output = Path("reports/weather_ablation_comparison.csv")
output.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output, index=False)

print("\n===== COMPARISON =====")
print(comparison.round(3).to_string(index=False))

mae_without = results[0]["MAE"]
mae_with = results[1]["MAE"]

if mae_with < mae_without:
    print("\nWeather features improved MAE on this test set.")
elif mae_with > mae_without:
    print("\nWeather features worsened MAE on this test set.")
else:
    print("\nMAE was equal for both models.")

print("\nSaved results to:", output)
