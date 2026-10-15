
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_FILE = Path("data/processed/hourly_forecast_dataset.csv")

TIME = "origin_time_utc"
TARGET_TIME = "target_time_utc"
TARGET = "target_pm25_next_hour"

FEATURES = [
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


# --------------------------------------------------
# 2. Metric calculation
# --------------------------------------------------

def calculate_metrics(actual, predictions):
    return {
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": mean_squared_error(actual, predictions) ** 0.5,
    }


# --------------------------------------------------
# 3. Load and prepare data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = [TIME, TARGET_TIME, TARGET] + FEATURES

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

for column in [TIME, TARGET_TIME]:
    df[column] = pd.to_datetime(
        df[column],
        utc=True,
        errors="coerce",
    )

df = (
    df.dropna(subset=required_columns)
    .sort_values(TIME)
    .drop_duplicates(subset=[TIME])
    .reset_index(drop=True)
)

if len(df) < 100:
    raise ValueError(
        "Not enough valid observations for evaluation."
    )

# Confirm that every target is exactly one hour ahead.
target_gap_minutes = (
    df[TARGET_TIME] - df[TIME]
).dt.total_seconds() / 60

if not target_gap_minutes.eq(60).all():
    raise ValueError(
        "Some targets are not exactly 60 minutes ahead."
    )

# Only evaluate rows with an available observation exactly
# one hour before the current observation.
df["continuous_hour"] = (
    df[TIME]
    .diff()
    .dt.total_seconds()
    .div(60)
    .eq(60)
)

print("\n=== AIR AWARE: SIMPLE BASELINE EXPERIMENT ===")
print(f"Valid observations: {len(df)}")


# --------------------------------------------------
# 4. Define four chronological test periods
# --------------------------------------------------

n = len(df)
block_size = int(n * 0.10)

test_starts = [
    int(n * 0.60),
    int(n * 0.70),
    int(n * 0.80),
    int(n * 0.90),
]

fold_results = []
pooled_predictions = []


# --------------------------------------------------
# 5. Evaluate the baselines on each test period
# --------------------------------------------------

for fold_number, start_idx in enumerate(test_starts, start=1):

    end_idx = min(start_idx + block_size, n)

    # Use the same test rows for all four baselines.
    test = df.iloc[start_idx:end_idx].copy()

    # Restrict evaluation to consecutive hourly observations.
    test = test.loc[test["continuous_hour"]].copy()

    if test.empty:
        print(f"Fold {fold_number}: no eligible hourly examples.")
        continue

    actual = test[TARGET].to_numpy(dtype=float)

    current = test["pm25_current"].to_numpy(dtype=float)

    previous = test[
        "pm25_previous_hour"
    ].to_numpy(dtype=float)

    rolling_mean = test[
        "pm25_mean_last_3h"
    ].to_numpy(dtype=float)

    # Baseline 1: persistence.
    # Predict that the next-hour value remains unchanged.
    persistence = current.copy()

    # Baseline 2: recent three-hour rolling mean.
    # Uses the rolling mean available at forecast time.
    rolling_mean_prediction = rolling_mean.copy()

    # Baseline 3: linear trend extrapolation.
    # Extend the latest one-hour change by one more hour.
    # Clip negative forecasts to zero because PM2.5 cannot be negative.
    trend_prediction = np.maximum(
        0.0,
        current + (current - previous),
    )

    predictions_by_model = {
        "Persistence": persistence,
        "Three-hour rolling mean": rolling_mean_prediction,
        "Linear trend extrapolation": trend_prediction,
    }

    print(
        f"\nFold {fold_number}: "
        f"test rows={len(test)}, "
        f"{test[TIME].min()} to {test[TIME].max()}"
    )

    fold_data = pd.DataFrame({
        "Fold": fold_number,
        "Actual": actual,
        **predictions_by_model,
    })

    pooled_predictions.append(fold_data)

    for model_name, predictions in predictions_by_model.items():
        fold_results.append({
            "Fold": fold_number,
            "Model": model_name,
            "Test examples": len(test),
            **calculate_metrics(actual, predictions),
        })


# --------------------------------------------------
# 6. Results by test period
# --------------------------------------------------

if not pooled_predictions:
    raise ValueError(
        "No test periods had eligible consecutive hourly examples."
    )

fold_df = pd.DataFrame(fold_results)

print("\n=== RESULTS BY TEST PERIOD ===")
print(
    fold_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)


# --------------------------------------------------
# 7. Pooled out-of-sample comparison
# --------------------------------------------------

pooled_df = pd.concat(
    pooled_predictions,
    ignore_index=True,
)

pooled_results = []

for model_name in [
    "Persistence",
    "Three-hour rolling mean",
    "Linear trend extrapolation",
]:
    pooled_results.append({
        "Model": model_name,
        **calculate_metrics(
            pooled_df["Actual"],
            pooled_df[model_name],
        ),
    })

pooled_result_df = (
    pd.DataFrame(pooled_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n=== POOLED ROLLING-ORIGIN RESULTS ===")
print(f"Total evaluated examples: {len(pooled_df)}")

print(
    pooled_result_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nLower MAE and RMSE are better.")
print(
    "All methods use information available at forecast time."
)
print(
    "Exploratory results only: production models, reports, "
    "and dashboard were not changed."
)
