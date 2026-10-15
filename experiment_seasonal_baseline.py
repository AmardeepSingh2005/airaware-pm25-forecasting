
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def calculate_metrics(actual, predictions):
    return {
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": mean_squared_error(actual, predictions) ** 0.5,
    }


# --------------------------------------------------
# 1. Load and validate data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

required = [TIME, TARGET_TIME, TARGET] + FEATURES
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for column in [TIME, TARGET_TIME]:
    df[column] = pd.to_datetime(
        df[column], utc=True, errors="coerce"
    )

df = (
    df.dropna(subset=required)
    .sort_values(TIME)
    .drop_duplicates(subset=[TIME])
    .reset_index(drop=True)
)

if len(df) < 100:
    raise ValueError("Not enough valid observations.")

target_gap = (
    df[TARGET_TIME] - df[TIME]
).dt.total_seconds() / 60

if not target_gap.eq(60).all():
    raise ValueError(
        "Some target timestamps are not exactly one hour ahead."
    )

# Keep rows whose immediately preceding available observation
# is exactly one hour earlier.
df["continuous_hour"] = (
    df[TIME].diff().dt.total_seconds().div(60).eq(60)
)

# Historical lookup: PM2.5 readings indexed by their timestamps.
pm25_by_time = pd.Series(
    df["pm25_current"].to_numpy(),
    index=pd.DatetimeIndex(df[TIME]),
)

print("\n=== AIR AWARE: SEASONAL BASELINE EXPERIMENT ===")
print(f"Valid observations: {len(df)}")


# --------------------------------------------------
# 2. Create four chronological test periods
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
# 3. Evaluate baselines on common eligible rows
# --------------------------------------------------

for fold_number, start_idx in enumerate(test_starts, start=1):
    end_idx = min(start_idx + block_size, n)

    test = df.iloc[start_idx:end_idx].copy()

    # Restrict to consecutive hourly observations.
    test = test.loc[test["continuous_hour"]].copy()

    if test.empty:
        print(f"\nFold {fold_number}: no eligible rows.")
        continue

    # The forecast is for time t+1 hour.
    # The same clock hour on the previous day is t-23 hours.
    seasonal_source_times = test[TIME] - pd.Timedelta(hours=23)

    seasonal_predictions = seasonal_source_times.map(pm25_by_time)

    # All methods must be assessed on the same examples.
    eligible = seasonal_predictions.notna()

    test = test.loc[eligible].copy()
    seasonal_predictions = seasonal_predictions.loc[test.index]

    if len(test) < 20:
        print(
            f"\nFold {fold_number}: only {len(test)} "
            "common examples; skipping."
        )
        continue

    actual = test[TARGET].to_numpy(dtype=float)
    current = test["pm25_current"].to_numpy(dtype=float)
    previous = test["pm25_previous_hour"].to_numpy(dtype=float)
    rolling_mean = test["pm25_mean_last_3h"].to_numpy(dtype=float)

    # Baseline 1: persistence
    persistence = current.copy()

    # Baseline 2: average of the recent three hours
    three_hour_mean = rolling_mean.copy()

    # Baseline 3: extend the latest observed one-hour change
    linear_trend = np.maximum(
        0.0,
        current + (current - previous),
    )

    # Baseline 4: same clock hour from the previous day
    seasonal = seasonal_predictions.to_numpy(dtype=float)

    predictions_by_model = {
        "Persistence": persistence,
        "Three-hour rolling mean": three_hour_mean,
        "Linear trend extrapolation": linear_trend,
        "24-hour seasonal persistence": seasonal,
    }

    print(
        f"\nFold {fold_number}: "
        f"common test examples={len(test)}, "
        f"{test[TIME].min()} to {test[TIME].max()}"
    )

    pooled_predictions.append(
        pd.DataFrame({
            "Actual": actual,
            **predictions_by_model,
        })
    )

    for model_name, predictions in predictions_by_model.items():
        fold_results.append({
            "Fold": fold_number,
            "Model": model_name,
            "Test examples": len(test),
            **calculate_metrics(actual, predictions),
        })


# --------------------------------------------------
# 4. Results by test period
# --------------------------------------------------

if not pooled_predictions:
    raise ValueError(
        "No folds had enough common eligible examples."
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
# 5. Pooled comparison
# --------------------------------------------------

pooled_df = pd.concat(
    pooled_predictions,
    ignore_index=True,
)

model_names = [
    "Persistence",
    "Three-hour rolling mean",
    "Linear trend extrapolation",
    "24-hour seasonal persistence",
]

pooled_results = []

for model_name in model_names:
    pooled_results.append({
        "Model": model_name,
        **calculate_metrics(
            pooled_df["Actual"],
            pooled_df[model_name],
        ),
    })

results_df = (
    pd.DataFrame(pooled_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n=== POOLED ROLLING-ORIGIN RESULTS ===")
print(f"Total common test examples: {len(pooled_df)}")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nLower MAE and RMSE are better.")
print(
    "All baselines are evaluated on identical examples "
    "within each fold."
)
print(
    "The seasonal baseline uses only a historical reading "
    "from the same clock hour on the previous day."
)
print(
    "Exploratory results only. Production files were unchanged."
)
