
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# Candidate thresholds are selected using validation data only.
THRESHOLDS = [
    0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90,
    float("inf"),
]


def mae(actual, predicted):
    return mean_absolute_error(actual, predicted)


def rmse(actual, predicted):
    return mean_squared_error(actual, predicted) ** 0.5


def make_model():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_predictions(actual, predictions):
    return {
        "MAE": mae(actual, predictions),
        "RMSE": rmse(actual, predictions),
    }


# --------------------------------------------------
# 2. Load and validate data
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
    raise ValueError("Insufficient data for chronological validation.")

# Confirm each row's target timestamp is exactly one hour ahead.
target_gap_minutes = (
    df[TARGET_TIME] - df[TIME]
).dt.total_seconds() / 60

if not target_gap_minutes.eq(60).all():
    raise ValueError(
        "Some targets are not exactly 60 minutes ahead. "
        "Investigate target construction before continuing."
    )

# Mark rows whose immediately preceding available row is one hour earlier.
gap_minutes = df[TIME].diff().dt.total_seconds() / 60
df["continuous_hour"] = gap_minutes.eq(60)


# --------------------------------------------------
# 3. Chronological train / validation / test split
# --------------------------------------------------

n = len(df)
train_end = int(n * 0.60)
validation_end = int(n * 0.80)

validation_start_time = df.loc[train_end, TIME]
test_start_time = df.loc[validation_end, TIME]

# Purge training targets that reach the validation period.
train = df[
    (df[TIME] < validation_start_time)
    & (df[TARGET_TIME] < validation_start_time)
].copy()

# Validation targets must finish before the test period starts.
validation = df[
    (df[TIME] >= validation_start_time)
    & (df[TIME] < test_start_time)
    & (df[TARGET_TIME] < test_start_time)
    & df["continuous_hour"]
].copy()

# Final period: evaluate only consecutive hourly examples.
test = df[
    (df[TIME] >= test_start_time)
    & df["continuous_hour"]
].copy()

if len(train) < 50 or len(validation) < 20 or len(test) < 20:
    raise ValueError(
        "Too few observations in one of the split periods. "
        "Inspect the timestamp gaps and split sizes."
    )

print("\n=== CHRONOLOGICAL VALIDATION SETUP ===")
print(f"Total usable rows: {n}")
print(f"Training rows: {len(train)}")
print(f"Continuous validation rows: {len(validation)}")
print(f"Continuous test rows: {len(test)}")
print(
    "The final test period is not used to select the threshold."
)


# --------------------------------------------------
# 4. Train on the training period
# --------------------------------------------------

validation_model = make_model()
validation_model.fit(train[FEATURES], train[TARGET])

validation_actual = validation[TARGET].to_numpy()
validation_current = validation["pm25_current"].to_numpy()

validation_rf = validation_model.predict(validation[FEATURES])
validation_persistence = validation_current.copy()

print("\n=== VALIDATION: BASELINE VS RANDOM FOREST ===")

validation_comparison = pd.DataFrame([
    {
        "Model": "Persistence",
        **evaluate_predictions(
            validation_actual, validation_persistence
        ),
    },
    {
        "Model": "Random Forest",
        **evaluate_predictions(
            validation_actual, validation_rf
        ),
    },
])

print(
    validation_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)


# --------------------------------------------------
# 5. Select threshold on validation data only
# --------------------------------------------------

threshold_results = []

for threshold in THRESHOLDS:
    use_rf = validation_current < threshold

    hybrid_predictions = validation_persistence.copy()
    hybrid_predictions[use_rf] = validation_rf[use_rf]

    threshold_results.append({
        "Threshold": threshold,
        "RF examples": int(use_rf.sum()),
        "Persistence examples": int((~use_rf).sum()),
        **evaluate_predictions(
            validation_actual, hybrid_predictions
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

# Prefer the simpler persistence-only rule when validation MAE ties.
threshold_df = threshold_df.sort_values(
    ["MAE", "RMSE", "Threshold"],
    ascending=[True, True, True],
).reset_index(drop=True)

best_threshold = threshold_df.iloc[0]["Threshold"]

print("\n=== THRESHOLD SELECTION (VALIDATION ONLY) ===")
print(
    threshold_df.head(8).to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print(f"\nSelected threshold: {best_threshold:g} µg/m³")


# --------------------------------------------------
# 6. Refit using available pre-test data
# --------------------------------------------------

# Do not train on any target timestamp at or beyond test start.
pretest = df[
    (df[TIME] < test_start_time)
    & (df[TARGET_TIME] < test_start_time)
].copy()

final_model = make_model()
final_model.fit(pretest[FEATURES], pretest[TARGET])

test_actual = test[TARGET].to_numpy()
test_current = test["pm25_current"].to_numpy()

test_persistence = test_current.copy()
test_rf = final_model.predict(test[FEATURES])

test_hybrid = test_persistence.copy()
use_rf_on_test = test_current < best_threshold
test_hybrid[use_rf_on_test] = test_rf[use_rf_on_test]


# --------------------------------------------------
# 7. Final-period comparison
# --------------------------------------------------

final_results = pd.DataFrame([
    {
        "Model": "Persistence baseline",
        **evaluate_predictions(test_actual, test_persistence),
    },
    {
        "Model": "Random Forest",
        **evaluate_predictions(test_actual, test_rf),
    },
    {
        "Model": "Hybrid (validation-selected threshold)",
        **evaluate_predictions(test_actual, test_hybrid),
    },
]).sort_values("MAE").reset_index(drop=True)

print("\n=== FINAL-PERIOD EXPLORATORY EVALUATION ===")
print(f"Test examples: {len(test)}")
print(f"Selected threshold: {best_threshold:g} µg/m³")
print(f"Random Forest used for: {use_rf_on_test.sum()} examples")
print(f"Persistence used for: {(~use_rf_on_test).sum()} examples")

print(
    final_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nLower MAE and RMSE are better.")
print(
    "Exploratory evaluation only. The existing production "
    "model, reports, and dashboard were not changed."
)
