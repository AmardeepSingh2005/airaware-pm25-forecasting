
from pathlib import Path

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

THRESHOLDS = [
    0, 10, 15, 20, 25, 30, 35, 40,
    45, 50, 60, 75, 90, float("inf"),
]


def make_model():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def metrics(actual, predictions):
    return {
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": mean_squared_error(actual, predictions) ** 0.5,
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
    raise ValueError("Not enough data for rolling validation.")

target_gap_minutes = (
    df[TARGET_TIME] - df[TIME]
).dt.total_seconds() / 60

if not target_gap_minutes.eq(60).all():
    raise ValueError(
        "Not all target timestamps are exactly 60 minutes ahead."
    )

# Exclude rows whose immediately previous available observation
# was not exactly one hour earlier.
df["continuous_hour"] = (
    df[TIME].diff().dt.total_seconds().div(60).eq(60)
)

n = len(df)
block = int(n * 0.10)

# Four sequential, non-overlapping test periods.
test_starts = [
    int(n * 0.60),
    int(n * 0.70),
    int(n * 0.80),
    int(n * 0.90),
]

print("\n=== AIR AWARE: ROLLING-ORIGIN VALIDATION ===")
print(f"Usable observations: {n}")
print(f"Approximate rows per validation/test block: {block}")
print(f"Number of test periods: {len(test_starts)}")


# --------------------------------------------------
# 3. Run rolling-origin experiments
# --------------------------------------------------

fold_results = []
all_predictions = []

for fold_number, test_start_idx in enumerate(test_starts, start=1):

    validation_start_idx = test_start_idx - block
    test_end_idx = min(test_start_idx + block, n)

    validation_start_time = df.loc[validation_start_idx, TIME]
    test_start_time = df.loc[test_start_idx, TIME]

    # Training targets must be known before validation begins.
    train = df[
        (df[TIME] < validation_start_time)
        & (df[TARGET_TIME] < validation_start_time)
    ].copy()

    # Validation targets must be known before the test period.
    validation = df[
        (df[TIME] >= validation_start_time)
        & (df[TIME] < test_start_time)
        & (df[TARGET_TIME] < test_start_time)
        & df["continuous_hour"]
    ].copy()

    # Each test period is evaluated only on consecutive hourly rows.
    if test_end_idx < n:
        test_end_time = df.loc[test_end_idx, TIME]
        test = df[
            (df[TIME] >= test_start_time)
            & (df[TIME] < test_end_time)
            & df["continuous_hour"]
        ].copy()
    else:
        test = df[
            (df[TIME] >= test_start_time)
            & df["continuous_hour"]
        ].copy()

    if min(len(train), len(validation), len(test)) < 20:
        print(f"\nFold {fold_number}: skipped due to insufficient data.")
        continue

    # Fit using data available before the validation period.
    validation_model = make_model()
    validation_model.fit(train[FEATURES], train[TARGET])

    validation_actual = validation[TARGET].to_numpy()
    validation_current = validation["pm25_current"].to_numpy()
    validation_rf = validation_model.predict(validation[FEATURES])
    validation_persistence = validation_current.copy()

    # Select threshold using validation data only.
    threshold_candidates = []

    for threshold in THRESHOLDS:
        use_rf = validation_current < threshold

        predictions = validation_persistence.copy()
        predictions[use_rf] = validation_rf[use_rf]

        threshold_candidates.append({
            "threshold": threshold,
            **metrics(validation_actual, predictions),
        })

    best = min(
        threshold_candidates,
        key=lambda row: (
            row["MAE"],
            row["RMSE"],
            row["threshold"],
        ),
    )

    selected_threshold = best["threshold"]

    # Refit on all eligible data before the test period.
    pretest = df[
        (df[TIME] < test_start_time)
        & (df[TARGET_TIME] < test_start_time)
    ].copy()

    final_model = make_model()
    final_model.fit(pretest[FEATURES], pretest[TARGET])

    actual = test[TARGET].to_numpy()
    persistence = test["pm25_current"].to_numpy()
    random_forest = final_model.predict(test[FEATURES])

    hybrid = persistence.copy()
    use_rf_test = test["pm25_current"].to_numpy() < selected_threshold
    hybrid[use_rf_test] = random_forest[use_rf_test]

    fold_predictions = {
        "Persistence": persistence,
        "Random Forest": random_forest,
        "Hybrid": hybrid,
    }

    for name, predictions in fold_predictions.items():
        fold_results.append({
            "Fold": fold_number,
            "Model": name,
            "Test examples": len(test),
            "Selected threshold": selected_threshold,
            **metrics(actual, predictions),
        })

    all_predictions.append(
        pd.DataFrame({
            "Actual": actual,
            "Persistence": persistence,
            "Random Forest": random_forest,
            "Hybrid": hybrid,
        })
    )

    print(
        f"\nFold {fold_number}: "
        f"train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}, threshold={selected_threshold:g}"
    )


# --------------------------------------------------
# 4. Summarize fold-level results
# --------------------------------------------------

if not all_predictions:
    raise ValueError(
        "No valid folds completed. Inspect dataset continuity."
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
# 5. Pooled out-of-sample results
# --------------------------------------------------

pooled = pd.concat(all_predictions, ignore_index=True)

pooled_results = []

for name in ["Persistence", "Random Forest", "Hybrid"]:
    pooled_results.append({
        "Model": name,
        **metrics(pooled["Actual"], pooled[name]),
    })

pooled_df = (
    pd.DataFrame(pooled_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n=== POOLED ROLLING-ORIGIN RESULTS ===")
print(f"Total evaluated examples: {len(pooled)}")
print(
    pooled_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nLower MAE and RMSE are better.")
print(
    "Exploratory rolling-origin evaluation only. "
    "No production files were changed."
)
print(
    "Some later periods were examined during earlier experiments, "
    "so this is not a completely untouched external test."
)
