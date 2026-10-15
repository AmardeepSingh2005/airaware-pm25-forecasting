
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_FILE = Path("data/processed/hourly_forecast_dataset.csv")

TIME_COLUMN = "origin_time_utc"
TARGET_TIME_COLUMN = "target_time_utc"
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

LAG_FEATURES = [
    "pm25_current",
    "pm25_previous_hour",
    "pm25_mean_last_3h",
]


# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = [TIME_COLUMN, TARGET] + FEATURES

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[TIME_COLUMN] = pd.to_datetime(
    df[TIME_COLUMN],
    utc=True,
    errors="coerce",
)

# Keep valid observations and sort chronologically.
df = (
    df.dropna(subset=required_columns)
    .sort_values(TIME_COLUMN)
    .reset_index(drop=True)
)

if len(df) < 10:
    raise ValueError(
        "Not enough valid observations to run the experiment."
    )


# --------------------------------------------------
# 3. Verify timestamps and target-value alignment
# --------------------------------------------------

print("\n=== TIMESTAMP GAP DIAGNOSTIC ===")

if TARGET_TIME_COLUMN in df.columns:
    df[TARGET_TIME_COLUMN] = pd.to_datetime(
        df[TARGET_TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    gap_minutes = (
        df[TARGET_TIME_COLUMN] - df[TIME_COLUMN]
    ).dt.total_seconds() / 60

    print("Target timestamp column:", TARGET_TIME_COLUMN)
    print(
        "Rows with valid target timestamps:",
        gap_minutes.notna().sum(),
    )
    print(
        "Rows with exactly 60-minute targets:",
        (gap_minutes == 60).sum(),
    )
    print(
        "Rows with other target gaps:",
        (gap_minutes.notna() & (gap_minutes != 60)).sum(),
    )

    # Compare each target value with the current PM2.5 value
    # at the corresponding target timestamp.
    alignment = df[
        [TARGET_TIME_COLUMN, TARGET]
    ].merge(
        df[[TIME_COLUMN, "pm25_current"]],
        left_on=TARGET_TIME_COLUMN,
        right_on=TIME_COLUMN,
        how="left",
        validate="many_to_one",
    )

    matched = alignment["pm25_current"].notna()

    differences = (
        alignment.loc[matched, TARGET].astype(float)
        - alignment.loc[matched, "pm25_current"].astype(float)
    ).abs()

    print("\n=== TARGET VALUE ALIGNMENT CHECK ===")
    print(
        f"Targets matched to a valid row: "
        f"{matched.sum()} / {len(alignment)}"
    )
    print(
        f"Matching target values: "
        f"{(differences <= 1e-6).sum()} / {len(differences)}"
    )
    print(
        "Mismatched target values:",
        (differences > 1e-6).sum(),
    )
    print(
        "Unmatched target timestamps:",
        (~matched).sum(),
    )

else:
    print(
        f"Column '{TARGET_TIME_COLUMN}' not found. "
        "Available columns:",
        df.columns.tolist(),
    )
    print("Target-value alignment check skipped.")


# --------------------------------------------------
# 4. Chronological train/test split
# --------------------------------------------------

split = int(len(df) * 0.8)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# --------------------------------------------------
# 5. Check continuity in the test set
# --------------------------------------------------

test_gaps = (
    test[TIME_COLUMN]
    .diff()
    .dt.total_seconds()
    .div(60)
    .dropna()
)

print("\n=== TEST SET CONTINUITY CHECK ===")
print(
    "Consecutive one-hour gaps:",
    (test_gaps == 60).sum(),
)
print(
    "Gaps longer than one hour:",
    (test_gaps > 60).sum(),
)

if not test_gaps.empty:
    print(
        f"Maximum gap (minutes): {test_gaps.max():.0f}"
    )


# --------------------------------------------------
# 6. Define machine-learning models
# --------------------------------------------------

models = {
    "Random Forest (all features)": RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="huber",
        random_state=42,
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),
}


# --------------------------------------------------
# 7. Evaluate persistence baseline
# --------------------------------------------------

results = []

# Persistence predicts that the next-hour value will equal
# the current PM2.5 reading.
baseline_predictions = X_test["pm25_current"]

results.append({
    "Model": "Persistence baseline",
    "MAE": mean_absolute_error(
        y_test, baseline_predictions
    ),
    "RMSE": mean_squared_error(
        y_test, baseline_predictions
    ) ** 0.5,
})


# --------------------------------------------------
# 8. Evaluate models using all features
# --------------------------------------------------

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(
            y_test, predictions
        ),
        "RMSE": mean_squared_error(
            y_test, predictions
        ) ** 0.5,
    })


# --------------------------------------------------
# 9. Ablation: Random Forest using PM2.5 history only
# --------------------------------------------------

lag_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

lag_model.fit(train[LAG_FEATURES], y_train)

lag_predictions = lag_model.predict(test[LAG_FEATURES])

results.append({
    "Model": "Random Forest (PM2.5 history only)",
    "MAE": mean_absolute_error(
        y_test, lag_predictions
    ),
    "RMSE": mean_squared_error(
        y_test, lag_predictions
    ) ** 0.5,
})


# --------------------------------------------------
# 10. Fair comparison on consecutive hourly observations
# --------------------------------------------------

test_gap_minutes = (
    test[TIME_COLUMN]
    .diff()
    .dt.total_seconds()
    .div(60)
)

# The first test row has no preceding test row,
# so it is excluded from this strict continuity check.
continuous_mask = test_gap_minutes.eq(60)

continuous_test = test.loc[continuous_mask]
continuous_y = continuous_test[TARGET]

print("\n=== STRICT HOURLY EVALUATION ===")
print(
    f"Consecutive hourly test examples: "
    f"{len(continuous_test)}"
)
print(
    f"Excluded test examples: "
    f"{len(test) - len(continuous_test)}"
)

if not continuous_test.empty:
    # Persistence baseline
    persistence_pred = continuous_test["pm25_current"]

    # Random Forest using all features
    full_rf = models["Random Forest (all features)"]
    full_rf_pred = full_rf.predict(
        continuous_test[FEATURES]
    )

    # Random Forest using PM2.5 history only
    history_rf_pred = lag_model.predict(
        continuous_test[LAG_FEATURES]
    )

    strict_results = [
        {
            "Model": "Persistence baseline",
            "MAE": mean_absolute_error(
                continuous_y, persistence_pred
            ),
            "RMSE": mean_squared_error(
                continuous_y, persistence_pred
            ) ** 0.5,
        },
        {
            "Model": "Random Forest (all features)",
            "MAE": mean_absolute_error(
                continuous_y, full_rf_pred
            ),
            "RMSE": mean_squared_error(
                continuous_y, full_rf_pred
            ) ** 0.5,
        },
        {
            "Model": "Random Forest (PM2.5 history only)",
            "MAE": mean_absolute_error(
                continuous_y, history_rf_pred
            ),
            "RMSE": mean_squared_error(
                continuous_y, history_rf_pred
            ) ** 0.5,
        },
    ]

    strict_df = (
        pd.DataFrame(strict_results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    print("\nStrict hourly results (sorted by MAE):")
    print(
        strict_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

else:
    print(
        "Not enough consecutive hourly observations "
        "to evaluate the models."
    )


# --------------------------------------------------
# 11. Print full model comparison
# --------------------------------------------------

result_df = (
    pd.DataFrame(results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n=== AIR AWARE: MODEL EXPERIMENT ===")
print(f"Valid observations: {len(df)}")
print(f"Train examples: {len(train)}")
print(f"Test examples: {len(test)}")

print("\nFull model comparison (sorted by MAE):")
print(
    result_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nLower MAE and RMSE are better.")
print(
    "Experiments only: production models, reports, "
    "and dashboard were not changed."
)



# --------------------------------------------------
# 12. Error analysis by CURRENT PM2.5 concentration
# --------------------------------------------------

print("\n=== ERROR BY CURRENT PM2.5 LEVEL ===")

if continuous_test.empty:
    print("No consecutive hourly test examples available.")

else:
    actual = continuous_test[TARGET].to_numpy()

    predictions_by_model = {
        "Persistence baseline": (
            continuous_test["pm25_current"].to_numpy()
        ),
        "Random Forest (PM2.5 history only)": (
            lag_model.predict(continuous_test[LAG_FEATURES])
        ),
        "Random Forest (all features)": (
            models["Random Forest (all features)"].predict(
                continuous_test[FEATURES]
            )
        ),
    }

    # Use current PM2.5 to define the bands.
    # These are exploratory ranges, not official AQI categories.
    bins = [
        float("-inf"),
        30,
        60,
        90,
        float("inf"),
    ]

    labels = [
        "<30",
        "30–<60",
        "60–<90",
        "90+",
    ]

    band_results = []

    for model_name, predictions in predictions_by_model.items():
        evaluation = pd.DataFrame({
            "Current PM2.5": (
                continuous_test["pm25_current"].to_numpy()
            ),
            "Actual next-hour PM2.5": actual,
            "Predicted PM2.5": predictions,
        })

        evaluation["Pollution band"] = pd.cut(
            evaluation["Current PM2.5"],
            bins=bins,
            labels=labels,
            right=False,
        )

        evaluation["Absolute error"] = (
            evaluation["Actual next-hour PM2.5"]
            - evaluation["Predicted PM2.5"]
        ).abs()

        evaluation["Squared error"] = (
            evaluation["Actual next-hour PM2.5"]
            - evaluation["Predicted PM2.5"]
        ) ** 2

        grouped = (
            evaluation.groupby(
                "Pollution band",
                observed=True,
            )
            .agg(
                Samples=("Absolute error", "count"),
                MAE=("Absolute error", "mean"),
                MSE=("Squared error", "mean"),
            )
            .reset_index()
        )

        grouped["RMSE"] = grouped["MSE"] ** 0.5
        grouped["Model"] = model_name

        band_results.append(
            grouped.drop(columns=["MSE"])
        )

    if band_results:
        band_df = pd.concat(
            band_results,
            ignore_index=True,
        )

        band_df = band_df[
            ["Pollution band", "Model", "Samples", "MAE", "RMSE"]
        ]

        band_df = band_df.sort_values(
            ["Pollution band", "MAE"]
        )

        print(
            band_df.to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

        print(
            "\nBands are based on CURRENT PM2.5, which is available "
            "when making a forecast."
        )
        print(
            "Errors are calculated against ACTUAL next-hour PM2.5."
        )
        print(
            "Lower MAE and RMSE are better. "
            "Interpret bands with small sample counts cautiously."
        )

# --------------------------------------------------
# 13. Exploratory hybrid forecasting strategy
# --------------------------------------------------

print("\n=== EXPLORATORY HYBRID STRATEGY ===")

if continuous_test.empty:
    print("No consecutive hourly test examples available.")

else:
    # Predictions from the two candidate approaches.
    current_pm25 = continuous_test["pm25_current"].to_numpy()

    persistence_predictions = current_pm25.copy()

    rf_predictions = models[
        "Random Forest (all features)"
    ].predict(continuous_test[FEATURES])

    # Experimental decision rule:
    # Use Random Forest below 30 µg/m³.
    # Use persistence at 30 µg/m³ or above.
    use_random_forest = current_pm25 < 30

    hybrid_predictions = persistence_predictions.copy()
    hybrid_predictions[use_random_forest] = (
        rf_predictions[use_random_forest]
    )

    actual = continuous_test[TARGET].to_numpy()

    comparison = []

    for model_name, predictions in [
        ("Persistence baseline", persistence_predictions),
        ("Hybrid strategy (experimental)", hybrid_predictions),
    ]:
        comparison.append({
            "Model": model_name,
            "MAE": mean_absolute_error(actual, predictions),
            "RMSE": mean_squared_error(
                actual, predictions
            ) ** 0.5,
        })

    comparison_df = (
        pd.DataFrame(comparison)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    print(f"Evaluation examples: {len(continuous_test)}")
    print(
        f"Examples using Random Forest: "
        f"{use_random_forest.sum()}"
    )
    print(
        f"Examples using persistence: "
        f"{(~use_random_forest).sum()}"
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(
        "\nExploratory result only: the selection rule was "
        "informed by this test set and requires independent validation."
    )
