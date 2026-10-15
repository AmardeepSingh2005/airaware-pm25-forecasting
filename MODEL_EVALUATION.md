# AirAware — Model Evaluation Report

## 1. Project Overview

**AirAware** is a PM2.5 analysis and short-term forecasting project focused on New Delhi, India. It investigates whether machine-learning models and weather information can improve one-hour-ahead PM2.5 forecasts.

The project evaluates forecasting accuracy against simple baseline methods rather than assuming that a more complex model will perform better.

### Research questions

1. Can PM2.5 be forecast one hour ahead?
2. Does adding weather information improve forecast accuracy?
3. Do more complex models outperform a persistence baseline?
4. Does model performance change across different PM2.5 concentration ranges?
5. Can a threshold-based hybrid strategy improve on persistence?

## 2. Dataset and Preparation

The processed forecasting dataset contains **1,520 usable examples**.

The feature set includes:

- Current PM2.5 concentration
- Previous-hour PM2.5 concentration
- Trailing three-hour mean PM2.5
- Temperature
- Relative humidity
- Precipitation
- Wind speed
- Wind direction
- Forecast hour in local time
- Forecast day of the week in local time

The target is `target_pm25_next_hour`, representing PM2.5 at the target timestamp one hour after the input timestamp.

### Target validation

The target timestamp diagnostic found:

| Check | Result |
|---|---:|
| Rows with valid target timestamps | 1,520 |
| Targets exactly 60 minutes after input time | 1,520 |
| Targets with other timestamp gaps | 0 |
| Target values matched to a row in the filtered dataset | 1,476 |
| Matching values among matched rows | 1,476 |
| Mismatched values among matched rows | 0 |
| Unmatched target timestamps | 44 |

All matched target values agreed with the recorded PM2.5 value at the corresponding target timestamp. The 44 unmatched timestamps could not be matched to rows in the filtered dataset; their precise causes have not yet been established.

The continuity diagnostic also identified missing intervals between observations. Consequently, a one-hour target horizon does not guarantee that every consecutive available observation is exactly one hour apart.

## 3. Evaluation Methodology

Several evaluation procedures were used during experimentation.

### Chronological holdout

The original model comparison used an 80/20 chronological split:

- Training examples: 1,216
- Test examples: 304

The earlier observations were used for training, and the later observations were used for testing.

### Consecutive-hour evaluation

A stricter evaluation retained test rows whose immediately preceding available observation was exactly 60 minutes earlier.

- Eligible test examples: 297
- Excluded test examples: 7

The persistence baseline and two Random Forest variants were compared on the same 297 examples.

### Rolling-origin evaluation

Four sequential test periods were used to assess forecasting performance over time. The hybrid strategy selected its switching threshold using earlier validation data within each fold.

The pooled rolling-origin results contain 594 evaluated examples.

### Shared-sample simple-baseline evaluation

The rolling mean, linear trend, and previous-day seasonal baseline were evaluated on common eligible examples within each fold. The pooled comparison contains 525 examples.

**Important:** Results from these different evaluation procedures have different sample sizes and eligibility rules. Their metrics should not be directly compared as though they were calculated on one identical test set.

## 4. Machine-Learning Model Comparison

The following results use the original chronological test set of 304 examples.

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistence baseline | **6.467** | **9.352** |
| Random Forest — PM2.5 history only | 8.169 | 11.632 |
| Random Forest — all features | 8.550 | 12.237 |
| Extra Trees | 8.673 | 12.578 |
| Hist Gradient Boosting | 8.772 | 12.985 |
| Gradient Boosting | 9.164 | 13.461 |

Lower MAE and RMSE indicate better predictive accuracy.

The persistence baseline outperformed every tested machine-learning model on this evaluation set.

### Feature ablation

The Random Forest using only three PM2.5 history features achieved an MAE of 8.169, compared with 8.550 when weather and time features were included.

This is approximately a 4.5% reduction in MAE relative to the full-feature Random Forest. However, both models performed worse than persistence.

These findings do not establish that weather information is universally unhelpful. They show that the tested weather and time features did not improve the full Random Forest's accuracy on this particular test set.

## 5. Consecutive-Hour Model Evaluation

On the 297 consecutive hourly test examples:

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistence baseline | **6.502** | **9.393** |
| Random Forest — PM2.5 history only | 8.154 | 11.655 |
| Random Forest — all features | 8.534 | 12.258 |

Persistence remained the best of these three methods after restricting evaluation to consecutive hourly observations.

## 6. Error Analysis by Current PM2.5 Concentration

The following analysis groups the 297 test examples using the current PM2.5 reading, which is available at forecast time.

The ranges below are exploratory concentration bands, not official AQI categories.

| Current PM2.5 band (µg/m³) | Examples | Persistence MAE | Best tested ML MAE |
|---|---:|---:|---:|
| Below 30 | 45 | 4.422 | **3.793** — RF, all features |
| 30 to below 60 | 148 | **5.669** | 6.464 — RF, all features |
| 60 to below 90 | 87 | **7.299** | 10.174 — RF, history only |
| 90 or higher | 17 | **15.176** | 22.258 — RF, history only |

Random Forest performed better than persistence in the lowest concentration band. Persistence performed better in the remaining bands.

The highest-concentration band contains only 17 examples, so its metrics are particularly uncertain. The apparent advantage at low concentrations is exploratory and requires validation on additional data before being used to select a production model.

## 7. Hybrid Forecasting Experiment

An experimental hybrid used Random Forest when current PM2.5 was below a selected threshold and persistence otherwise.

### Single validation-selected threshold

A threshold of 25 µg/m³ was selected from validation data.

On the subsequent 298-example evaluation segment:

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Hybrid | **6.428** | **9.342** |
| Persistence | 6.480 | 9.377 |
| Random Forest | 8.532 | 12.261 |

The hybrid achieved a small improvement on this segment. However, the period had already been examined during earlier development, so this result should not be treated as a completely independent final test.

### Rolling-origin hybrid evaluation

Across four test periods, the selected threshold varied between 0, 25, 35, and 50 µg/m³.

| Model | Pooled MAE (µg/m³) | Pooled RMSE (µg/m³) |
|---|---:|---:|
| Persistence | **6.227** | **8.941** |
| Hybrid | 6.309 | 8.946 |
| Random Forest | 7.147 | 10.371 |

The hybrid did not outperform persistence on the pooled rolling-origin evaluation. The changing thresholds also indicate that the switching rule was not stable across periods.

The hybrid strategy is therefore not recommended for production deployment based on the current evidence.

## 8. Alternative Simple Baselines

Three alternative approaches were compared against persistence over 525 common test examples.

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistence | **6.238** | **8.982** |
| Three-hour rolling mean | 7.668 | 11.209 |
| Linear trend extrapolation | 8.710 | 12.261 |
| 24-hour seasonal persistence | 15.638 | 21.007 |

Persistence also outperformed these alternatives in each of the four reported test periods.

The previous-day seasonal baseline performed particularly poorly. In the current dataset, a reading from the same clock hour on the previous day was less useful than the current reading for predicting the next-hour value.

These metrics use 525 examples because all methods were compared on the same eligible rows within each fold. They should not be substituted for the 594-example rolling-origin model comparison.

## 9. Findings

The experiments support the following conclusions:

1. **Persistence is the strongest tested forecasting method.** It outperformed the machine-learning models and alternative simple baselines under the evaluation procedures used.
2. **Additional feature complexity did not improve overall accuracy.** Weather and time features did not make the tested Random Forest better than its PM2.5-history-only variant.
3. **Performance varies with concentration.** Random Forest showed a potential benefit at low current concentrations, while persistence was stronger at higher concentrations.
4. **The hybrid strategy is not sufficiently reliable for production.** Its improvement on one evaluation segment was small, and rolling-origin results did not confirm an overall benefit.
5. **Evaluation quality matters.** Timestamp validation, continuity checks, feature ablation, and multiple chronological test periods provided more useful evidence than choosing a model based on complexity alone.

## 10. Limitations

- The dataset represents a limited observation period and one monitoring location.
- Missing readings and large gaps reduce the number of eligible consecutive hourly examples.
- Results may change across seasons, locations, sensor conditions, and pollution episodes.
- Some test periods were examined repeatedly during model development; a future period or external dataset should be reserved for independent evaluation.
- The current results do not establish that the models generalize to other cities.
- Forecast errors and performance differences are estimates from the available sample, not guarantees of future accuracy.
- Error differences within individual concentration bands may be unstable, especially for bands with few observations.

## 11. Conclusion

AirAware demonstrates an end-to-end workflow for PM2.5 data preparation, exploratory analysis, next-hour forecasting, model comparison, feature ablation, and chronological evaluation.

Although several machine-learning methods were tested, none consistently outperformed the persistence baseline on the evaluated dataset. The appropriate current benchmark is therefore persistence, with the Random Forest and hybrid approaches remaining experimental.

The next priority is to collect more continuous observations, preserve an untouched future test period, and repeat evaluation as new data becomes available before considering changes to the production forecasting method.

---

*This report summarizes the experimental outputs currently available in the AirAware project. It does not claim that machine learning improves on persistence, and it does not describe the existing production model or dashboard as having been changed by these experiments.*