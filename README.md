# AirAware: Air-Quality Forecasting and Data Science

AirAware is an exploratory data science project investigating
PM2.5 air pollution in New Delhi, India. It combines historical
air-quality observations with weather estimates, performs exploratory
analysis, and evaluates machine-learning forecasts.

The project emphasises reproducibility, chronological evaluation,
model comparison, and transparent reporting of limitations.

## Research Questions

These are the project's long-term research objectives:

1. How accurately can we forecast next-day PM2.5 levels?
2. Does adding weather data improve forecasting performance?
3. Does a model trained on one city generalise to another city?
4. How can we explain predictions and communicate uncertainty?

### Current Project Scope

To produce a working prototype within three days, the current
implementation focuses on predicting PM2.5 one hour ahead.

The original research questions are not all answered by this
prototype. Their current status is documented below.


| Research question | Current status |
|---|---|
| Next-day forecasting | Not evaluated; the current model predicts one hour ahead |
| Added value of weather features | Tested on a chronological holdout; adding weather slightly worsened Random Forest performance |
| Cross-city generalisation | Future work; only one sensor in one city has been analysed |
| Explainability and uncertainty | Random Forest feature importance is available; SHAP analysis and uncertainty estimation remain future work |


## Data Sources

- Air quality: OpenAQ API v3.
- Weather: Open-Meteo Historical Weather API.
- Location: New Delhi, India.
- PM2.5 sensor ID: 23534.
- Observation period: 15 July–13 October 2026.

The recent air-quality dataset contains 1,663 observations.
The weather dataset contains 2,184 hourly records. All 1,663
pollution records were matched to corresponding weather records.

## Data Quality

- 1,663 pollution observations were downloaded.
- No duplicate pollution timestamps were found in the recent dataset.
- Approximately 77% of expected hourly timestamps were covered.
- The largest gap between consecutive observations was 357 hours.
- The hourly modelling dataset contains 1,520 usable examples.

Missing measurement periods were not filled with invented values.
The project preserves these limitations when preparing the data.

## Exploratory Data Analysis

The initial Pearson correlations with PM2.5 were:

| Variable | Pearson correlation |
|---|---:|
| Temperature | 0.109 |
| Relative humidity | -0.230 |
| Precipitation | -0.115 |
| Wind speed | -0.235 |

These are observational associations, not causal effects. The results
may be affected by temporal patterns, missing observations, and the
limited study period.

## Forecasting Methodology

### Target

Predict the next hour's PM2.5 concentration.

### Features

- Current PM2.5 measurement.
- Previous-hour PM2.5 measurement.
- Three-hour rolling PM2.5 average.
- Temperature and relative humidity.
- Precipitation and wind speed.
- Wind direction.
- Local hour and day of the week.

### Models

1. Persistence baseline: use the current PM2.5 observation as the
   next-hour forecast.
2. Random Forest regression: use recent pollution, weather, and
   time features.

The dataset was split chronologically: the earliest 80% of usable
examples were used for training, and the latest 20% were held out
for testing. This produced 1,216 training examples and 304 test
examples.

### Initial Test Results

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistence baseline | 6.467 | 9.352 |
| Random Forest | 8.550 | 12.237 |

The persistence baseline outperformed Random Forest on this
particular test set. This shows that adding model complexity did
not automatically improve forecast accuracy.

The test period covered September–October 2026. The results are
preliminary and have not been validated across multiple cities or
independent monitoring stations.

## Explainability

The Random Forest ranked current PM2.5 as its most important feature,
with a reported importance of approximately 0.794.

Feature importance describes how the fitted model uses its inputs.
It does not establish causation or quantify prediction uncertainty.

SHAP-based explanations and calibrated uncertainty estimates have
not yet been implemented.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- OpenAQ API
- Open-Meteo Historical Weather API

## Run the Dashboard

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the interactive dashboard from the project root:

```powershell
python -m streamlit run app/dashboard.py
```

The dashboard requires the relevant data and report files to exist
in the project directories.

## Important Files

- `app/dashboard.py` — interactive Streamlit dashboard.
- `data/raw/` — downloaded pollution and weather datasets.
- `data/processed/` — combined and prepared datasets.
- `reports/hourly_model_predictions.csv` — test predictions.
- `reports/feature_importance.csv` — Random Forest feature importance.
- `reports/initial_findings.md` — exploratory findings.
- `reports/baseline_comparison.csv` — daily baseline comparison.

## Limitations

- The study covers one sensor in one city.
- The observation period is approximately three months.
- Significant gaps exist in the pollution time series.
- Historical weather values are model-based estimates, not direct
  measurements at the pollution sensor.
- The model evaluation uses one chronological test split.
- The original next-day forecasting question remains untested.
- Weather ablation, cross-city validation, and uncertainty estimation
  remain incomplete.

AirAware is an educational research prototype. Its predictions
should not be used for official environmental assessments or
health decisions.

## Future Work

1. Compare models with and without weather features using identical
   chronological splits.
2. Evaluate next-day forecasts using suitable daily observations.
3. Test generalisation across multiple monitoring stations and cities.
4. Add SHAP explanations and quantify forecast uncertainty.
5. Use repeated chronological evaluation to assess robustness.
6. Extend the dataset and investigate seasonal patterns.

## Research Integrity

The reported results represent the experiments completed to date.
Uncompleted research questions are explicitly identified rather than
presented as established findings.
