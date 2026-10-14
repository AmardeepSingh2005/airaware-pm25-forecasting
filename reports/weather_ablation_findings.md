# Weather Feature Ablation Experiment

## Research question

Does adding weather data improve forecasting performance?

## Experimental design

Two Random Forest regressors were compared:

- Model A: recent PM2.5 readings and time features.
- Model B: the same features plus temperature, humidity,
  precipitation, wind speed, and wind direction.

Both models used the same 1,216 training examples and 304
chronologically held-out test examples.

The target was PM2.5 concentration one hour ahead.
Performance was assessed using MAE and RMSE, in micrograms
per cubic metre (µg/m³). Lower scores are better.

## Results

| Model | Features | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest without weather | 5 | 8.366 | 11.999 |
| Random Forest with weather | 10 | 8.552 | 12.230 |

## Interpretation

Adding weather features did not improve performance on this
test set. MAE increased by approximately 2.2%, while RMSE
increased by approximately 1.9%.

The persistence baseline from the initial model evaluation
achieved MAE 6.467 and RMSE 9.352 µg/m³, outperforming both
Random Forest variants in their respective evaluations.

A possible explanation is that recent pollution measurements
provide a strong short-term predictive signal. This remains
a hypothesis, not an established cause.

## Limitations

- Evaluation used one chronological holdout period.
- The dataset covers one monitoring sensor and roughly three months.
- Hourly observations have substantial gaps.
- Weather inputs are historical model-based estimates.
- The experiment does not establish that weather is unhelpful
  for all models, locations, or forecasting horizons.

## Next experiments

1. Compare a model using only the current PM2.5 observation
   with models using additional pollution-history features.
2. Evaluate next-day forecasts separately.
3. Test on a longer dataset and additional monitoring stations.
4. Use repeated chronological evaluation to assess robustness.
