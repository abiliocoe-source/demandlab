---
language: en
license: other
library_name: scikit-learn
tags: [time-series, forecasting, demand-forecasting, tabular]
datasets: [uci-bike-sharing]
---
# DemandLab daily demand forecaster

**Status: NOT published to the Hugging Face Hub.** `hf_repo/` is only the prepared folder structure.

## What it is
Three small scikit-learn models (series `total`, `registered`, `casual`) forecasting daily bike rentals up to 28 days ahead. Chosen per series by validation WAPE among: seasonal-naive baseline, Ridge (alpha 1, 10), RandomForest, GradientBoosting. Selected: Ridge in all three series (`ridge_a10` for total/registered, `ridge_a1` for casual).

## Data
UCI Bike Sharing daily data, Washington D.C., 2011-2012 (see `data/SOURCE.md`; citation required). Split chronologically: train to 2012-07-04, validation to 2012-10-02, test 2012-10-03..2012-12-31. Final models are refit on train+validation.

## Features
Demand lagged by 7 and 14 days, mean of days t-13..t-7, day index, weekday, month (sin/cos), holiday flag. No weather. Forecast horizon 7 days with real history; longer horizons are recursive.

## Measured results (test, WAPE; lower is better)
| series | seasonal-naive | chosen model |
|---|---|---|
| total | 0.281 | 0.247 |
| registered | 0.277 | 0.231 |
| casual | 0.501 | 0.430 |

28-day recursive test WAPE (7 forecast origins x 28 days): total 0.206 vs baseline 0.318; registered 0.183 vs 0.314; casual 0.443 vs 0.606. Full tables: `reports/metrics.csv`, `reports/results.json`.

## Limitations and failure cases
- Test errors are about 2x validation errors for `total` (validation WAPE 0.110 vs test 0.247): the test period is autumn-to-winter, where demand falls, and the models learned a mostly rising pattern. Two years of data cannot teach a full seasonal cycle twice.
- Largest errors fall on 2012-10-29 and 2012-10-30 (actual 22 and 1,096 rentals vs ~6,400 predicted), and on 2012-12-26. A storm is a plausible explanation for the first two but this was not verified with the weather columns; the model has no weather input and cannot anticipate such events.
- Tree models were worse than Ridge here, probably because they cannot extrapolate the upward trend (hypothesis, not tested).
- Not suitable for operational decisions without more data and weather/event inputs.

## Use
`from demandlab.predict import load_bundle, forecast; forecast(load_bundle("total"), 14)`
