# 2-minute demo script
1. (0:00) `streamlit run app.py`. Say: real public data (UCI Bike Sharing), forecasts come from the trained model files.
2. (0:20) Pick `total`, horizon 14: history vs dashed forecast. Mention the model name shown (Ridge, chosen on validation).
3. (0:50) Switch to `casual`: noisier series, higher error (WAPE 0.43 on test) - be upfront about it.
4. (1:10) Simulation: set capacity, add +20% uplift, show days over capacity and unmet demand. Stress: these are YOUR assumptions.
5. (1:35) Open `reports/metrics.csv` / notebook: baseline vs models, temporal split, worst days (storm period).
6. (1:55) Limits: 2 years of data, no weather, errors grow with horizon.
