# Two-minute demo

1. Run `streamlit run app.py`. The data is real (UCI Bike Sharing) and the forecast comes from the saved model files.
2. Pick total rentals and a 14-day horizon. Show history against the dashed forecast and the model name in the table (Ridge, chosen on validation).
3. Switch to casual riders. It's a noisier series and the test error is much higher (WAPE 0.43). Say so before anyone asks.
4. In the what-if card, set a capacity and push demand up 20%. Show the days over capacity and the unmet demand, and point out that these are your assumptions, not model output.
5. Open `reports/metrics.csv` or the notebook: baseline against the candidates, the temporal split, and the worst days at the end of October 2012.
6. Close with the limits: two years of data, no weather input, error grows with the horizon.
