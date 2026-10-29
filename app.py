"""DemandLab - demand forecast report. Run: streamlit run app.py (needs `python -m demandlab.train` first)."""
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from demandlab.predict import forecast, load_bundle

st.set_page_config(page_title="Forecast · DemandLab", page_icon=str(Path(__file__).parent / "assets" / "favicon.png"), layout="wide")
ui.inject_css()

with st.sidebar:
    ui.brand("Capital Bikeshare, Washington D.C.")
    st.subheader("Series")
    series = st.selectbox("Series", list(ui.SERIES), format_func=ui.SERIES.get, label_visibility="collapsed")
    st.subheader("Window")
    horizon = st.slider("Forecast horizon, days", 7, 28, 14)
    history_days = st.slider("History shown, days", 30, 365, 120)

try:
    bundle = load_bundle(series)
except FileNotFoundError:
    ui.note("error", "No trained model.", "Run  python -m demandlab.train  once, then reload this page.")
    st.stop()

with st.spinner("Running the forecast..."):
    hist, fc = bundle["history"], forecast(bundle, horizon)
metrics, results = ui.load_reports()

# ---- the one finding, built from the forecast and from the measured test error ------------------
avg_fc, avg_prev = float(fc.mean()), float(hist.iloc[-horizon:].mean())
chg = avg_fc / avg_prev - 1
dek = (f"Expect about <b>{avg_fc:,.0f} rentals a day</b> over the next {horizon} days, <b>{chg:+.0%}</b> against the last {horizon} days.")
rec = (results or {}).get("series", {}).get(series, {}).get("recursive_28d_test")
if rec:
    dek += (f" At 28 days out this model's error was <b>{rec['chosen']['WAPE']:.0%}</b> on the test period (WAPE), "
            f"against {rec['baseline']['WAPE']:.0%} for repeating last week.")
table = pd.DataFrame({"forecast": fc.round(0)})
t_col, a_col = st.columns([5, 1.4])
with t_col:
    ui.title("Demand forecast", dek, f"{ui.SERIES[series]} · {hist.index[0]:%d %b %Y} to {hist.index[-1]:%d %b %Y} · UCI Bike Sharing")
a_col.download_button("Forecast (.csv)", table.to_csv(index_label="date"), f"forecast_{series}.csv", "text/csv", width="stretch")

# ---- report card: hero number + legend table, premises, chart -------------------------------------
with st.container(border=True):
    ui.card_header(f"Rentals per day: {ui.SERIES[series].lower()}", f"last {history_days} days and next {horizon}")
    left, right = st.columns([1, 2.4])
    with left:
        rows = [(ui.INK, f"Actual, last {horizon} days", f"{hist.index[-horizon]:%d %b} to {hist.index[-1]:%d %b}", avg_prev),
                (ui.FC, f"Forecast, next {horizon} days", f"{fc.index[0]:%d %b} to {fc.index[-1]:%d %b}", avg_fc)]
        st.markdown(ui.hero_and_legend(horizon, avg_fc, avg_prev, rows) + ui.premises(bundle, results, horizon, "desk"), unsafe_allow_html=True)
    with right:
        st.plotly_chart(ui.forecast_figure(hist, fc, history_days), width="stretch", config={"displayModeBar": False})
        st.markdown("<div class='footnote'>The model also reads weekday, month and the long-run trend, not only recent demand, "
                    "so its first forecast day can sit well above the last actuals.</div>", unsafe_allow_html=True)
    st.markdown(ui.premises(bundle, results, horizon, "mob"), unsafe_allow_html=True)  # shown on phones, after the chart

# ---- how wrong can it be -----------------------------------------------------------------------------
with st.container(border=True):
    ui.card_header("How wrong can it be?", "last 90 days of data, never used for choosing the model")
    if metrics is None:
        ui.note("empty", "No results table yet.", "reports/metrics.csv is missing. Run  python -m demandlab.train.")
    else:
        st.markdown(ui.model_table(metrics, series, bundle["model_name"]), unsafe_allow_html=True)
        st.markdown("<div class='footnote ruler-note'>Ruler: each dot is a model's test error from 0 to 0.6; the tick is the baseline. Left of the tick beats repeating last week.</div>"
                    "<div class='footnote'>WAPE is the sum of absolute errors divided by the sum of actual rentals. The model was chosen on validation error only; "
                    "the test column is one honest look. The test period is autumn into winter, so errors run higher than in validation.</div>", unsafe_allow_html=True)

# ---- what if ---------------------------------------------------------------------------------------------
with st.container(border=True):
    ui.card_header("What if demand or capacity changes?", "your assumptions, not model output")
    default_cap = int(hist.iloc[-90:].quantile(0.9))
    c1, c2 = st.columns(2)
    capacity = c1.number_input("Daily capacity, rentals", 0, 100_000, default_cap)
    uplift = c2.slider("Demand against the forecast, %", -30, 50, 0)
    sim = fc * (1 + uplift / 100)
    over = (sim - capacity).clip(lower=0)
    st.markdown(f"<div class='assume'>Assuming capacity <b>{capacity:,.0f} a day</b> and demand <b>{uplift:+d}%</b> against the forecast.</div>", unsafe_allow_html=True)
    st.markdown(ui.scenario_strip(int((over > 0).sum()), horizon, float(over.sum()), f"{sim.max() / capacity * 100:.0f}% of capacity" if capacity else "n/a"),
                unsafe_allow_html=True)
    st.plotly_chart(ui.scenario_figure(sim, capacity), width="stretch", config={"displayModeBar": False})
    with st.expander("Day by day"):
        day = pd.DataFrame({"forecast": fc.round(0), "scenario": sim.round(0), "capacity": capacity, "unmet": over.round(0)})
        day.index = day.index.date
        st.dataframe(day, width="stretch")
