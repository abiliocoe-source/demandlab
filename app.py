"""DemandLab - forecasting notebook. Run: streamlit run app.py (needs `python -m demandlab.train` first)."""
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from demandlab.predict import forecast, load_bundle

st.set_page_config(page_title="Notebook · DemandLab", page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
                   layout="wide")
ui.inject_css()

with st.sidebar:
    st.header("Bench")
    series = st.selectbox("Series", ["total", "registered", "casual"])
    horizon = st.slider("Forecast horizon (days)", 7, 28, 14)
    history_days = st.slider("History shown (days)", 30, 365, 120)

try:
    bundle = load_bundle(series)
except FileNotFoundError:
    ui.notice("error", "No trained model on the bench.", "Run  python -m demandlab.train  once, then reload this page.")
    st.stop()

with st.spinner("Running the forecast..."):
    hist, fc = bundle["history"], forecast(bundle, horizon)
metrics, results = ui.load_reports()

ui.masthead(series, bundle["model_name"], bundle["trained_through"], f"{hist.index[0]:%d %b %Y}", f"{hist.index[-1]:%d %b %Y}")

# What-if inputs are read first so the premises box can always show them.
default_cap = int(hist.iloc[-90:].quantile(0.9))
capacity = st.session_state.get("capacity", default_cap)
uplift = st.session_state.get("uplift", 0)

main, side = st.columns([3, 1.2])
with main:
    ui.fig_title(1, f"Daily rentals ({series}): the last {history_days} days and the next {horizon}")
    st.plotly_chart(ui.forecast_fig(hist, fc, history_days), width="stretch", config={"displayModeBar": False})
    ui.caption("Fig. 1.", "Blue is what happened. Crimson is what the model expects. The dotted line is the last day of data. "
                          "The model has no weather input, so a storm day looks like a surprise.")
with side:
    st.markdown(ui.premises(bundle, results, horizon, capacity, uplift), unsafe_allow_html=True)

ui.section("How wrong can it be?")
if metrics is None:
    ui.notice("empty", "No results table yet.", "reports/metrics.csv is missing. Run  python -m demandlab.train.")
else:
    ui.fig_title(2, "Error ruler: test WAPE per model")
    st.markdown(ui.error_ruler(metrics, series, bundle["model_name"]), unsafe_allow_html=True)
    ui.fig_title(3, "Model comparison, test period")
    st.markdown(ui.model_table(metrics, series, bundle["model_name"]), unsafe_allow_html=True)
    ui.caption("Test period.", "Last 90 days of data (autumn into winter). WAPE = sum of absolute errors / sum of actual rentals. "
                               "Selection used validation only; the test column is a single honest look.")

ui.section("What if demand and capacity change?")
c1, c2 = st.columns(2)
capacity = c1.number_input("Daily capacity (rentals per day)", 0, 100_000, default_cap, key="capacity")
uplift = c2.slider("Demand vs forecast (%)", -30, 50, 0, key="uplift")
sim = fc * (1 + uplift / 100)
over = (sim - capacity).clip(lower=0)
st.markdown(f"<div class='what-if'>Assuming capacity {capacity:,.0f}/day and demand {uplift:+d}% vs the forecast. "
            "Your numbers, not model output.</div>", unsafe_allow_html=True)
a, b, c = st.columns(3)
for col, label, value, unit in [(a, "days over capacity", f"{int((over > 0).sum())}", f"of {horizon}"),
                                (b, "unmet demand", f"{over.sum():,.0f}", "rentals"),
                                (c, "peak load", f"{sim.max() / capacity * 100:.0f}" if capacity else "n/a", "% of capacity")]:
    col.markdown(f"<div class='stat-l'>{label}</div><div class='big'>{value}<small>{unit}</small></div>", unsafe_allow_html=True)
ui.fig_title(4, "Scenario demand against capacity")
st.plotly_chart(ui.scenario_fig(sim, capacity), width="stretch", config={"displayModeBar": False})
ui.caption("Fig. 4.", "Dark crimson bars exceed capacity; light bars fit. Colour repeats the label above: days over capacity are counted, not just tinted.")
table = pd.DataFrame({"forecast": fc.round(0), "scenario": sim.round(0), "capacity": capacity, "unmet": over.round(0)})
table.index = table.index.date
with st.expander("Day-by-day table"):
    st.dataframe(table, width="stretch")
st.download_button("Download forecast (CSV)", table.to_csv(), f"forecast_{series}.csv", "text/csv")
