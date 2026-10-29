"""DemandLab - Streamlit app. Run: streamlit run app.py (needs `python -m demandlab.train` first)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandlab.predict import forecast, load_bundle

st.set_page_config(page_title="DemandLab", page_icon="📈", layout="wide")
st.title("DemandLab")
st.caption("Daily bike-rental demand (UCI Bike Sharing, Washington D.C., 2011-2012). "
           "Forecasts come from the trained model in models/; nothing is hard-coded.")

with st.sidebar:
    series = st.selectbox("Series", ["total", "registered", "casual"])
    horizon = st.slider("Forecast horizon (days)", 7, 28, 14)
    history_days = st.slider("History shown (days)", 30, 365, 120)

try:
    bundle = load_bundle(series)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

hist, fc = bundle["history"], forecast(bundle, horizon)
fig = go.Figure()
fig.add_scatter(x=hist.index[-history_days:], y=hist.iloc[-history_days:], name="History")
fig.add_scatter(x=fc.index, y=fc, name="Forecast", line=dict(dash="dash", color="#F28E2B"))
fig.update_layout(yaxis_title="rentals per day", legend_orientation="h", margin=dict(t=30))
st.plotly_chart(fig, width="stretch")
st.info(f"Model: **{bundle['model_name']}** (chosen by validation WAPE; fitted through "
        f"{bundle['trained_through']}). Accuracy falls with the horizon; see README for measured errors. "
        "Weather is not used, so storms or extreme days are not anticipated.")

st.subheader("Demand vs capacity simulation")
st.caption("ASSUMPTIONS you control. The result is a what-if on the forecast, not a prediction of operations.")
c1, c2 = st.columns(2)
capacity = c1.number_input("Daily capacity (rentals/day)", 0, 100_000, int(hist.iloc[-90:].quantile(0.9)))
uplift = c2.slider("Demand uplift vs forecast (%)", -30, 50, 0)
sim = fc * (1 + uplift / 100)
over = (sim - capacity).clip(lower=0)
m = st.columns(3)
m[0].metric("Days over capacity", int((over > 0).sum()), help=f"of {horizon} forecast days")
m[1].metric("Total unmet demand", f"{over.sum():,.0f}")
m[2].metric("Peak utilization", f"{(sim.max() / capacity * 100):.0f}%" if capacity else "n/a")
table = pd.DataFrame({"forecast": fc.round(0), "scenario_demand": sim.round(0),
                      "capacity": capacity, "unmet": over.round(0)})
table.index = table.index.date
st.dataframe(table, width="stretch")
st.download_button("Download forecast (CSV)", table.to_csv(), f"forecast_{series}.csv", "text/csv")
