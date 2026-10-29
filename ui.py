"""Presentation components for DemandLab. Reads saved reports for display only; no modelling here."""
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SANS = "Geist, system-ui, sans-serif"
INK, MUTED, FC, GRID, GREY = "#141414", "#6B6B6B", "#C8325F", "#E4E7EB", "#8A8A8A"
NAMES = {"seasonal_naive": "Baseline: last week repeated", "ridge_a1": "Ridge, alpha 1", "ridge_a10": "Ridge, alpha 10",
         "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
SERIES = {"total": "Total rentals", "registered": "Registered riders", "casual": "Casual riders"}


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def note(kind: str, title: str, text: str) -> None:
    st.markdown(f"<div class='note {kind}'><b>{esc(title)}</b> {esc(text)}</div>", unsafe_allow_html=True)


def load_reports():
    """(metrics DataFrame | None, results dict | None) from reports/, written by `python -m demandlab.train`."""
    try:
        return (pd.read_csv(ROOT / "reports" / "metrics.csv"),
                json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None, None


def brand(scope: str) -> None:
    """Product mark and scope at the top of the sidebar: history in ink, the forecast in the one hue."""
    mark = ("<svg width='32' height='32' viewBox='0 0 32 32' aria-hidden='true'><rect x='1' y='1' width='30' height='30' rx='4' fill='#fff' stroke='#E4E7EB'/>"
            "<polyline points='6,22 12,16 17,19 22,10' fill='none' stroke='#141414' stroke-width='2'/>"
            "<polyline points='22,10 25,14 28,12' fill='none' stroke='#C8325F' stroke-width='2' stroke-dasharray='3 2'/></svg>")
    st.markdown(f"<div class='brand'>{mark}<div><div class='n'>DemandLab</div><div class='s'>{esc(scope)}</div></div></div>", unsafe_allow_html=True)


def title(h1: str, dek_html: str, stamp: str) -> None:
    st.markdown(f"<div class='title'><h1>{esc(h1)}</h1><p class='dek serif'>{dek_html}</p><div class='stamp'>{esc(stamp)}</div></div>",
                unsafe_allow_html=True)


def card_header(h: str, sub: str = "") -> None:
    st.markdown(f"<div class='card-h'><h3>{esc(h)}</h3><span class='sub'>{esc(sub)}</span></div>", unsafe_allow_html=True)


def hero_and_legend(horizon: int, avg_fc: float, avg_prev: float, rows) -> str:
    chg = avg_fc / avg_prev - 1 if avg_prev else None
    pill = "<span class='pill neu'>no earlier period</span>" if chg is None else f"<span class='pill neu'>{chg:+.0%} vs the last {horizon} days</span>"
    body = "".join(f"<tr><td><span class='dot' style='background:{c}'></span>{esc(n)}<span class='when'>{esc(sub)}</span></td><td class='r'>{v:,.0f}</td></tr>"
                   for c, n, sub, v in rows)
    return (f"<div class='hero'><div class='l'>Next {horizon} days, daily average</div>"
            f"<div class='v'>{avg_fc:,.0f}<small>rentals a day</small></div>{pill}</div>"
            f"<table class='legend'><tbody>{body}</tbody></table>")


def premises(bundle, results, horizon: int, where: str = "desk") -> str:
    rec = (results or {}).get("series", {}).get(bundle["series"], {}).get("recursive_28d_test", {})
    err = f"{rec['chosen']['WAPE']:.3f} (baseline {rec['baseline']['WAPE']:.3f})" if rec else "not available"
    rows = [("Model", f"{NAMES.get(bundle['model_name'], bundle['model_name'])}, chosen on validation error"),
            ("Fitted through", bundle["trained_through"]), ("Horizon", f"{horizon} days, recursive"),
            ("Error at 28 days", f"WAPE {err} on the test period"),
            ("Inputs", "past demand, weekday, month, holidays"), ("Cannot see", "weather and one-off events")]
    return (f"<div class='prem {where}-only'><div class='t'>Premises</div><dl>" + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in rows) + "</dl></div>")


def forecast_figure(hist: pd.Series, fc: pd.Series, days: int) -> go.Figure:
    h = hist.iloc[-days:]
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h, mode="lines", line=dict(color=INK, width=2), hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>actual</extra>")
    fx, fy = [h.index[-1], *fc.index], [float(h.iloc[-1]), *fc.to_numpy()]
    fig.add_scatter(x=fx, y=fy, mode="lines", line=dict(color=FC, width=2, dash="dash"), hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>forecast</extra>")
    top = max(float(h.max()), float(fc.max())) * 1.12
    fig.add_shape(type="line", x0=h.index[-1], x1=h.index[-1], y0=0, y1=top, line=dict(color=GREY, width=1, dash="dot"))
    fig.add_annotation(x=h.index[-1], y=top, text="Last data", showarrow=False, xanchor="right", xshift=-6, yanchor="top", font=dict(family=SANS, size=12, color=MUTED))
    fig.add_annotation(x=fc.index[-1], y=float(fc.iloc[-1]), text=f"{fc.iloc[-1]:,.0f} on {fc.index[-1]:%d %b}", showarrow=False,
                       xanchor="right", yanchor="bottom", yshift=10, font=dict(family=SANS, size=13, color=FC))
    low = h.idxmin()
    fig.add_annotation(x=low, y=float(h.min()), text=f"lowest day: {low:%d %b}, {h.min():,.0f}", showarrow=False, xanchor="left", xshift=8, yshift=10,
                       font=dict(family=SANS, size=12, color=INK))
    fig.update_layout(height=380, margin=dict(l=0, r=8, t=8, b=0), showlegend=False, hovermode="x", font=dict(family=SANS, size=13, color=MUTED),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    fig.update_yaxes(gridcolor=GRID, range=[0, top], nticks=5, tickformat=",d", title=None, zeroline=False, showline=False)
    fig.update_xaxes(showgrid=False, linecolor=GRID, nticks=7, tickformat="%d %b", title=None)
    return fig


def model_table(metrics: pd.DataFrame, series: str, chosen: str) -> str:
    """The error ruler is a column: each model's test WAPE as a dot on a 0 to scale track, a tick at the baseline."""
    m = metrics[metrics.series == series]
    val = m[m.split == "validation"].set_index("model")["WAPE"]
    test = m[m.split == "test"].set_index("model")
    scale = max(0.6, float(int(test["WAPE"].max() * 10) + 1) / 10)
    base = float(test.loc["seasonal_naive", "WAPE"])
    rows = ""
    for name in test["WAPE"].sort_values().index:
        t = test.loc[name]
        sel = name == chosen
        pill = " <span class='pill neu'>Chosen</span>" if sel else ""
        ruler = (f"<span class='rul'><i class='b' style='left:{base / scale * 100:.1f}%'></i>"
                 f"<i class='d{' c' if sel else ''}' style='left:{t['WAPE'] / scale * 100:.1f}%'></i></span>")
        rows += (f"<tr{' class=chosen' if sel else ''}><td>{esc(NAMES.get(name, name))}{pill}</td><td class='m'>{t['MAE']:,.0f}</td><td class='m'>{t['RMSE']:,.0f}</td>"
                 f"<td class='m'>{val[name]:.3f}</td><td class='w'>{t['WAPE']:.3f}</td><td class='k'>{ruler}</td></tr>")
    axis = f"<div class='axis'><span>0</span><span>{scale / 2:.1f}</span><span>{scale:.1f}</span></div>"
    head = ("<th>Model</th><th class='m'>MAE, rentals</th><th class='m'>RMSE, rentals</th><th class='m'>WAPE, validation</th><th>WAPE, test</th>"
            f"<th>Test error, tick = baseline{axis}</th>")
    return f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def scenario_strip(over_days: int, horizon: int, unmet: float, peak: str) -> str:
    return (f"<div class='kpis'><div class='kpi'><div class='l'>Days over capacity</div><div class='v'>{over_days}<small>of {horizon}</small></div></div>"
            f"<div class='kpi'><div class='l'>Unmet demand</div><div class='v'>{unmet:,.0f}<small>rentals</small></div></div>"
            f"<div class='kpi'><div class='l'>Peak load</div><div class='v'>{peak}</div></div></div>")


def scenario_figure(sim: pd.Series, capacity: float) -> go.Figure:
    over = sim > capacity
    fig = go.Figure(go.Bar(x=sim.index, y=sim, marker_color=[FC if o else "rgba(200,50,95,0.35)" for o in over],
                           hovertemplate="%{x|%d %b}: %{y:,.0f}<extra>scenario</extra>"))
    fig.add_hline(y=capacity, line=dict(color=INK, width=1, dash="dash"), annotation_text=f"capacity {capacity:,.0f}",
                  annotation_position="top left", annotation_font=dict(family=SANS, size=12, color=INK))
    fig.update_layout(height=240, margin=dict(l=0, r=8, t=8, b=0), showlegend=False, font=dict(family=SANS, size=13, color=MUTED),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", bargap=0.35)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero", nticks=5, tickformat=",d", title=None, zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickformat="%d %b", title=None)
    return fig
