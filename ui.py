"""Presentation components for DemandLab. Reads saved reports for display only; no modelling here."""
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MONO = "DM Mono, ui-monospace, Menlo, Consolas, monospace"
STEEL, FC, INK = "#3E5C76", "#C8325F", "#13263B"
NAMES = {"seasonal_naive": "baseline (last week)", "ridge_a1": "ridge a=1", "ridge_a10": "ridge a=10",
         "random_forest": "random forest", "gradient_boosting": "gradient boosting"}


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def notice(kind: str, title: str, text: str) -> None:
    st.markdown(f"<div class='notice {kind}'><b>{esc(title)}</b><br>{esc(text)}</div>", unsafe_allow_html=True)


def load_reports():
    """(metrics DataFrame | None, results dict | None) from reports/, written by `python -m demandlab.train`."""
    try:
        metrics = pd.read_csv(ROOT / "reports" / "metrics.csv")
        results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
        return metrics, results
    except (OSError, ValueError):
        return None, None


def masthead(series: str, model_name: str, trained_through: str, first: str, last: str) -> None:
    st.markdown(
        "<div class='mast'><h1>DemandLab</h1>"
        "<p class='dek'>How many bikes will people rent next week, and how wrong could we be? "
        "A forecasting notebook on two years of Washington D.C. bike-share data.</p>"
        f"<div class='entry'>series <b>{esc(series)}</b> · data <b>{esc(first)} to {esc(last)}</b> · "
        f"model <b>{esc(NAMES.get(model_name, model_name))}</b> · fitted through <b>{esc(trained_through)}</b></div></div>",
        unsafe_allow_html=True)


def fig_title(n: int, title: str) -> None:
    st.markdown(f"<div class='fig-title'>Fig. {n}</div><h3 style='margin:.1rem 0 0'>{esc(title)}</h3>", unsafe_allow_html=True)


def caption(label: str, text: str) -> None:
    st.markdown(f"<div class='cap'><b>{esc(label)}</b> {esc(text)}</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<h2 class='sec'>{esc(title)}</h2>", unsafe_allow_html=True)


def forecast_fig(hist: pd.Series, fc: pd.Series, days: int) -> go.Figure:
    h = hist.iloc[-days:]
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h, mode="lines", line=dict(color=STEEL, width=2), hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>actual</extra>")
    fig.add_scatter(x=fc.index, y=fc, mode="lines+markers", line=dict(color=FC, width=2.5, dash="dash"),
                    marker=dict(size=6, color=FC), hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>forecast</extra>")
    low = h.idxmin()
    fig.add_annotation(x=low + pd.Timedelta(days=1.5), y=h.min() + 120, text=f"lowest day: {low:%d %b} · {h.min():,.0f}",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(family=MONO, size=11, color=INK))
    fig.add_annotation(x=h.index[-1], y=h.iloc[-1], text="actual", showarrow=True, ax=-34, ay=-28, arrowcolor=STEEL,
                       font=dict(family=MONO, size=12, color=STEEL))
    fig.add_annotation(x=fc.index[len(fc) // 2], y=fc.iloc[len(fc) // 2], text="forecast", showarrow=True, ax=0, ay=-38,
                       arrowcolor=FC, font=dict(family=MONO, size=12, color=FC))
    fig.add_vline(x=h.index[-1], line=dict(color=INK, width=1, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, hovermode="x",
                      font=dict(family=MONO, size=11, color=INK), paper_bgcolor="#F6F8FA", plot_bgcolor="#F6F8FA")
    fig.update_yaxes(gridcolor="#C9D5E1", rangemode="tozero", title="rentals / day", zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor=INK)
    return fig


def scenario_fig(sim: pd.Series, capacity: float) -> go.Figure:
    over = sim > capacity
    fig = go.Figure(go.Bar(x=sim.index, y=sim, marker_color=[FC if o else "#E9A9BA" for o in over],
                           hovertemplate="%{x|%d %b}: %{y:,.0f}<extra>scenario</extra>"))
    fig.add_hline(y=capacity, line=dict(color=INK, width=1.5, dash="dash"), annotation_text=f"capacity {capacity:,.0f}",
                  annotation_position="top left", annotation_font=dict(family=MONO, size=11, color=INK))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      font=dict(family=MONO, size=11, color=INK), paper_bgcolor="#F6F8FA", plot_bgcolor="#F6F8FA")
    fig.update_yaxes(gridcolor="#C9D5E1", rangemode="tozero", title="rentals / day")
    fig.update_xaxes(showgrid=False, linecolor=INK)
    return fig


def premises(bundle, results, horizon: int, capacity: float, uplift: int) -> str:
    rec = (results or {}).get("series", {}).get(bundle["series"], {}).get("recursive_28d_test", {})
    acc = (f"{rec['chosen']['WAPE']:.3f} (baseline {rec['baseline']['WAPE']:.3f})" if rec else "n/a")
    return ("<div class='premises'><h3>Premises</h3><dl>"
            f"<dt>model</dt><dd>{esc(NAMES.get(bundle['model_name'], bundle['model_name']))}, chosen on validation WAPE</dd>"
            f"<dt>fitted through</dt><dd class='num'>{esc(bundle['trained_through'])}</dd>"
            f"<dt>forecast horizon</dt><dd class='num'>{horizon} days</dd>"
            f"<dt>28-day test error, WAPE</dt><dd class='num'>{esc(acc)}</dd>"
            "<dt>inputs</dt><dd>past demand, weekday, month, holidays. No weather.</dd>"
            f"<dt>your what-if</dt><dd class='num'>capacity {capacity:,.0f}/day · uplift {uplift:+d}%</dd>"
            "</dl></div>")


def model_table(metrics: pd.DataFrame, series: str, chosen: str) -> str:
    m = metrics[metrics.series == series]
    val = m[m.split == "validation"].set_index("model")["WAPE"]
    test = m[m.split == "test"].set_index("model")
    base = test.loc["seasonal_naive", "WAPE"]
    lowest = test["WAPE"].idxmin()
    rows = ""
    for name in test.index:
        t = test.loc[name]
        seals = (" <span class='seal'>chosen</span>" if name == chosen else "") + \
                (" <span class='seal'>lowest test WAPE</span>" if name == lowest else "")
        d = "—" if name == "seasonal_naive" else f"{(t['WAPE'] - base) * 100:+.1f} pp"
        rows += (f"<tr{' class=best' if name == chosen else ''}><td>{esc(NAMES.get(name, name))}{seals}</td>"
                 f"<td>{t['MAE']:,.0f}</td><td>{t['RMSE']:,.0f}</td><td>{val[name]:.3f}</td><td>{t['WAPE']:.3f}</td><td>{d}</td></tr>")
    head = "".join(f"<th>{h}</th>" for h in ["model", "MAE (rentals)", "RMSE (rentals)", "WAPE val", "WAPE test", "vs baseline"])
    return f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def error_ruler(metrics: pd.DataFrame, series: str, chosen: str) -> str:
    test = metrics[(metrics.series == series) & (metrics.split == "test")].set_index("model")["WAPE"]
    top = max(0.6, float(int(test.max() * 10) + 1) / 10)
    base = test["seasonal_naive"]
    rows = ""
    for name, w in test.sort_values().items():
        cls = " best" if name == chosen else (" base" if name == "seasonal_naive" else "")
        rows += (f"<div class='rrow{cls}'><span>{esc(NAMES.get(name, name))}</span>"
                 f"<div class='rtrack'><i class='rbase' style='left:{base / top * 100:.1f}%'></i>"
                 f"<i class='rdot' style='left:{w / top * 100:.1f}%'></i></div><span class='rval'>{w:.3f}</span></div>")
    ticks = "".join(f"<span>{top * k / 5:.2f}</span>" for k in range(6))
    return (f"<div class='ruler'>{rows}<div class='raxis'><div></div><div>{ticks}</div><div></div></div>"
            f"<div class='note' style='color:#5B6C7E;margin-top:.3rem'>Dashed line: baseline ({base:.3f}). "
            "Left of it is better than repeating last week. Shaded row: model chosen on validation.</div></div>")
