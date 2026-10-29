"""Exploratory analysis: trend, seasonality, anomalies. Plain numbers, easy to explain."""
import numpy as np
import pandas as pd


def trend(y: pd.Series) -> dict:
    """Linear trend (units per day) and year-over-year comparison of the first vs last 365 days."""
    t = np.arange(len(y))
    slope, intercept = np.polyfit(t, y.to_numpy(), 1)
    out = {"slope_per_day": float(slope), "intercept": float(intercept)}
    if len(y) >= 730:
        first, last = y.iloc[:365].mean(), y.iloc[-365:].mean()
        out.update(first_365_mean=float(first), last_365_mean=float(last),
                   growth=float(last / first - 1) if first else float("nan"))
    return out


def weekday_profile(y: pd.Series) -> pd.Series:
    """Mean demand per weekday divided by the overall mean (1.0 = average day)."""
    return (y.groupby(y.index.day_of_week).mean() / y.mean()).rename("index")


def month_profile(y: pd.Series) -> pd.Series:
    return (y.groupby(y.index.month).mean() / y.mean()).rename("index")


def detect_anomalies(y: pd.Series, window: int = 28, threshold: float = 3.5) -> pd.DataFrame:
    """Flag days far from the centered rolling median, using a robust z-score (median/MAD).
    Rolling median follows trend and season, so ordinary growth is not flagged."""
    med = y.rolling(window, center=True, min_periods=window // 2).median()
    resid = y - med
    mad = resid.abs().rolling(window * 3, center=True, min_periods=window).median()
    z = 0.6745 * resid / mad.replace(0, np.nan)
    flagged = z.abs() > threshold
    return pd.DataFrame({"value": y, "expected": med, "robust_z": z})[flagged]
