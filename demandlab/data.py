"""Ingestion and preparation: raw UCI Bike Sharing daily file -> clean daily series."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "day.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
# series name -> column in day.csv (daily bike rentals: all / casual users / registered users)
SERIES = {"total": "cnt", "casual": "casual", "registered": "registered"}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read day.csv, check it, return one row per day indexed by date."""
    df = pd.read_csv(path, parse_dates=["dteday"])
    missing = {"dteday", "holiday", *SERIES.values()} - set(df.columns)
    if missing:
        raise ValueError(f"day.csv is missing columns: {sorted(missing)}")
    df = df.drop_duplicates("dteday").sort_values("dteday").set_index("dteday")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    if len(full) != len(df):  # gaps would silently break lag features
        raise ValueError(f"{len(full) - len(df)} missing days in the series")
    if (df[list(SERIES.values())] < 0).any().any():
        raise ValueError("negative demand values found")
    df.index.name = "date"
    return df.asfreq("D")


def get_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[SERIES[name]].astype(float).rename(name)


def holiday_dates(df: pd.DataFrame) -> set:
    return set(df.index[df["holiday"] == 1])


def temporal_split(y: pd.Series, val_days: int = 90, test_days: int = 90):
    """Chronological train / validation / test. No shuffling, so no future leaks into the past."""
    if len(y) <= val_days + test_days + 60:
        raise ValueError("series too short for this split")
    n = len(y)
    return y.iloc[: n - val_days - test_days], y.iloc[n - val_days - test_days: n - test_days], y.iloc[n - test_days:]


def synthetic_demand(days: int = 731, seed: int = 0) -> pd.DataFrame:
    """FALLBACK ONLY: fictional demand with weekly/yearly seasonality, growth and noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-01", periods=days, freq="D", name="date")
    t = np.arange(days)
    base = 3000 + 3 * t + 800 * np.sin(2 * np.pi * (t - 80) / 365.25) + 300 * (idx.day_of_week < 5)
    cnt = np.clip(base + rng.normal(0, 300, days), 100, None).round()
    return pd.DataFrame({"holiday": 0, "cnt": cnt, "casual": (cnt * .2).round(),
                         "registered": (cnt * .8).round()}, index=idx)
