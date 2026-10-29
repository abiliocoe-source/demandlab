"""Feature engineering. Forecast horizon is 7 days: every demand-derived feature for day t
uses only values up to t-7, so a forecast made on day t-7 is possible (no leakage)."""
import numpy as np
import pandas as pd

MIN_LAG = 7
ORIGIN = pd.Timestamp("2011-01-01")
FEATURES = (["lag7", "lag14", "roll7", "t", "is_holiday", "month_sin", "month_cos"]
            + [f"dow_{i}" for i in range(7)])


def build_features(y: pd.Series, holidays: set) -> pd.DataFrame:
    """One row per date in y.index. y may contain NaN for future days (they only feed later rows
    after being filled with predictions)."""
    f = pd.DataFrame(index=y.index)
    f["lag7"] = y.shift(MIN_LAG)
    f["lag14"] = y.shift(MIN_LAG + 7)
    f["roll7"] = y.shift(MIN_LAG).rolling(7).mean()  # mean of days t-13 .. t-7
    f["t"] = (y.index - ORIGIN).days
    f["is_holiday"] = [int(d in holidays) for d in y.index]
    month = y.index.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    for i in range(7):
        f[f"dow_{i}"] = (y.index.day_of_week == i).astype(int)
    return f[FEATURES]
