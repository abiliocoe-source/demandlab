"""Forecast metrics. WAPE = sum|error| / sum|actual|; undefined (NaN) when the denominator is 0."""
import numpy as np


def _arrays(y_true, y_pred):
    a, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    return a, p


def mae(y_true, y_pred) -> float:
    a, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - p)))


def rmse(y_true, y_pred) -> float:
    a, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def wape(y_true, y_pred) -> float:
    a, p = _arrays(y_true, y_pred)
    den = np.abs(a).sum()
    return float("nan") if den == 0 else float(np.abs(a - p).sum() / den)


def all_metrics(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "WAPE": wape(y_true, y_pred)}
