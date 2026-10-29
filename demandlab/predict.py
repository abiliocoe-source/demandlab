"""Inference, independent of the notebook and of training: load a saved bundle, forecast ahead."""
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURES, MIN_LAG, build_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_bundle(series: str, models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / f"{series}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m demandlab.train")
    return joblib.load(path)


def forecast(bundle: dict, horizon: int, history: pd.Series | None = None) -> pd.Series:
    """Recursive daily forecast. Day t uses demand up to t-7; beyond 7 days it consumes its own
    earlier predictions, so error can grow with the horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    y = (bundle["history"] if history is None else history).astype(float).copy()
    holidays, model = bundle["holidays"], bundle["model"]
    preds = {}
    for d in pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"):
        y.loc[d] = float("nan")
        row = build_features(y.iloc[-(MIN_LAG + 30):], holidays).loc[[d], FEATURES]
        # the t feature depends on absolute date, unaffected by slicing
        value = max(float(model.predict(row)[0]), 0.0)  # demand cannot be negative
        y.loc[d] = value
        preds[d] = value
    return pd.Series(preds, name="forecast")
