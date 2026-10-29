import numpy as np
import pandas as pd
import pytest

from demandlab.analysis import detect_anomalies, trend, weekday_profile
from demandlab.data import load_raw, get_series, synthetic_demand, temporal_split
from demandlab.features import FEATURES, build_features
from demandlab.metrics import mae, rmse, wape
from demandlab.models import SeasonalNaive
from demandlab.predict import forecast, load_bundle
from demandlab import train as train_mod


def test_metrics_known_values_and_zero_denominator():
    assert mae([10, 20], [12, 18]) == 2
    assert rmse([10, 20], [12, 18]) == 2
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)
    assert np.isnan(wape([0, 0], [1, 2]))  # undefined, not a crash and not a fake number
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_raw_data_loads_and_has_no_gaps():
    raw = load_raw()
    assert len(raw) == 731 and raw.index.is_monotonic_increasing
    assert raw.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_temporal_split_is_chronological_and_complete():
    y = get_series(load_raw(), "total")
    tr, va, te = temporal_split(y, 90, 90)
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()
    assert len(tr) + len(va) + len(te) == len(y)
    with pytest.raises(ValueError):
        temporal_split(y.iloc[:100])


def test_features_do_not_leak_the_future():
    """Changing demand on/after day d must not change the features of day d (lags start at 7)."""
    y = get_series(load_raw(), "total")
    d = y.index[300]
    altered = y.copy()
    altered.loc[d - pd.Timedelta(days=6):] += 99999  # everything from t-6 onwards
    a, b = build_features(y, set()), build_features(altered, set())
    pd.testing.assert_series_equal(a.loc[d], b.loc[d])
    assert list(a.columns) == FEATURES


def test_baseline_is_same_weekday_last_week():
    y = get_series(load_raw(), "total")
    X = build_features(y, set()).dropna()
    assert (SeasonalNaive().predict(X) == y.shift(7).loc[X.index].to_numpy()).all()


def test_analysis_helpers():
    y = get_series(load_raw(), "total")
    assert trend(y)["slope_per_day"] > 0
    assert len(weekday_profile(y)) == 7
    spiked = y.copy()
    spiked.iloc[400] = 5.0
    assert y.index[400] in detect_anomalies(spiked).index


def test_train_and_forecast_end_to_end(tmp_path):
    """Small end-to-end run on the fallback synthetic data (fast, no dependence on saved models)."""
    res = train_mod.run(tmp_path / "m", tmp_path / "r", raw=synthetic_demand())
    assert set(res["series"]) == {"total", "casual", "registered"}
    bundle = load_bundle("total", tmp_path / "m")
    f = forecast(bundle, 14)
    assert len(f) == 14 and (f >= 0).all()
    assert f.index[0] == bundle["history"].index[-1] + pd.Timedelta(days=1)
    assert (tmp_path / "r" / "metrics.csv").exists()
    again = forecast(bundle, 14)
    pd.testing.assert_series_equal(f, again)  # deterministic
    with pytest.raises(ValueError):
        forecast(bundle, 0)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="demandlab.train"):
        load_bundle("total", tmp_path)
