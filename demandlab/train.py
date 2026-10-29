"""Training pipeline: python -m demandlab.train

For each series: temporal split -> fit every candidate on train -> choose by VALIDATION WAPE ->
refit all candidates on train+val -> report TEST metrics -> save the chosen model.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import SERIES, get_series, holiday_dates, load_raw, temporal_split
from .features import FEATURES, build_features
from .metrics import all_metrics
from .models import candidates
from .predict import forecast

ROOT = Path(__file__).resolve().parent.parent
VAL_DAYS = TEST_DAYS = 90
REC_HORIZON, REC_STEP = 28, 14


def _xy(F, y, idx):
    rows = F.loc[idx].dropna()  # first ~14 days have no lags
    return rows[FEATURES], y.loc[rows.index]


def recursive_eval(bundle, y, test_index) -> dict:
    """Multi-step check: from several origins inside the test period, forecast 28 days ahead
    using only history before the origin, then pool the errors."""
    actual, pred = [], []
    for k in range(0, len(test_index) - REC_HORIZON + 1, REC_STEP):
        origin = test_index[k]
        hist = y.loc[: origin - pd.Timedelta(days=1)]
        f = forecast(bundle, REC_HORIZON, history=hist)
        actual += list(y.loc[f.index]); pred += list(f)
    return {"horizon_days": REC_HORIZON, "n_points": len(actual), **all_metrics(actual, pred)}


def run(models_dir: Path = ROOT / "models", reports_dir: Path = ROOT / "reports", raw=None) -> dict:
    raw = load_raw() if raw is None else raw
    hol = holiday_dates(raw)
    models_dir.mkdir(exist_ok=True); reports_dir.mkdir(exist_ok=True)
    rows, results = [], {"config": {"val_days": VAL_DAYS, "test_days": TEST_DAYS, "horizon_days": 7,
                                    "features": FEATURES, "selection_metric": "validation WAPE"},
                         "series": {}}
    for name in SERIES:
        y = get_series(raw, name)
        train, val, test = temporal_split(y, VAL_DAYS, TEST_DAYS)
        F = build_features(y, hol)
        Xtr, ytr = _xy(F, y, train.index)
        Xva, yva = _xy(F, y, val.index)
        Xte, yte = _xy(F, y, test.index)
        Xtv, ytv = _xy(F, y, train.index.append(val.index))

        val_wape = {}
        for cname, model in candidates().items():
            m = all_metrics(yva, model.fit(Xtr, ytr).predict(Xva))
            val_wape[cname] = m["WAPE"]
            rows.append({"series": name, "model": cname, "split": "validation", **m})
        chosen = min(val_wape, key=val_wape.get)

        fitted, test_pred = {}, {}
        for cname, model in candidates().items():  # refit on train+val, score on untouched test
            fitted[cname] = model.fit(Xtv, ytv)
            test_pred[cname] = fitted[cname].predict(Xte)
            rows.append({"series": name, "model": cname, "split": "test",
                         **all_metrics(yte, test_pred[cname])})

        err = (yte - test_pred[chosen]).abs().sort_values(ascending=False).head(5)
        bundle = {"series": name, "model_name": chosen, "model": fitted[chosen], "history": y,
                  "holidays": hol, "features": FEATURES, "trained_through": str(val.index[-1].date())}
        joblib.dump(bundle, models_dir / f"{name}.joblib")
        base_bundle = {**bundle, "model": fitted["seasonal_naive"]}
        results["series"][name] = {
            "chosen_model": chosen,
            "validation_wape": val_wape,
            "test": {"chosen": all_metrics(yte, test_pred[chosen]),
                     "baseline": all_metrics(yte, test_pred["seasonal_naive"])},
            "recursive_28d_test": {"chosen": recursive_eval(bundle, y, test.index),
                                   "baseline": recursive_eval(base_bundle, y, test.index)},
            "worst_test_days": [{"date": str(d.date()), "actual": float(y.loc[d]),
                                 "predicted": float(test_pred[chosen][list(yte.index).index(d)]),
                                 "abs_error": float(e)} for d, e in err.items()],
            "split_dates": {"train_end": str(train.index[-1].date()), "val_end": str(val.index[-1].date()),
                            "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date())},
        }
    pd.DataFrame(rows).round(4).to_csv(reports_dir / "metrics.csv", index=False)
    (reports_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    res = run()
    for s, r in res["series"].items():
        t, b = r["test"]["chosen"], r["test"]["baseline"]
        print(f"{s}: chosen={r['chosen_model']} test WAPE {t['WAPE']:.3f} vs baseline {b['WAPE']:.3f}")
