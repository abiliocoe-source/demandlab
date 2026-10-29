"""Candidate models. All expose fit(X, y) / predict(X) on the FEATURES columns."""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SeasonalNaive:
    """Baseline: demand equals the same weekday one week earlier (the lag7 feature)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag7"].to_numpy()


def candidates() -> dict:
    """name -> unfitted model. Small, fixed configs (also recorded in reports/results.json)."""
    return {
        "seasonal_naive": SeasonalNaive(),
        "ridge_a1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_a10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                        learning_rate=0.05, random_state=0),
    }
