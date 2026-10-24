"""Swappable pipeline stages. Each stage is a registry: name -> function or class.

A config (see experiments.py) picks one entry per stage by name, plus its parameters:
    features: list of specs   e.g. ["park_means", {"name": "smooth", "steps": 6}]
    weights:  one spec        e.g. {"name": "exp_decay", "half_life_days": 14}
    model:    one spec        e.g. {"name": "ols"}
    power:    one spec        e.g. {"mode": "point"}
    interval: one spec        e.g. {"name": "gaussian"}

Rule for every feature: read reference turbines (parks H and O) only, never park T.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd

from baseline import H_COLS, O_COLS, park_means
from preprocessing import fill_short_gaps, wind_to_power

Data = dict[str, pd.DataFrame]  # sensor name -> timestamp x turbine (see core.load_data)


def _spec(spec) -> tuple[str, dict]:
    """'name' or {'name': ..., **params} -> (name, params)."""
    if isinstance(spec, str):
        return spec, {}
    params = dict(spec)
    return params.pop("name"), params


# ---------------------------------------------------------------- features
# Each feature function: (data, **params) -> DataFrame (timestamp index, one or more columns).

def f_park_means(data: Data) -> pd.DataFrame:
    """H and O: average wind speed of the park's reporting turbines (the baseline inputs)."""
    return park_means(data["wind_speed"])


def f_all_refs(data: Data) -> pd.DataFrame:
    """All 7 reference wind speeds (H1..H5, O1, O2), gaps of <= 2 steps interpolated."""
    ws = data["wind_speed"]
    return pd.DataFrame({c: fill_short_gaps(ws[c], 2) for c in H_COLS + O_COLS})


def f_park_means_smooth(data: Data, steps: int) -> pd.DataFrame:
    """Centred rolling mean of the H and O park means over `steps` steps."""
    pm = park_means(data["wind_speed"])
    sm = pm.rolling(steps, center=True, min_periods=steps // 2 + 1).mean()
    return sm.rename(columns=lambda c: f"{c}_s{steps}")


def f_all_refs_smooth(data: Data, steps: int) -> pd.DataFrame:
    """Centred rolling mean of each of the 7 references over `steps` steps."""
    sm = f_all_refs(data).rolling(steps, center=True, min_periods=steps // 2 + 1).mean()
    return sm.rename(columns=lambda c: f"{c}_s{steps}")


def f_lags(data: Data, steps: list[int], parks=("H", "O")) -> pd.DataFrame:
    """Park means shifted by k steps (k > 0: value from k steps earlier)."""
    pm = park_means(data["wind_speed"])
    return pd.DataFrame({f"{p}_lag{k:+d}": pm[p].shift(k) for p in parks for k in steps})


def f_hour(data: Data, interact: bool = True) -> pd.DataFrame:
    """Time of day as sin/cos (24 h period); optionally multiplied with the H park mean."""
    idx = data["wind_speed"].index
    h = idx.hour + idx.minute / 60
    out = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24)}, index=idx)
    if interact:
        H = park_means(data["wind_speed"])["H"]
        out["H_x_hour_sin"] = H * out["hour_sin"]
        out["H_x_hour_cos"] = H * out["hour_cos"]
    return out


def _h_direction(data: Data) -> pd.Series:
    """Circular-mean wind direction of park H in degrees [0, 360) (NaN if no H turbine reports)."""
    wd = np.deg2rad(data["wind_direction"][H_COLS])
    s, c = np.sin(wd).mean(axis=1, skipna=True), np.cos(wd).mean(axis=1, skipna=True)
    return np.rad2deg(np.arctan2(s, c)) % 360


def _sector(deg: pd.Series, sectors: int) -> pd.Series:
    """Sector index 0..sectors-1 (sector 0 is centred on north); NaN where deg is NaN."""
    return ((deg + 180 / sectors) % 360) // (360 / sectors)


def f_direction(data: Data, sectors: int = 0, smooth_steps: int = 0, o_factor: bool = False) -> pd.DataFrame:
    """Circular-mean H wind direction as sin/cos, or as sector dummies x H park mean (sectors > 0).

    smooth_steps > 0: the interaction terms use the H (and O) mean smoothed over that many steps.
    o_factor: also add O_x_sec{i}, a separate O slope per sector.
    """
    if sectors == 0:
        wd = np.deg2rad(data["wind_direction"][H_COLS])
        s, c = np.sin(wd).mean(axis=1, skipna=True), np.cos(wd).mean(axis=1, skipna=True)
        return pd.DataFrame({"dir_sin": s, "dir_cos": c})
    deg = _h_direction(data)
    sec = _sector(deg, sectors)
    pm = park_means(data["wind_speed"])
    if smooth_steps:
        pm = pm.rolling(smooth_steps, center=True, min_periods=smooth_steps // 2 + 1).mean()
    out = {}
    for i in range(1, sectors):
        d = (sec == i).astype(float).where(deg.notna())
        out[f"sec{i}"] = d
        out[f"H_x_sec{i}"] = pm["H"] * d
        if o_factor:
            out[f"O_x_sec{i}"] = pm["O"] * d
    return pd.DataFrame(out)


def f_h_stddev(data: Data) -> pd.DataFrame:
    """Mean wind-speed standard deviation (turbulence) of park H."""
    return pd.DataFrame({"H_std": data["wind_speed_stddev"][H_COLS].mean(axis=1)})


def f_temperature(data: Data) -> pd.DataFrame:
    """Mean ambient temperature of parks H and O."""
    t = data["ambient_temperature"]
    return pd.DataFrame({"H_temp": t[H_COLS].mean(axis=1), "O_temp": t[O_COLS].mean(axis=1)})


def f_pressure(data: Data) -> pd.DataFrame:
    """Mean air pressure of park H (park O has none)."""
    return pd.DataFrame({"H_pres": data["air_pressure"][H_COLS].mean(axis=1)})


def f_hinges(data: Data, knots=(3, 6, 9)) -> pd.DataFrame:
    """max(0, x - k) of the park means per knot (with park_means: piecewise-linear fit)."""
    pm = park_means(data["wind_speed"])
    return pd.DataFrame({f"{p}_h{k}": (pm[p] - k).clip(lower=0) for p in pm.columns for k in knots})


def f_park_means_tf(data: Data, fn: str = "log1p") -> pd.DataFrame:
    """Transformed park means (fn: log1p or sqrt)."""
    f = {"log1p": np.log1p, "sqrt": np.sqrt}[fn]
    return park_means(data["wind_speed"]).apply(f).rename(columns=lambda c: f"{c}_{fn}")


FEATURES = {
    "park_means": f_park_means,
    "all_refs": f_all_refs,
    "park_means_smooth": f_park_means_smooth,
    "all_refs_smooth": f_all_refs_smooth,
    "lags": f_lags,
    "hour": f_hour,
    "direction": f_direction,
    "h_stddev": f_h_stddev,
    "temperature": f_temperature,
    "pressure": f_pressure,
    "hinges": f_hinges,
    "park_means_tf": f_park_means_tf,
}


def build_X(data: Data, specs: list) -> pd.DataFrame:
    """Concatenate all requested features column-wise."""
    parts = []
    for spec in specs:
        name, params = _spec(spec)
        parts.append(FEATURES[name](data, **params))
    X = pd.concat(parts, axis=1)
    assert not any(c.startswith("T") for c in X.columns), "park T must never be a feature"
    return X


# ---------------------------------------------------------------- weights
# Each weight function: (index, window, **params) -> np.ndarray of weights >= 0, one per step.
# `window` is the outage [start, end). Weights may use data on both sides (the gap is filled offline).

def w_none(index: pd.DatetimeIndex, window) -> np.ndarray:
    return np.ones(len(index))


def _dist_days(index: pd.DatetimeIndex, window) -> np.ndarray:
    """Distance in days from each step to the window [start, end) (0 inside)."""
    start, end = (pd.Timestamp(x) for x in window)
    d = np.maximum((start - index) / pd.Timedelta(days=1), (index - end) / pd.Timedelta(days=1))
    return np.maximum(np.asarray(d, dtype=float), 0.0)


def w_exp_decay(index: pd.DatetimeIndex, window, half_life_days: float) -> np.ndarray:
    """0.5 ** (d / half_life): weight halves every `half_life_days` away from the window (two-sided)."""
    return 0.5 ** (_dist_days(index, window) / half_life_days)


def w_window(index: pd.DatetimeIndex, window, days: float) -> np.ndarray:
    """1 within `days` of the window, else 0."""
    return (_dist_days(index, window) <= days).astype(float)


def w_direction_match(index: pd.DatetimeIndex, window, data: Data, sectors: int = 8, floor: float = 0.2,
                      speed_bins: int = 0) -> np.ndarray:
    """Weight training steps by how common their weather is INSIDE the outage window.

    Weather = H wind-direction sector (x H-speed tercile if speed_bins > 1). H and O are known during the
    outage (only T is missing), so the window's histogram is legitimate. Weight = floor + (1 - floor) * share / max share.
    """
    deg = _h_direction(data).reindex(index)
    key = _sector(deg, sectors)
    if speed_bins > 1:
        H = park_means(data["wind_speed"])["H"].reindex(index)
        q = H.quantile(np.linspace(0, 1, speed_bins + 1)[1:-1]).values
        key = key * speed_bins + pd.Series(np.searchsorted(q, H.values, side="right"), index=index).where(H.notna())
    inside = in_window_mask(index, window)
    share = key[inside].value_counts(normalize=True)
    if share.empty:
        return np.ones(len(index))
    w = floor + (1 - floor) * key.map(share).fillna(0.0) / share.max()
    return np.where(key.isna(), floor, w).astype(float)


def in_window_mask(index: pd.DatetimeIndex, window) -> np.ndarray:
    start, end = (pd.Timestamp(x) for x in window)
    return np.asarray((index >= start) & (index < end))


WEIGHTS = {
    "direction_match": w_direction_match,
    "none": w_none,
    "exp_decay": w_exp_decay,
    "window": w_window,
}


def sample_weights(index: pd.DatetimeIndex, window, spec, data: Data | None = None) -> np.ndarray:
    """Weights for every step; `data` is passed only to weight functions that declare a `data` argument."""
    name, params = _spec(spec)
    fn = WEIGHTS[name]
    if "data" in inspect.signature(fn).parameters:
        params["data"] = data
    return fn(index, window, **params)


# ---------------------------------------------------------------- models
# Model protocol: fit(X, y, w) -> self; predict(X) -> np.ndarray; describe() -> dict (for the GUI).
# X has no NaN when passed in; the pipeline clips predictions at 0 afterwards.

class OLS:
    """y = a + sum(b_j * x_j), weighted least squares (w = 1 gives the baseline)."""

    def fit(self, X: pd.DataFrame, y: pd.Series, w: np.ndarray | None = None) -> "OLS":
        A = np.column_stack([np.ones(len(X)), X.values])
        sw = np.ones(len(X)) if w is None else np.sqrt(w)
        self.beta_, *_ = np.linalg.lstsq(A * sw[:, None], y.values * sw, rcond=None)
        self.cols_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.beta_[0] + X[self.cols_].values @ self.beta_[1:]

    def describe(self) -> dict:
        return {"intercept": float(self.beta_[0]), **{c: float(b) for c, b in zip(self.cols_, self.beta_[1:])}}


class VarianceRatioOLS(OLS):
    """OLS, then rescale predictions so weighted mean/std on the training set match y."""

    def fit(self, X, y, w=None):
        super().fit(X, y, w)
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        f = super().predict(X)
        self.mu_f_, self.mu_y_ = np.average(f, weights=w), np.average(y.values, weights=w)
        sd_f = np.sqrt(np.average((f - self.mu_f_) ** 2, weights=w))
        sd_y = np.sqrt(np.average((y.values - self.mu_y_) ** 2, weights=w))
        self.scale_ = sd_y / sd_f
        return self

    def predict(self, X):
        return self.mu_y_ + self.scale_ * (super().predict(X) - self.mu_f_)

    def describe(self) -> dict:
        return {**super().describe(), "scale": float(self.scale_)}


class Ridge:
    """Standardised ridge regression (sample-weighted); coefficients reported in original units."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y, w=None):
        from sklearn.linear_model import Ridge as _Ridge
        from sklearn.preprocessing import StandardScaler
        self.cols_ = list(X.columns)
        self.sc_ = StandardScaler().fit(X.values)
        self.m_ = _Ridge(alpha=self.alpha).fit(self.sc_.transform(X.values), y.values, sample_weight=w)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(X[self.cols_].values))

    def describe(self) -> dict:
        b = self.m_.coef_ / self.sc_.scale_
        return {"intercept": float(self.m_.intercept_ - b @ self.sc_.mean_), **{c: float(v) for c, v in zip(self.cols_, b)}}


class HGB:
    """Gradient-boosted trees (sklearn HistGradientBoostingRegressor)."""

    def __init__(self, **kwargs):
        self.kwargs = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
        self.kwargs.update(kwargs)

    def fit(self, X, y, w=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.cols_ = list(X.columns)
        self.m_ = HistGradientBoostingRegressor(**self.kwargs).fit(X.values, y.values, sample_weight=w)
        return self

    def predict(self, X):
        return self.m_.predict(X[self.cols_].values)

    def describe(self) -> dict:
        return {"type": "hgb", "n_features": len(self.cols_)}


class OLSForward(OLS):
    """OLS with automatic reference selection: greedy forward choice among the reference columns.

    Reference columns are those named like H1_s6 / O2_s6. All other columns are always kept. Each candidate
    is scored by blocked K-fold CV (contiguous time folds) on the TRAINING rows only; selection stops when the
    CV RMSE improves by less than `min_gain` (relative).
    """

    def __init__(self, folds: int = 5, min_gain: float = 0.005):
        self.folds, self.min_gain = folds, min_gain

    def _cv_rmse(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, cols: list) -> float:
        n = len(y)
        edges = np.linspace(0, n, self.folds + 1).astype(int)
        se = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            te = np.zeros(n, dtype=bool)
            te[a:b] = True
            m = OLS().fit(X.loc[~te, cols], pd.Series(y[~te]), w[~te])
            se += float(((y[te] - m.predict(X.loc[te, cols])) ** 2).sum())
        return float(np.sqrt(se / n))

    def fit(self, X, y, w=None):
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        X = X.reset_index(drop=True)
        yv = y.values
        refs = [c for c in X.columns if re.match(r"^[HO]\d", c)]
        fixed = [c for c in X.columns if c not in refs]
        chosen, best = [], np.inf
        if fixed:
            best = self._cv_rmse(X, yv, w, fixed)
        while True:
            left = [c for c in refs if c not in chosen]
            if not left:
                break
            scores = {c: self._cv_rmse(X, yv, w, fixed + chosen + [c]) for c in left}
            c = min(scores, key=scores.get)
            if chosen and scores[c] > best * (1 - self.min_gain):
                break
            chosen.append(c)
            best = scores[c]
        self.selected_ = chosen
        self.cv_rmse_ = best
        super().fit(X[fixed + chosen], pd.Series(yv), w)
        return self

    def describe(self) -> dict:
        return {"selected_refs": list(self.selected_), "cv_rmse": float(self.cv_rmse_), **super().describe()}


MODELS = {
    "ols_forward": OLSForward,
    "ols": OLS,
    "variance_ratio": VarianceRatioOLS,
    "ridge": Ridge,
    "hgb": HGB,
}


def make_model(spec):
    name, params = _spec(spec)
    return MODELS[name](**params)


# ---------------------------------------------------------------- uncertainty
# Each interval function: (pred, calib_resid, calib_fitted, **params) -> (lo, hi, sigma), Series on pred's index.
# calib_* are residuals (actual - predicted) and the matching predictions used for calibration;
# by default the training residuals, later out-of-sample backtest residuals (E7).

def _bin_of(values: np.ndarray, calib_fitted: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin index of `values` and of the calibration fits (outside the range -> edge bins)."""
    inner = np.quantile(calib_fitted, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(inner, values, side="right"), np.searchsorted(inner, calib_fitted, side="right")


def i_gaussian(pred: pd.Series, calib_resid: np.ndarray, calib_fitted: np.ndarray, z: float = 1.96, bins: int = 1):
    """pred +- z * sigma, sigma = spread of the calibration residuals (per fitted-value bin if bins > 1)."""
    if bins == 1:
        sigma = pd.Series(float(np.std(calib_resid)), index=pred.index)
    else:
        b, cb = _bin_of(pred.values, calib_fitted, bins)
        sd = np.array([np.std(calib_resid[cb == k]) if (cb == k).any() else np.std(calib_resid) for k in range(bins)])
        sigma = pd.Series(np.where(pred.isna(), np.nan, sd[b]), index=pred.index)
    return (pred - z * sigma).clip(lower=0), pred + z * sigma, sigma


def i_conformal(pred: pd.Series, calib_resid: np.ndarray, calib_fitted: np.ndarray, level: float = 0.95, bins: int = 1):
    """pred + empirical residual quantiles (per fitted-value bin if bins > 1); sigma = residual std there."""
    q = [(1 - level) / 2, (1 + level) / 2]
    b, cb = _bin_of(pred.values, calib_fitted, bins)
    rs = [calib_resid[cb == k] if (cb == k).any() else calib_resid for k in range(bins)]
    ql = np.array([np.quantile(r, q) for r in rs])
    sd = np.array([np.std(r) for r in rs])
    nan = pred.isna().values
    lo = pd.Series(np.where(nan, np.nan, pred.values + ql[b, 0]), index=pred.index).clip(lower=0)
    hi = pd.Series(np.where(nan, np.nan, pred.values + ql[b, 1]), index=pred.index)
    return lo, hi, pd.Series(np.where(nan, np.nan, sd[b]), index=pred.index)


INTERVALS = {
    "gaussian": i_gaussian,
    "conformal": i_conformal,
}


def band(pred: pd.Series, calib_resid, calib_fitted, spec):
    name, params = _spec(spec)
    return INTERVALS[name](pred, np.asarray(calib_resid), np.asarray(calib_fitted), **params)


# ---------------------------------------------------------------- power
# Each power function: (pred, sigma, calib_resid, curve, **params) -> np.ndarray kW (NaN where pred is NaN).

def p_point(pred: pd.Series, sigma: pd.Series, calib_resid: np.ndarray, curve: pd.DataFrame) -> np.ndarray:
    """P(v_hat): the predicted wind speed straight through the power curve."""
    return wind_to_power(pred.values, curve)


def p_expected_gauss(pred: pd.Series, sigma: pd.Series, calib_resid: np.ndarray, curve: pd.DataFrame,
                     nodes: int = 21) -> np.ndarray:
    """E[P(v)] with v ~ N(pred, sigma) clipped at 0 (Gauss-Hermite quadrature)."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    v = np.clip(pred.values[:, None] + sigma.values[:, None] * x[None, :], 0, None)
    return wind_to_power(v, curve) @ w


def p_expected_empirical(pred: pd.Series, sigma: pd.Series, calib_resid: np.ndarray, curve: pd.DataFrame,
                         n: int = 41) -> np.ndarray:
    """Mean of P(max(0, pred + r_q)) over n quantiles r_q of the calibration residuals."""
    r = np.quantile(calib_resid, np.linspace(0.5 / n, 1 - 0.5 / n, n))
    v = np.clip(pred.values[:, None] + r[None, :], 0, None)
    return wind_to_power(v, curve).mean(axis=1)


POWER = {
    "point": p_point,
    "expected_gauss": p_expected_gauss,
    "expected_empirical": p_expected_empirical,
}


def to_power(pred: pd.Series, sigma: pd.Series, calib_resid, curve: pd.DataFrame, spec) -> np.ndarray:
    params = dict(spec)
    return POWER[params.pop("mode")](pred, sigma, np.asarray(calib_resid), curve, **params)
