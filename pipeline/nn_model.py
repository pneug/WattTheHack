"""Small, weighted neural regressor implementing the pipeline model protocol.

Epoch selection uses only a chronological tail of the rows supplied to ``fit``.
The selected number of epochs is then refitted on all those rows. Neither phase
sees outage targets; feature and target scalers are fitted afresh in each phase.
Requires scikit-learn >= 1.7 for weighted MLP training.
"""

from __future__ import annotations

import inspect
from numbers import Integral

import numpy as np
import pandas as pd


class SmallMLP:
    """Reproducible CPU MLP with weighted scaling, loss, and epoch selection.

    ``max_epochs`` bounds each of selection and final refitting. A zero validation
    fraction disables selection. Fewer than 20 positive-weight rows also use the
    fixed epoch budget, rather than holding out an uninformative tiny tail.
    """

    def __init__(self, hidden_layer_sizes=(32, 16), max_epochs=80, batch_size=256,
                 learning_rate_init=0.001, alpha=0.001, validation_fraction=0.1,
                 patience=10, min_delta=0.0001, random_state=0):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        if not self.hidden_layer_sizes or any(
                not isinstance(n, Integral) or isinstance(n, bool) or n <= 0
                for n in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers")
        for name, value in (("max_epochs", max_epochs), ("batch_size", batch_size),
                            ("patience", patience)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        for name, value, strictly_positive in (
                ("learning_rate_init", learning_rate_init, True),
                ("alpha", alpha, False), ("min_delta", min_delta, False)):
            if not np.isfinite(value) or value < 0 or (strictly_positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
        if not np.isfinite(validation_fraction) or not 0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        self.max_epochs, self.batch_size = int(max_epochs), int(batch_size)
        self.learning_rate_init, self.alpha = float(learning_rate_init), float(alpha)
        self.validation_fraction, self.patience = float(validation_fraction), int(patience)
        self.min_delta, self.random_state = float(min_delta), int(random_state)

    def _new_model(self, rows):
        from sklearn.neural_network import MLPRegressor

        if "sample_weight" not in inspect.signature(MLPRegressor.partial_fit).parameters:
            raise RuntimeError("SmallMLP requires scikit-learn >= 1.7 for sample_weight support")
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes, activation="relu", solver="adam",
            alpha=self.alpha, batch_size=min(self.batch_size, rows),
            learning_rate_init=self.learning_rate_init, random_state=self.random_state,
            early_stopping=False, shuffle=True, max_iter=1, tol=0.0,
            n_iter_no_change=self.max_epochs + 1,
        )

    @staticmethod
    def _scaled(X, y, w):
        from sklearn.preprocessing import StandardScaler

        xs = StandardScaler().fit(X, sample_weight=w)
        ys = StandardScaler().fit(y[:, None], sample_weight=w)
        return xs, ys, xs.transform(X), ys.transform(y[:, None]).ravel()

    def fit(self, X: pd.DataFrame, y: pd.Series, w=None):
        """Fit only supplied rows; ``w`` is passed as genuine sample weights.

        Datetime-indexed inputs are sorted before making the inner temporal split.
        Zero-weight rows contribute to neither scalers, loss, nor epoch selection.
        Positive weights are normalised to mean one, keeping alpha comparable.
        """
        from threadpoolctl import threadpool_limits

        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or X.shape[1] == 0:
            raise ValueError("X must be a DataFrame with unique feature columns")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("X and y indexes must match")
        xv, yv = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        weights = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        if yv.shape != (len(X),) or weights.shape != (len(X),):
            raise ValueError("y and w must be one-dimensional with one value per X row")
        if not np.isfinite(xv).all() or not np.isfinite(yv).all():
            raise ValueError("X and y must contain only finite values")
        if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
            raise ValueError("w must be finite, nonnegative, and have positive total weight")
        order = np.argsort(X.index.asi8, kind="stable") if isinstance(X.index, pd.DatetimeIndex) else np.arange(len(X))
        order = order[weights[order] > 0]
        xv, yv, weights = xv[order], yv[order], weights[order]
        if len(yv) < 2:
            raise ValueError("SmallMLP needs at least two positive-weight training rows")
        weights = weights / weights.max()
        weights = weights / weights.mean()
        self.cols_ = list(X.columns)
        self.n_train_ = len(yv)
        n_val = max(1, int(np.ceil(len(yv) * self.validation_fraction))) if self.validation_fraction and len(yv) >= 20 else 0
        self.n_validation_ = n_val
        self.selection_epochs_ = 0
        self.best_validation_mse_ = None
        self.selected_epochs_ = self.max_epochs

        # Limit BLAS overhead and keep these small fits reproducible on shared CPUs.
        with threadpool_limits(limits=1):
            if n_val:
                cut = len(yv) - n_val
                xsc, ysc, xt, yt = self._scaled(xv[:cut], yv[:cut], weights[:cut])
                self.selection_x_mean_ = xsc.mean_.copy()
                self.selection_y_mean_ = float(ysc.mean_[0])
                xval = xsc.transform(xv[cut:])
                yval = ysc.transform(yv[cut:, None]).ravel()
                model, best, stale = self._new_model(cut), np.inf, 0
                # Normalise this prefix independently so regularisation has the
                # same interpretation as in the final full-training fit.
                wt = weights[:cut] / weights[:cut].mean()
                for epoch in range(1, self.max_epochs + 1):
                    model.partial_fit(xt, yt, sample_weight=wt)
                    mse = float(np.average((model.predict(xval) - yval) ** 2, weights=weights[cut:]))
                    if not np.isfinite(mse):
                        raise ValueError("MLP training produced a nonfinite validation loss")
                    self.selection_epochs_ = epoch
                    if mse < best - self.min_delta:
                        best, stale, self.selected_epochs_ = mse, 0, epoch
                    else:
                        stale += 1
                    if stale >= self.patience:
                        break
                self.best_validation_mse_ = float(best)
            self.x_scaler_, self.y_scaler_, xt, yt = self._scaled(xv, yv, weights)
            self.m_ = self._new_model(len(yv))
            for _ in range(self.selected_epochs_):
                self.m_.partial_fit(xt, yt, sample_weight=weights)
        # Also verify the full fit when epoch selection has been disabled.
        if not np.isfinite(self.predict(X)).all():
            raise ValueError("MLP training produced nonfinite predictions")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "m_"):
            raise RuntimeError("SmallMLP must be fitted before prediction")
        xv = np.asarray(X[self.cols_], dtype=float)
        if not np.isfinite(xv).all():
            raise ValueError("X must contain only finite values")
        if len(xv) == 0:
            return np.empty(0, dtype=float)
        pred = self.m_.predict(self.x_scaler_.transform(xv))
        return self.y_scaler_.inverse_transform(pred[:, None]).ravel()

    def describe(self) -> dict:
        if not hasattr(self, "m_"):
            raise RuntimeError("SmallMLP must be fitted before describe")
        return {"type": "small_mlp", "hidden_layer_sizes": list(self.hidden_layer_sizes),
                "n_features": len(self.cols_), "n_train": self.n_train_,
                "selected_epochs": self.selected_epochs_, "selection_epochs": self.selection_epochs_,
                "n_validation": self.n_validation_, "validation_mse_scaled": self.best_validation_mse_,
                "random_state": self.random_state, "sample_weight": True,
                "early_stopping": "chronological training tail, then refit all training rows"}
