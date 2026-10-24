"""Optional hosted TabPFN side experiment; importing this module never authenticates.

Uses the stable TabPFN-3.5-Plus API, not the alpha Fast model. The caller must
use ``calibration_mask_`` to calibrate residuals on rows withheld from context.
See https://docs.priorlabs.ai/models/selecting-model-version and
https://docs.priorlabs.ai/capabilities/kv-cache.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import shlex
import threading
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKEN_KEYS = ("TABPFN_TOKEN", "TABPFN_API_KEY", "tabpfn_API_key", "PRIORLABS_API_KEY")
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def _load_api_token(path: Path | None = None) -> str:
    """Read only supported token entries, without logging or expanding .env values."""
    for key in TOKEN_KEYS:
        if os.environ.get(key, "").strip():
            return os.environ[key].strip()
    path = ROOT / ".env" if path is None else path
    if path.exists():
        entries = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if sep and key.strip() in TOKEN_KEYS:
                try:
                    parts = shlex.split(value, comments=True, posix=True)
                except ValueError:
                    raise ValueError("Invalid quoting in TabPFN token entry in .env") from None
                if len(parts) == 1 and parts[0].strip():
                    entries[key.strip()] = parts[0].strip()
        for key in TOKEN_KEYS:
            if key in entries:
                return entries[key]
    raise RuntimeError("Set TABPFN_TOKEN or a supported TabPFN API key in the root .env")


def _paced_call(fn, *args, interval: float, **kwargs):
    """Smooth sequential fits/predicts across models; do not retry billable calls."""
    global _LAST_REQUEST
    with _REQUEST_LOCK:
        delay = interval - (time.monotonic() - _LAST_REQUEST)
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST = time.monotonic()
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        # The client may include headers/URLs in errors. Never forward its text.
        raise RuntimeError(
            f"TabPFN API request failed ({type(exc).__name__}); client details withheld"
        ) from None


class TabPFNAPI:
    """Pipeline fit/predict/describe adapter with bounded training and inference.

    Reserve up to 512 training rows for honest residual calibration before
    selecting a seeded uniform context of at most 4096 other rows. Neither
    selection uses targets or the outage. Nonuniform weights are rejected,
    because the hosted client does not accept sample_weight.
    """

    def __init__(self, model_path: str = "v3.5_default", max_train_rows: int = 4096,
                 calibration_rows: int = 512, batch_size: int = 4096,
                 n_estimators: int = 8, random_state: int = 0,
                 fit_mode: str = "fit_with_cache", request_interval: float = 1.05):
        for name, value in (("max_train_rows", max_train_rows),
                            ("calibration_rows", calibration_rows), ("batch_size", batch_size),
                            ("n_estimators", n_estimators)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_train_rows < 2 or n_estimators > 8:
            raise ValueError("TabPFN requires max_train_rows >= 2 and n_estimators <= 8")
        if not np.isfinite(request_interval) or request_interval < 0:
            raise ValueError("request_interval must be finite and nonnegative")
        self.model_path, self.max_train_rows = model_path, max_train_rows
        self.calibration_rows, self.batch_size = calibration_rows, batch_size
        self.n_estimators, self.random_state = n_estimators, random_state
        self.fit_mode, self.request_interval = fit_mode, request_interval

    def _client(self):
        try:
            client = importlib.import_module("tabpfn_client")
        except ImportError:
            raise ImportError(
                "TabPFN is optional: pip install -r requirements-side-experiments.txt"
            ) from None
        # Explicit token setter bypasses all interactive/cached-login paths.
        token = _load_api_token()
        try:
            self.client_version_ = getattr(client, "__version__", "unknown")
            client.set_access_token(token)
            return client.TabPFNRegressor(
                model_path=self.model_path, n_estimators=self.n_estimators,
                random_state=self.random_state, fit_mode=self.fit_mode,
                thinking_mode=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"TabPFN client setup failed ({type(exc).__name__}); client details withheld"
            ) from None

    def fit(self, X: pd.DataFrame, y: pd.Series, w=None) -> "TabPFNAPI":
        self.__dict__.pop("m_", None)
        self._pred_cache = {}
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError("TabPFN requires a DataFrame with unique feature columns")
        if len(X) < 3 or len(y) != len(X):
            raise ValueError("TabPFN needs at least three aligned training rows")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("TabPFN X and y indices must align")
        xv, yv = X.to_numpy(dtype=float), np.asarray(y, dtype=float)
        if yv.ndim != 1 or not np.isfinite(xv).all() or not np.isfinite(yv).all():
            raise ValueError("TabPFN requires finite numerical features and one finite target")
        if X.shape[1] == 0:
            raise ValueError("TabPFN requires at least one feature")
        if w is not None:
            weights = np.asarray(w, dtype=float)
            if (weights.shape != (len(X),) or not np.isfinite(weights).all()
                    or (weights <= 0).any() or not np.all(weights == weights[0])):
                raise ValueError("TabPFN API supports only uniform positive weights; use weights='none'")
        self.cols_ = list(X.columns)
        order = np.random.default_rng(self.random_state).permutation(len(X))
        n_cal = min(self.calibration_rows, max(1, len(X) // 5), len(X) - 2)
        self.calibration_mask_ = np.zeros(len(X), dtype=bool)
        self.calibration_mask_[order[:n_cal]] = True
        self.context_indices_ = np.sort(order[n_cal:n_cal + self.max_train_rows])
        self.n_train_rows_ = len(X)
        model = self._client()
        _paced_call(model.fit, X.iloc[self.context_indices_].reset_index(drop=True),
                    yv[self.context_indices_], interval=self.request_interval)
        self.m_ = model
        self.n_predict_requests_ = 0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "m_"):
            raise RuntimeError("Fit TabPFN before predicting")
        values = X[self.cols_].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("TabPFN prediction features must be finite")
        parts = []
        for start in range(0, len(X), self.batch_size):
            batch = np.ascontiguousarray(values[start:start + self.batch_size])
            key = (batch.shape, hashlib.sha256(batch.tobytes()).digest())
            if key not in self._pred_cache:
                pred = np.asarray(_paced_call(
                    self.m_.predict, pd.DataFrame(batch, columns=self.cols_),
                    output_type="mean", interval=self.request_interval,
                ), dtype=float)
                if pred.shape != (len(batch),) or not np.isfinite(pred).all():
                    raise RuntimeError("TabPFN API returned invalid predictions")
                self._pred_cache[key] = pred.copy()
                self.n_predict_requests_ += 1
            parts.append(self._pred_cache[key].copy())
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    def describe(self) -> dict:
        return {
            "type": "tabpfn_api", "model_path": self.model_path,
            "client_version": getattr(self, "client_version_", "unknown"),
            "n_estimators": self.n_estimators, "random_state": self.random_state,
            "fit_mode": self.fit_mode, "max_train_rows": self.max_train_rows,
            "context_rows": len(self.context_indices_), "available_training_rows": self.n_train_rows_,
            "calibration_rows": int(self.calibration_mask_.sum()),
            "calibration": "held_out_training_rows", "n_features": len(self.cols_),
            "predict_requests": self.n_predict_requests_,
        }
