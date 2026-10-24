"""Local JSON checkpoints for expensive side backtests; never stores credentials."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd

from preprocessing import TARGETS

FORMAT_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = tuple(ROOT / name for name in (
    "pipeline/core.py", "pipeline/stages.py", "pipeline/tabpfn_model.py",
    "preprocessing.py", "baseline.py", "pipeline/side_cache.py"))
COLUMNS = ["pred", "lo", "hi", "sigma", "kw", "actual"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def fingerprint(config, data, curve=None):
    """Hash config, all sensor values/schema/index, implementation and optional curve."""
    digest = hashlib.sha256()

    def add(value):
        raw = value if isinstance(value, bytes) else _json(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    add(FORMAT_VERSION)
    add(config)
    for path in SOURCE_PATHS:
        add(path.name)
        add(path.read_bytes())
    frames = dict(data)
    if curve is not None:
        frames["__power_curve__"] = curve
    for sensor, frame in sorted(frames.items()):
        add(sensor)
        add({"columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes],
             "index_dtype": str(frame.index.dtype), "index_names": frame.index.names})
        add(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        add(pd.util.hash_pandas_object(frame.columns).values.tobytes())
    return digest.hexdigest()


class SideBacktestCache:
    """One complete per-window result per atomic file, scoped to an input fingerprint."""

    def __init__(self, config, data, *, targets=TARGETS, root=None, curve=None):
        self.fingerprint = fingerprint(config, data, curve)
        self.directory = Path(root or ROOT / ".cache" / "side_backtests") / self.fingerprint
        self.targets = tuple(targets)

    def _identity(self, name, start, end):
        return {"name": str(name), "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat()}

    def _path(self, name, start, end):
        key = hashlib.sha256(_json(self._identity(name, start, end)).encode()).hexdigest()
        return self.directory / f"{key}.json"

    def _validate(self, result):
        if not isinstance(result, dict) or set(result) != set(self.targets):
            raise ValueError("unexpected target keys")
        for frame in result.values():
            if not isinstance(frame, pd.DataFrame) or list(frame.columns) != COLUMNS:
                raise ValueError("unexpected prediction columns")
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or not frame.index.is_unique:
                raise ValueError("invalid timestamp index")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
                raise ValueError("nonnumeric prediction columns")
            attrs = frame.attrs
            if not isinstance(attrs.get("model"), dict) or not isinstance(attrs.get("resid"), list) or not isinstance(attrs.get("fitted"), list):
                raise ValueError("missing calibration metadata")
            if len(attrs["resid"]) != len(attrs["fitted"]):
                raise ValueError("unaligned calibration metadata")

    def load(self, name, start, end):
        path = self._path(name, start, end)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["version"] != FORMAT_VERSION or payload["window"] != self._identity(name, start, end):
                raise ValueError("checkpoint identity mismatch")
            result = {}
            for target, entry in payload["targets"].items():
                frame = pd.read_json(io.StringIO(entry["frame"]), orient="split", dtype=False)
                frame = frame.astype(entry["dtypes"])
                index = pd.to_datetime(entry["index_ns"], unit="ns", utc=True)
                index = index.tz_convert(entry["timezone"]) if entry["timezone"] else index.tz_localize(None)
                frame.index = pd.DatetimeIndex(index, name=entry["index_name"], freq=entry["frequency"])
                frame.index = frame.index.astype(entry["index_dtype"])
                frame.attrs = entry["attrs"]
                result[target] = frame
            self._validate(result)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            warnings.warn("[side cache] invalid checkpoint ignored", RuntimeWarning, stacklevel=2)
            return None
        print(f"[side cache] hit {name}", flush=True)
        return result

    def save(self, name, start, end, result):
        self._validate(result)
        targets = {}
        for target, frame in result.items():
            targets[target] = {
                "frame": frame.to_json(orient="split", date_format="iso", date_unit="ns", double_precision=15),
                "index_ns": frame.index.as_unit("ns").asi8.tolist(),
                "timezone": str(frame.index.tz) if frame.index.tz else None,
                "index_name": frame.index.name, "index_dtype": str(frame.index.dtype),
                "frequency": frame.index.freqstr,
                "dtypes": {key: str(dtype) for key, dtype in frame.dtypes.items()},
                "attrs": {key: frame.attrs[key] for key in ("model", "resid", "fitted")},
            }
        payload = _json({"version": FORMAT_VERSION, "window": self._identity(name, start, end),
                         "targets": targets})
        path = self._path(name, start, end)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"[side cache] saved {name}", flush=True)
