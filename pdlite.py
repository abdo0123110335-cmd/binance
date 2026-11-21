"""pdlite: tiny numpy-only subset of pandas used by the scanners.

Exists so the Android build does not need to compile pandas (which fails on
Python 3.14 / Android). Only the features used by scanner_v5 / scanner_htf are
implemented. API mirrors pandas so scanners just do `import pdlite as pd`.
"""
from __future__ import annotations

import numpy as np

_ERR = dict(divide="ignore", invalid="ignore", over="ignore")


def _u(o):
    return o.v if isinstance(o, Series) else o


def _nan_reduce(fn, v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(fn(v)) if v.size else float("nan")


class _Loc:
    def __init__(self, obj):
        self.obj = obj

    def _positions(self, sl):
        idx = self.obj.index
        start = 0 if sl.start is None else int(np.searchsorted(idx, sl.start, side="left"))
        stop = len(idx) if sl.stop is None else int(np.searchsorted(idx, sl.stop, side="right"))
        return slice(start, stop)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.obj._slice(self._positions(key))
        raise TypeError("pdlite .loc supports label slices only")

    def __setitem__(self, key, value):
        rows, col = key
        pos = self._positions(rows)
        self.obj._set_slice(col, pos, value)


class _ILoc:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.obj._slice(key)
        return self.obj._at(int(key))


class _EWM:
    def __init__(self, s, alpha, min_periods):
        self.s, self.alpha, self.mp = s, alpha, min_periods

    def mean(self):
        x = self.s.v.astype(float)
        out = np.full(x.shape, np.nan)
        a = self.alpha
        prev = None
        count = 0
        for i, xi in enumerate(x):
            if np.isnan(xi):
                if prev is not None:
                    out[i] = prev
                continue
            count += 1
            prev = xi if prev is None else (1.0 - a) * prev + a * xi
            if count >= max(self.mp, 1):
                out[i] = prev
        return Series(out, self.s.index)


class _Rolling:
    def __init__(self, s, n):
        self.s, self.n = s, n

    def mean(self):
        x = self.s.v.astype(float)
        out = np.full(x.shape, np.nan)
        n = self.n
        for i in range(n - 1, len(x)):
            w = x[i - n + 1:i + 1]
            if not np.isnan(w).any():
                out[i] = w.mean()
        return Series(out, self.s.index)


class Series:
    __array_ufunc__ = None

    def __init__(self, v, index=None):
        self.v = np.asarray(v)
        self.index = np.arange(len(self.v)) if index is None else np.asarray(index)

    # ---- container protocol
    def __len__(self):
        return len(self.v)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.v, dtype=dtype)

    def _slice(self, sl):
        return Series(self.v[sl], self.index[sl])

    def _at(self, i):
        return self.v[i]

    @property
    def iloc(self):
        return _ILoc(self)

    @property
    def loc(self):
        return _Loc(self)

    @property
    def values(self):
        return self.v

    def to_numpy(self, dtype=None):
        return np.asarray(self.v, dtype=dtype)

    def astype(self, dtype):
        return Series(self.v.astype(dtype), self.index)

    def copy(self):
        return Series(self.v.copy(), self.index.copy())

    def tail(self, n=5):
        return self._slice(slice(max(len(self.v) - n, 0), None)) if n > 0 else self._slice(slice(0, 0))

    # ---- arithmetic / comparison
    def _bin(self, o, f):
        with np.errstate(**_ERR):
            return Series(f(self.v, _u(o)), self.index)

    def __add__(self, o): return self._bin(o, np.add)
    def __radd__(self, o): return self._bin(o, lambda a, b: b + a)
    def __sub__(self, o): return self._bin(o, np.subtract)
    def __rsub__(self, o): return self._bin(o, lambda a, b: b - a)
    def __mul__(self, o): return self._bin(o, np.multiply)
    def __rmul__(self, o): return self._bin(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._bin(o, np.true_divide)
    def __rtruediv__(self, o): return self._bin(o, lambda a, b: b / a)
    def __pow__(self, o): return self._bin(o, np.power)
    def __gt__(self, o): return self._bin(o, np.greater)
    def __ge__(self, o): return self._bin(o, np.greater_equal)
    def __lt__(self, o): return self._bin(o, np.less)
    def __le__(self, o): return self._bin(o, np.less_equal)
    def __eq__(self, o): return self._bin(o, np.equal)
    def __ne__(self, o): return self._bin(o, np.not_equal)
    def __and__(self, o): return self._bin(o, np.logical_and)
    def __or__ (self, o): return self._bin(o, np.logical_or)
    __hash__ = None

    def __neg__(self):
        return Series(-self.v, self.index)

    def __invert__(self):
        return Series(~self.v.astype(bool), self.index)

    def abs(self):
        return Series(np.abs(self.v), self.index)

    # ---- transforms
    def shift(self, n=1):
        x = self.v.astype(float)
        out = np.full(x.shape, np.nan)
        if n > 0:
            out[n:] = x[:-n] if n < len(x) else out[n:]
        elif n < 0:
            out[:n] = x[-n:]
        else:
            out = x.copy()
        return Series(out, self.index)

    def diff(self):
        return self - self.shift(1)

    def clip(self, lower=None, upper=None):
        x = self.v.astype(float)
        out = x.copy()
        ok = ~np.isnan(x)
        if lower is not None:
            out[ok] = np.maximum(out[ok], lower)
        if upper is not None:
            out[ok] = np.minimum(out[ok], upper)
        return Series(out, self.index)

    def where(self, cond, other=np.nan):
        return Series(np.where(_u(cond).astype(bool), self.v, _u(other)), self.index)

    def replace(self, old, new):
        return Series(np.where(self.v == old, new, self.v), self.index)

    def cumsum(self):
        return Series(np.cumsum(self.v.astype(float)), self.index)

    def ewm(self, span=None, alpha=None, adjust=False, min_periods=0):
        if adjust:
            raise NotImplementedError("pdlite supports adjust=False only")
        if alpha is None:
            alpha = 2.0 / (span + 1.0)
        return _EWM(self, alpha, min_periods)

    def rolling(self, n):
        return _Rolling(self, n)

    # ---- reductions (skipna like pandas)
    def sum(self):
        return float(np.nansum(self.v.astype(float)))

    def mean(self):
        return _nan_reduce(np.mean, self.v)

    def min(self):
        return _nan_reduce(np.min, self.v)

    def max(self):
        return _nan_reduce(np.max, self.v)

    def any(self):
        return bool(np.any(self.v))

    def all(self):
        return bool(np.all(self.v))

    def idxmin(self):
        return self.index[int(np.nanargmin(self.v.astype(float)))]

    def idxmax(self):
        return self.index[int(np.nanargmax(self.v.astype(float)))]


class _Row:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)


class DataFrame:
    def __init__(self, data=None, columns=None):
        self._cols: dict[str, np.ndarray] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                self._cols[k] = np.asarray(v)
        elif data is not None:
            rows = list(data)
            for i, c in enumerate(columns or []):
                self._cols[c] = np.array([r[i] for r in rows], dtype=object)
        n = len(next(iter(self._cols.values()))) if self._cols else 0
        self.index = np.arange(n)

    # ---- basics
    def __len__(self):
        return len(self.index)

    @property
    def iloc(self):
        return _ILoc(self)

    @property
    def loc(self):
        return _Loc(self)

    def _new(self, cols, index):
        df = DataFrame()
        df._cols = cols
        df.index = index
        return df

    def _slice(self, sl):
        return self._new({k: v[sl] for k, v in self._cols.items()}, self.index[sl])

    def _at(self, i):
        return _Row({k: v[i] for k, v in self._cols.items()})

    def _set_slice(self, col, pos, value):
        if col not in self._cols:
            self._cols[col] = np.full(len(self), np.nan)
        self._cols[col][pos] = _u(value)

    def copy(self):
        return self._new({k: v.copy() for k, v in self._cols.items()}, self.index.copy())

    def tail(self, n=5):
        return self._slice(slice(max(len(self) - n, 0), None)) if n > 0 else self._slice(slice(0, 0))

    def iterrows(self):
        for i in range(len(self)):
            yield self.index[i], self._at(i)

    def reset_index(self, drop=True, inplace=False):
        if inplace:
            self.index = np.arange(len(self))
            return None
        out = self.copy()
        out.index = np.arange(len(out))
        return out

    def dropna(self, subset=None):
        cols = subset if subset is not None else list(self._cols)
        ok = np.ones(len(self), dtype=bool)
        for c in cols:
            ok &= ~np.isnan(np.asarray(self._cols[c], dtype=float))
        return self._new({k: v[ok] for k, v in self._cols.items()}, self.index[ok])

    # ---- column access
    def __getitem__(self, key):
        if isinstance(key, Series):
            m = key.v.astype(bool)
            return self._new({k: v[m] for k, v in self._cols.items()}, self.index[m])
        return Series(self._cols[key], self.index)

    def __setitem__(self, key, value):
        v = _u(value)
        if np.isscalar(v):
            v = np.full(len(self), v)
        self._cols[key] = np.asarray(v)


# ---- module-level helpers mirroring pandas -------------------------------

def to_numeric(s, errors="raise"):
    src = s.v if isinstance(s, Series) else np.asarray(s)
    out = np.empty(len(src), dtype=float)
    for i, x in enumerate(src):
        try:
            out[i] = float(x)
        except (TypeError, ValueError):
            if errors == "coerce":
                out[i] = np.nan
            else:
                raise
    return Series(out, s.index if isinstance(s, Series) else None)


def notna(x):
    if isinstance(x, Series):
        return Series(~np.isnan(x.v.astype(float)), x.index)
    try:
        return not bool(np.isnan(x))
    except TypeError:
        return x is not None


class _Concat:
    def __init__(self, series):
        self.series = series

    def max(self, axis=1):
        stack = np.vstack([np.asarray(s.v, dtype=float) for s in self.series])
        with np.errstate(**_ERR):
            out = np.fmax.reduce(stack, axis=0)
        return Series(out, self.series[0].index)


def concat(objs, axis=1):
    if axis != 1:
        raise NotImplementedError("pdlite concat supports axis=1 only")
    return _Concat(list(objs))
