from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig
from chart_patterns.models import PatternDetection


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def find_pivots(
    series: np.ndarray,
    left: int = 2,
    right: int = 2,
    mode: str = "high",
) -> list[int]:
    """Return local pivot indices (local to the series)."""
    n = len(series)
    if n < left + right + 1:
        return []

    pivots: list[int] = []
    for i in range(left, n - right):
        window = series[i - left : i + right + 1]
        center = series[i]
        if mode == "high":
            if center >= window.max() and np.sum(window == center) == 1:
                pivots.append(i)
            elif center >= window.max():
                # allow ties if center is first max
                if np.argmax(window) == left:
                    pivots.append(i)
        else:
            if center <= window.min() and np.sum(window == center) == 1:
                pivots.append(i)
            elif center <= window.min():
                if np.argmin(window) == left:
                    pivots.append(i)
    return pivots


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, r2."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def rel_close(a: float, b: float, tol: float) -> bool:
    base = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / base <= tol


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def make_detection(
    *,
    name: str,
    status: str,
    direction: str,
    confidence: float,
    local_start: int,
    local_end: int,
    offset: int,
    breakout_local: int | None = None,
    reason: str = "",
    meta: dict | None = None,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    conf = clamp01(confidence)
    if conf < cfg.min_confidence and status != "forming":
        return None
    if status == "forming" and conf < cfg.min_confidence * 0.85:
        return None

    return PatternDetection(
        name=name,
        status=status,
        direction=direction,
        confidence=conf,
        start_pos=offset + local_start,
        end_pos=offset + local_end,
        breakout_pos=(
            offset + breakout_local
            if breakout_local is not None
            else None
        ),
        reason=reason,
        meta=meta or {},
    )


def window_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
    }
