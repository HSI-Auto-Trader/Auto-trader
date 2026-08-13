from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig
from chart_patterns.models import PatternDetection
from chart_patterns.utils import (
    find_pivots,
    linear_fit,
    make_detection,
    window_arrays,
)


def detect_flag(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    n = len(window)
    if n < 30:
        return None

    pole_end = int(n * 0.55)
    pole = arr["close"][:pole_end]
    flag = arr["close"][pole_end:]
    if len(flag) < 8:
        return None

    pole_move = float(pole[-1] - pole[0])
    atr_proxy = float(np.mean(arr["high"] - arr["low"]) + 1e-9)
    if pole_move < 2.0 * atr_proxy:
        return None

    x = np.arange(len(flag), dtype=float)
    slope, intercept, r2 = linear_fit(x, flag)
    # Bull flag: shallow downward/flat consolidation after sharp rise
    if slope > cfg.slope_flat_tol * atr_proxy:
        return None
    if abs(flag[-1] - flag[0]) > 0.65 * pole_move:
        return None

    highs = find_pivots(arr["high"][pole_end:], 1, 1, "high")
    resistance = float(np.max(arr["high"][pole_end:]))
    support = float(np.min(arr["low"][pole_end:]))
    close = arr["close"][-1]
    breakout = close > resistance * (1 + cfg.level_tol * 0.35)
    status = "confirmed" if breakout else "forming"
    conf = 0.48 + 0.2 * r2 + (0.15 if breakout else 0.05)

    return make_detection(
        name="flag",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=max(0, pole_end - max(8, pole_end // 3)),
        local_end=n - 1,
        offset=offset,
        breakout_local=n - 1 if breakout else None,
        reason="Bull flag after impulsive advance",
        meta={
            "breakout_level": resistance,
            "structure_low": support,
            "structure_high": resistance,
            "height": float(pole_move),
            "flag_pivots": len(highs),
        },
        config=cfg,
    )


def detect_consolidation(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    n = len(window)
    if n < 24:
        return None

    segment = max(20, n // 3)
    prices = arr["close"][-segment:]
    highs = arr["high"][-segment:]
    lows = arr["low"][-segment:]
    rng = float(highs.max() - lows.min())
    mid = float(np.median(prices))
    if mid <= 0:
        return None

    compression = rng / mid
    if compression > 0.025:  # too wide for consolidation on gold H1
        return None

    x = np.arange(segment, dtype=float)
    slope, _, r2 = linear_fit(x, prices)
    if abs(slope) > cfg.slope_flat_tol * mid * 2:
        return None

    # Touches near both boundaries
    near_high = np.sum(highs >= highs.max() * (1 - cfg.level_tol))
    near_low = np.sum(lows <= lows.min() * (1 + cfg.level_tol))
    if near_high < 2 or near_low < 2:
        return None

    close = arr["close"][-1]
    resistance = float(highs.max())
    support = float(lows.min())
    if close > resistance * (1 + cfg.level_tol * 0.4):
        direction, breakout = "bullish", True
    elif close < support * (1 - cfg.level_tol * 0.4):
        direction, breakout = "bearish", True
    else:
        direction, breakout = "neutral", False

    status = "confirmed" if breakout else "forming"
    conf = 0.5 + 0.15 * r2 + (0.12 if breakout else 0.0)
    if compression < 0.015:
        conf += 0.08

    return make_detection(
        name="consolidation",
        status=status,
        direction=direction,
        confidence=conf,
        local_start=n - segment,
        local_end=n - 1,
        offset=offset,
        breakout_local=n - 1 if breakout else None,
        reason="Tight range consolidation",
        meta={
            "breakout_level": resistance if direction != "bearish" else support,
            "structure_low": support,
            "structure_high": resistance,
            "height": rng,
        },
        config=cfg,
    )
