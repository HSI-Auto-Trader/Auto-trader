from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig
from chart_patterns.models import PatternDetection
from chart_patterns.utils import (
    find_pivots,
    linear_fit,
    make_detection,
    rel_close,
    window_arrays,
)


def _pivots(df: pd.DataFrame, config: PatternConfig):
    arr = window_arrays(df)
    highs = find_pivots(arr["high"], config.pivot_left, config.pivot_right, "high")
    lows = find_pivots(arr["low"], config.pivot_left, config.pivot_right, "low")
    return arr, highs, lows


def detect_falling_wedge(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr, highs, lows = _pivots(window, cfg)
    if len(highs) < 2 or len(lows) < 2:
        return None

    hx = np.array(highs[-3:], dtype=float)
    hy = arr["high"][highs[-3:]]
    lx = np.array(lows[-3:], dtype=float)
    ly = arr["low"][lows[-3:]]

    hs, hi, hr = linear_fit(hx, hy)
    ls, li, lr = linear_fit(lx, ly)

    # Both sloping down; upper steeper (less negative) wait - falling wedge:
    # both down, resistance declines faster than support => hs < ls < 0 and converging
    if not (hs < 0 and ls < 0):
        return None
    if hs >= ls:  # upper should fall faster (more negative)
        return None

    start = min(int(hx[0]), int(lx[0]))
    end = len(window) - 1
    width_start = (hs * start + hi) - (ls * start + li)
    width_end = (hs * end + hi) - (ls * end + li)
    if width_start <= 0 or width_end >= width_start * (1 - cfg.convergence_min):
        return None

    upper_now = hs * end + hi
    close = arr["close"][end]
    breakout = close > upper_now
    status = "confirmed" if breakout else "forming"
    conf = 0.45 + 0.25 * ((hr + lr) / 2) + (0.15 if breakout else 0.0)
    if abs(hs - ls) / max(abs(hs), 1e-9) > 0.2:
        conf += 0.05

    return make_detection(
        name="falling_wedge",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=start,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Falling wedge with converging down-sloping rails",
        meta={
            "breakout_level": float(upper_now),
            "structure_low": float(ly.min()),
            "structure_high": float(hy.max()),
            "height": float(max(width_start, hy.max() - ly.min())),
        },
        config=cfg,
    )


def detect_symmetrical_triangle(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr, highs, lows = _pivots(window, cfg)
    if len(highs) < 2 or len(lows) < 2:
        return None

    hx = np.array(highs[-3:], dtype=float)
    hy = arr["high"][highs[-3:]]
    lx = np.array(lows[-3:], dtype=float)
    ly = arr["low"][lows[-3:]]
    hs, hi, hr = linear_fit(hx, hy)
    ls, li, lr = linear_fit(lx, ly)

    if not (hs < -cfg.slope_flat_tol and ls > cfg.slope_flat_tol):
        return None

    start = min(int(hx[0]), int(lx[0]))
    end = len(window) - 1
    width_start = (hs * start + hi) - (ls * start + li)
    width_end = (hs * end + hi) - (ls * end + li)
    if width_start <= 0 or width_end >= width_start * (1 - cfg.convergence_min):
        return None

    upper_now = hs * end + hi
    lower_now = ls * end + li
    close = arr["close"][end]
    if close > upper_now:
        direction, breakout = "bullish", True
    elif close < lower_now:
        direction, breakout = "bearish", True
    else:
        direction, breakout = "neutral", False

    status = "confirmed" if breakout else "forming"
    conf = 0.45 + 0.25 * ((hr + lr) / 2) + (0.15 if breakout else 0.0)

    return make_detection(
        name="symmetrical_triangle",
        status=status,
        direction=direction if breakout else "neutral",
        confidence=conf,
        local_start=start,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Symmetrical triangle with opposing trendlines",
        meta={
            "breakout_level": float(upper_now if direction != "bearish" else lower_now),
            "structure_low": float(ly.min()),
            "structure_high": float(hy.max()),
            "height": float(width_start),
        },
        config=cfg,
    )


def detect_ascending_triangle(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr, highs, lows = _pivots(window, cfg)
    if len(highs) < 2 or len(lows) < 2:
        return None

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]
    hy = arr["high"][recent_highs]
    lx = np.array(recent_lows, dtype=float)
    ly = arr["low"][recent_lows]

    resistance = float(np.median(hy))
    if not all(rel_close(float(h), resistance, cfg.level_tol * 1.8) for h in hy):
        return None

    ls, li, lr = linear_fit(lx, ly)
    if ls <= cfg.slope_flat_tol:
        return None

    start = min(recent_highs[0], recent_lows[0])
    end = len(window) - 1
    close = arr["close"][end]
    breakout = close > resistance * (1 + cfg.level_tol * 0.5)
    status = "confirmed" if breakout else "forming"
    conf = 0.48 + 0.25 * lr + (0.15 if breakout else 0.0)

    return make_detection(
        name="ascending_triangle",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=start,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Ascending triangle: flat resistance, rising support",
        meta={
            "breakout_level": resistance,
            "structure_low": float(ly.min()),
            "structure_high": resistance,
            "height": float(resistance - ly.min()),
        },
        config=cfg,
    )


def detect_bullish_rectangle(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr, highs, lows = _pivots(window, cfg)
    if len(highs) < 2 or len(lows) < 2:
        return None

    hy = arr["high"][highs[-3:]]
    ly = arr["low"][lows[-3:]]
    resistance = float(np.median(hy))
    support = float(np.median(ly))
    if resistance <= support:
        return None
    if not all(rel_close(float(h), resistance, cfg.level_tol * 2) for h in hy):
        return None
    if not all(rel_close(float(l), support, cfg.level_tol * 2) for l in ly):
        return None

    # Prior uptrend bias
    mid = len(window) // 3
    if arr["close"][mid] <= arr["close"][0]:
        return None

    start = min(highs[-3:][0], lows[-3:][0])
    end = len(window) - 1
    close = arr["close"][end]
    breakout = close > resistance * (1 + cfg.level_tol * 0.5)
    status = "confirmed" if breakout else "forming"
    conf = 0.5 + (0.18 if breakout else 0.05)

    return make_detection(
        name="bullish_rectangle",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=start,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Bullish rectangle / consolidation after advance",
        meta={
            "breakout_level": resistance,
            "structure_low": support,
            "structure_high": resistance,
            "height": float(resistance - support),
        },
        config=cfg,
    )
