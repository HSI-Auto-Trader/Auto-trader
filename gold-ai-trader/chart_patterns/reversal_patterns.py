from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig
from chart_patterns.models import PatternDetection
from chart_patterns.utils import (
    find_pivots,
    make_detection,
    rel_close,
    window_arrays,
)


def detect_double_bottom(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    lows = find_pivots(arr["low"], cfg.pivot_left, cfg.pivot_right, "low")
    highs = find_pivots(arr["high"], cfg.pivot_left, cfg.pivot_right, "high")
    if len(lows) < 2 or len(highs) < 1:
        return None

    i1, i2 = lows[-2], lows[-1]
    if i2 - i1 < 5:
        return None
    l1, l2 = float(arr["low"][i1]), float(arr["low"][i2])
    if not rel_close(l1, l2, cfg.level_tol * 2.2):
        return None

    mid_highs = [h for h in highs if i1 < h < i2]
    if not mid_highs:
        return None
    neckline = float(max(arr["high"][h] for h in mid_highs))
    trough = min(l1, l2)
    if neckline <= trough:
        return None

    end = len(window) - 1
    close = arr["close"][end]
    breakout = close > neckline
    status = "confirmed" if breakout else "forming"
    depth = neckline - trough
    conf = 0.5 + (0.18 if breakout else 0.05)
    if abs(l1 - l2) / max(abs(trough), 1e-9) < cfg.level_tol:
        conf += 0.08

    return make_detection(
        name="double_bottom",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=i1,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Double bottom near equal lows with neckline",
        meta={
            "breakout_level": neckline,
            "structure_low": trough,
            "structure_high": neckline,
            "height": float(depth),
        },
        config=cfg,
    )


def detect_double_top(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    highs = find_pivots(arr["high"], cfg.pivot_left, cfg.pivot_right, "high")
    lows = find_pivots(arr["low"], cfg.pivot_left, cfg.pivot_right, "low")
    if len(highs) < 2 or len(lows) < 1:
        return None

    i1, i2 = highs[-2], highs[-1]
    if i2 - i1 < 5:
        return None
    h1, h2 = float(arr["high"][i1]), float(arr["high"][i2])
    if not rel_close(h1, h2, cfg.level_tol * 2.2):
        return None

    mid_lows = [l for l in lows if i1 < l < i2]
    if not mid_lows:
        return None
    neckline = float(min(arr["low"][l] for l in mid_lows))
    peak = max(h1, h2)
    if peak <= neckline:
        return None

    end = len(window) - 1
    close = arr["close"][end]
    breakout = close < neckline
    status = "confirmed" if breakout else "forming"
    height = peak - neckline
    conf = 0.5 + (0.18 if breakout else 0.05)
    if abs(h1 - h2) / max(abs(peak), 1e-9) < cfg.level_tol:
        conf += 0.08

    return make_detection(
        name="double_top",
        status=status,
        direction="bearish",
        confidence=conf,
        local_start=i1,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Double top near equal highs with neckline break",
        meta={
            "breakout_level": neckline,
            "structure_low": neckline,
            "structure_high": peak,
            "height": float(height),
        },
        config=cfg,
    )


def detect_inverse_head_and_shoulders(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    lows = find_pivots(arr["low"], cfg.pivot_left, cfg.pivot_right, "low")
    highs = find_pivots(arr["high"], cfg.pivot_left, cfg.pivot_right, "high")
    if len(lows) < 3 or len(highs) < 2:
        return None

    atr_proxy = float(np.mean(arr["high"] - arr["low"]) + 1e-9)
    candidates = []
    # Prefer recent, well-spaced triplets near the end of the window
    for i in range(max(0, len(lows) - 5), len(lows) - 2):
        ls, head, rs = lows[i], lows[i + 1], lows[i + 2]
        if head - ls < 4 or rs - head < 4:
            continue
        lv = arr["low"][[ls, head, rs]]
        if not (lv[1] < lv[0] and lv[1] < lv[2]):
            continue
        if not rel_close(float(lv[0]), float(lv[2]), cfg.level_tol * 2.5):
            continue
        # shoulders clearly above head
        if min(lv[0], lv[2]) - lv[1] < atr_proxy * 1.4:
            continue
        left_peaks = [arr["high"][h] for h in highs if ls < h < head]
        right_peaks = [arr["high"][h] for h in highs if head < h < rs]
        if not left_peaks or not right_peaks:
            continue
        left_peak = float(max(left_peaks))
        right_peak = float(max(right_peaks))
        if not rel_close(left_peak, right_peak, cfg.level_tol * 3.0):
            continue
        neckline = (left_peak + right_peak) / 2.0
        if neckline - float(lv[1]) < atr_proxy * 2.0:
            continue
        # Pattern should still be relevant near the window end
        if len(window) - 1 - rs > max(12, len(window) // 5):
            continue
        candidates.append((ls, head, rs, neckline, float(lv[1])))

    if not candidates:
        return None

    ls, head, rs, neckline, head_low = candidates[-1]
    end = len(window) - 1
    close = arr["close"][end]
    breakout = close > neckline * (1 + cfg.level_tol * 0.35)
    status = "confirmed" if breakout else "forming"
    conf = 0.55 + (0.18 if breakout else 0.0)

    return make_detection(
        name="inverse_head_and_shoulders",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=ls,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Inverse head and shoulders structure",
        meta={
            "breakout_level": neckline,
            "structure_low": head_low,
            "structure_high": neckline,
            "height": float(neckline - head_low),
            "head": head,
            "shoulders": (ls, rs),
        },
        config=cfg,
    )


def detect_cup_and_handle(
    window: pd.DataFrame,
    offset: int = 0,
    config: PatternConfig | None = None,
) -> PatternDetection | None:
    cfg = config or PatternConfig()
    arr = window_arrays(window)
    n = len(window)
    if n < 40:
        return None

    cup_end = int(n * 0.75)
    cup = arr["close"][:cup_end]
    handle = arr["close"][cup_end:]
    if len(handle) < 5:
        return None

    left = float(cup[: max(5, len(cup) // 5)].mean())
    right = float(cup[-max(5, len(cup) // 5) :].mean())
    bottom = float(cup.min())
    if not rel_close(left, right, cfg.level_tol * 4):
        return None
    depth = ((left + right) / 2) - bottom
    if depth < np.mean(arr["high"] - arr["low"]) * 2:
        return None

    # Cup should be rounded-ish: middle lower than sides
    mid = float(cup[len(cup) // 2])
    if mid > bottom + 0.55 * depth:
        return None

    handle_high = float(arr["high"][cup_end:].max())
    handle_low = float(arr["low"][cup_end:].min())
    rim = max(left, right)
    if handle_high > rim * (1 + cfg.level_tol * 2):
        return None
    if (rim - handle_low) > 0.55 * depth:
        return None

    end = n - 1
    close = arr["close"][end]
    breakout = close > rim
    status = "confirmed" if breakout else "forming"
    conf = 0.5 + (0.16 if breakout else 0.04)

    return make_detection(
        name="cup_and_handle",
        status=status,
        direction="bullish",
        confidence=conf,
        local_start=0,
        local_end=end,
        offset=offset,
        breakout_local=end if breakout else None,
        reason="Cup and handle rounded base with shallow handle",
        meta={
            "breakout_level": rim,
            "structure_low": bottom,
            "structure_high": rim,
            "height": float(depth),
        },
        config=cfg,
    )
