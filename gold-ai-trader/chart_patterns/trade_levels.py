from __future__ import annotations

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig
from chart_patterns.models import PatternDetection, TradePlan
from chart_patterns.utils import atr


def attach_trade_plan(
    market: pd.DataFrame,
    detection: PatternDetection,
    config: PatternConfig | None = None,
) -> PatternDetection:
    """Attach entry / stop / target levels using ATR and pattern geometry."""
    cfg = config or PatternConfig()

    end = int(detection.end_pos)
    if end < 0 or end >= len(market):
        detection.trade_plan = TradePlan()
        return detection

    atr_series = atr(market, cfg.atr_period)
    atr_value = float(atr_series.iloc[end])
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = float(
            (market["high"] - market["low"]).iloc[max(0, end - 20) : end + 1].mean()
        )

    close = float(market.iloc[end]["close"])
    high = float(market.iloc[end]["high"])
    low = float(market.iloc[end]["low"])

    meta = detection.meta or {}
    breakout_level = meta.get("breakout_level")
    structure_low = meta.get("structure_low", low)
    structure_high = meta.get("structure_high", high)
    height = meta.get("height")

    if breakout_level is None:
        if detection.direction == "bullish":
            breakout_level = float(structure_high)
        elif detection.direction == "bearish":
            breakout_level = float(structure_low)
        else:
            breakout_level = close

    breakout_level = float(breakout_level)
    if height is None:
        height = max(
            abs(float(structure_high) - float(structure_low)),
            atr_value * 1.5,
        )
    height = float(max(height, atr_value * 0.5))

    if detection.direction == "bullish":
        entry = max(close, breakout_level)
        stop = float(structure_low) - 0.25 * atr_value
        risk = max(entry - stop, atr_value * 0.35)
        target = entry + cfg.risk_reward_target * risk
        # Prefer measured-move style target when available
        measured = breakout_level + height
        if measured > entry:
            target = max(target, measured)
    elif detection.direction == "bearish":
        entry = min(close, breakout_level)
        stop = float(structure_high) + 0.25 * atr_value
        risk = max(stop - entry, atr_value * 0.35)
        target = entry - cfg.risk_reward_target * risk
        measured = breakout_level - height
        if measured < entry:
            target = min(target, measured)
    else:
        entry = close
        stop = close - atr_value
        target = close + cfg.risk_reward_target * atr_value
        risk = atr_value

    rr = abs(target - entry) / risk if risk > 1e-9 else None

    detection.trade_plan = TradePlan(
        entry_price=float(entry),
        stop_loss=float(stop),
        target_price=float(target),
        risk_reward=float(rr) if rr is not None else None,
        breakout_level=float(breakout_level),
        atr_at_entry=float(atr_value),
    )
    return detection
