from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradePlan:
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    risk_reward: float | None = None
    breakout_level: float | None = None
    atr_at_entry: float | None = None


@dataclass
class PatternDetection:
    name: str
    status: str  # confirmed | forming
    direction: str  # bullish | bearish | neutral
    confidence: float
    start_pos: int
    end_pos: int
    breakout_pos: int | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    trade_plan: TradePlan | None = None
