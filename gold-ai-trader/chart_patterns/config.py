from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatternConfig:
    min_window_bars: int = 40
    max_window_bars: int = 220

    pivot_left: int = 2
    pivot_right: int = 2

    min_confidence: float = 0.45
    atr_period: int = 14
    risk_reward_target: float = 2.0

    # Relative tolerance for level equality (fraction of price)
    level_tol: float = 0.0025
    # Trendline / slope tolerances
    slope_flat_tol: float = 0.00015
    convergence_min: float = 0.15
