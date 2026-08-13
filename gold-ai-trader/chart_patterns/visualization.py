from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_patterns.models import PatternDetection


def render_clean_candlestick_image(
    market: pd.DataFrame,
    output_path: str | Path,
    image_size: int = 224,
    include_volume: bool = False,
) -> Path:
    """Render a clean candlestick chart for CNN input (no axes/labels)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)
    n = len(market)
    x = np.arange(n)

    dpi = 100
    fig_w = image_size / dpi
    fig_h = image_size / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    up = closes >= opens
    down = ~up
    width = 0.6

    ax.bar(
        x[up],
        (closes - opens)[up],
        bottom=opens[up],
        width=width,
        color="#26a69a",
        linewidth=0,
    )
    ax.bar(
        x[down],
        (opens - closes)[down],
        bottom=closes[down],
        width=width,
        color="#ef5350",
        linewidth=0,
    )
    ax.vlines(x[up], lows[up], highs[up], color="#26a69a", linewidth=0.7)
    ax.vlines(x[down], lows[down], highs[down], color="#ef5350", linewidth=0.7)

    if include_volume and "volume" in market.columns:
        # kept for API compatibility; default unused for CNN cleanliness
        pass

    ax.set_xlim(-1, n)
    pad = max((highs.max() - lows.min()) * 0.03, 1e-6)
    ax.set_ylim(lows.min() - pad, highs.max() + pad)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_chart_with_patterns(
    market: pd.DataFrame,
    detections: list[PatternDetection],
    output_path: str | Path,
    title: str = "",
    show_volume: bool = False,
    last_n_bars: int | None = 200,
    dpi: int = 120,
) -> Path:
    """Annotated chart for human review (not for CNN training)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = market
    index_offset = 0
    if last_n_bars is not None and len(market) > last_n_bars:
        index_offset = len(market) - last_n_bars
        frame = market.iloc[-last_n_bars:].reset_index(drop=True)

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    n = len(frame)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    up = closes >= opens
    down = ~up
    ax.bar(x[up], (closes - opens)[up], bottom=opens[up], width=0.6, color="#26a69a")
    ax.bar(
        x[down],
        (opens - closes)[down],
        bottom=closes[down],
        width=0.6,
        color="#ef5350",
    )
    ax.vlines(x, lows, highs, color="#9e9e9e", linewidth=0.6)

    for det in detections:
        start = det.start_pos - index_offset
        end = det.end_pos - index_offset
        if end < 0 or start >= n:
            continue
        start = max(0, start)
        end = min(n - 1, end)
        ax.axvspan(start, end, color="#42a5f5", alpha=0.15)
        ax.plot(
            [start, end],
            [closes[start], closes[end]],
            color="#ffee58",
            linewidth=1.2,
            linestyle="--",
        )
        if det.breakout_pos is not None:
            bx = det.breakout_pos - index_offset
            if 0 <= bx < n:
                ax.scatter([bx], [closes[bx]], color="#ffeb3b", s=40, zorder=5)
        label = f"{det.name} ({det.status}, {det.confidence:.2f})"
        ax.text(
            start,
            highs[start:end + 1].max(),
            label,
            fontsize=8,
            color="#eeeeee",
            backgroundcolor="#00000088",
        )

    ax.set_title(title or "Pattern review")
    ax.grid(True, alpha=0.2)
    if not show_volume:
        pass
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
