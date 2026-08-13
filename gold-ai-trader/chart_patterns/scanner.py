from __future__ import annotations

import pandas as pd


REQUIRED = ("time", "open", "high", "low", "close")


def validate_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate OHLC data for pattern scanning."""
    if raw is None or len(raw) == 0:
        raise ValueError("OHLC dataframe is empty.")

    frame = raw.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        bad = int(frame["time"].isna().sum())
        raise ValueError(f"Could not parse {bad} timestamps in 'time'.")

    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "tick_volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["tick_volume"], errors="coerce")
    elif "real_volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["real_volume"], errors="coerce")
    elif "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    else:
        frame["volume"] = 0.0

    frame = frame.dropna(subset=list(REQUIRED)).sort_values("time")
    frame = frame.drop_duplicates(subset=["time"], keep="last")
    frame = frame.reset_index(drop=True)

    invalid = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    )
    if invalid.any():
        frame = frame.loc[~invalid].reset_index(drop=True)

    if len(frame) < 50:
        raise ValueError("Not enough valid OHLC rows after cleaning.")

    return frame
