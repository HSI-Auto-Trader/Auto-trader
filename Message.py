from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# ============================================================
# SETTINGS
# ============================================================

# None = auto-detect Kaggle or Google Colab repository
REPO_HINT = Path(__file__).resolve().parent / "gold-ai-trader"

DATA_RELATIVE_PATH = Path(
    "data/raw/gold_h1_5years.csv"
)

OUTPUT_RELATIVE_PATH = Path(
    "research/gold_h1_5years.csv"
)

# Each detector will test these chart lengths
WINDOWS = (
    48,
    64,
    80,
    100,
    120,
    160,
    200,
)

# Each generated image contains the last 200 H1 candles
IMAGE_BARS = 200

# CNN input image resolution
IMAGE_SIZE = 224

# Save progress every 50 market days
CHECKPOINT_EVERY = 50

# False = only clean CNN images
# True = also create annotated positive images for human review
CREATE_REVIEW_IMAGES = False

# False recommended during generation
# Change to True later to create one ZIP per strategy
ZIP_EACH_STRATEGY = False

# None = use all dates in the CSV
START_DATE = None
END_DATE = None

# None = process all market days across the full 11 years
# For testing, use for example LIMIT_DAYS = 30
# Start with a short smoke run; set to None for the full dataset.
LIMIT_DAYS = 10


# ============================================================
# LOCATE REPOSITORY
# ============================================================

def locate_repository() -> Path:
    candidates: list[Path] = []

    if REPO_HINT:
        candidates.append(Path(REPO_HINT))

    candidates.extend(
        [
            Path("/kaggle/working/gold-ai-trader"),
            Path("/content/gold-ai-trader"),
            Path.cwd(),
        ]
    )

    for candidate in candidates:
        if (candidate / "chart_patterns").is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "gold-ai-trader repository not found.\n"
        "Clone the repository first or set REPO_HINT."
    )


REPO_PATH = locate_repository()

os.chdir(REPO_PATH)

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))


# ============================================================
# INSTALL REQUIRED PACKAGES
# ============================================================

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import matplotlib  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numpy>=1.26",
            "pandas>=2.0",
            "matplotlib>=3.8",
            "pillow>=10",
        ],
        check=True,
    )


# ============================================================
# IMPORT PROJECT CODE
# ============================================================

import numpy as np
import pandas as pd

from chart_patterns.config import PatternConfig

from chart_patterns.continuation_patterns import (
    detect_consolidation,
    detect_flag,
)

from chart_patterns.reversal_patterns import (
    detect_cup_and_handle,
    detect_double_bottom,
    detect_double_top,
    detect_inverse_head_and_shoulders,
)

from chart_patterns.scanner import validate_ohlc

from chart_patterns.trade_levels import attach_trade_plan

from chart_patterns.trendline_patterns import (
    detect_ascending_triangle,
    detect_bullish_rectangle,
    detect_falling_wedge,
    detect_symmetrical_triangle,
)

from chart_patterns.visualization import (
    plot_chart_with_patterns,
    render_clean_candlestick_image,
)


# ============================================================
# STRATEGIES
# Order is important:
# complete every day for strategy 1, then strategy 2, etc.
# ============================================================

STRATEGIES = {
    "falling_wedge": detect_falling_wedge,
    "symmetrical_triangle": detect_symmetrical_triangle,
    "inverse_head_and_shoulders": (
        detect_inverse_head_and_shoulders
    ),
    "cup_and_handle": detect_cup_and_handle,
    "ascending_triangle": detect_ascending_triangle,
    "bullish_rectangle": detect_bullish_rectangle,
    "double_bottom": detect_double_bottom,
    "double_top": detect_double_top,
    "flag": detect_flag,
    "consolidation": detect_consolidation,
}


# ============================================================
# PATHS
# ============================================================

DATA_PATH = REPO_PATH / DATA_RELATIVE_PATH
OUTPUT_ROOT = REPO_PATH / OUTPUT_RELATIVE_PATH

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Market data not found:\n{DATA_PATH}"
    )

print("=" * 70)
print("Repository:", REPO_PATH)
print("Market data:", DATA_PATH)
print("Output:", OUTPUT_ROOT)
print("=" * 70)


# ============================================================
# LOAD AND VALIDATE OHLC
# ============================================================

raw_market = pd.read_csv(DATA_PATH)

market = validate_ohlc(raw_market)

if "time" not in market.columns:
    raise ValueError(
        "The CSV must contain a 'time' column."
    )

config = PatternConfig()

valid_windows = tuple(
    window
    for window in sorted(set(WINDOWS))
    if (
        config.min_window_bars
        <= window
        <= config.max_window_bars
        and window <= len(market)
    )
)

if not valid_windows:
    raise ValueError(
        "No valid scan windows are available."
    )

print("OHLC rows:", f"{len(market):,}")
print("First time:", market["time"].min())
print("Last time:", market["time"].max())
print("Windows:", valid_windows)


# ============================================================
# CREATE ONE ENDPOINT PER MARKET DAY
# ============================================================

times = pd.to_datetime(
    market["time"],
    utc=True,
    errors="raise",
)

day_keys = times.dt.floor("D")

# Last existing H1 candle from every UTC day
last_positions = (
    market.groupby(day_keys, sort=True)
    .tail(1)
    .index
    .to_numpy(dtype=int)
)

start_timestamp = (
    pd.Timestamp(START_DATE, tz="UTC")
    if START_DATE
    else None
)

end_timestamp = (
    pd.Timestamp(END_DATE, tz="UTC")
    if END_DATE
    else None
)

daily_endpoints: list[tuple[str, int]] = []

for position in last_positions:
    day = day_keys.iloc[position]
    endpoint = int(position) + 1

    # Need enough candles to create the image
    if endpoint < max(
        IMAGE_BARS,
        min(valid_windows),
    ):
        continue

    if (
        start_timestamp is not None
        and day < start_timestamp.floor("D")
    ):
        continue

    if (
        end_timestamp is not None
        and day > end_timestamp.floor("D")
    ):
        continue

    daily_endpoints.append(
        (
            day.strftime("%Y-%m-%d"),
            endpoint,
        )
    )

if LIMIT_DAYS is not None:
    daily_endpoints = daily_endpoints[
        : int(LIMIT_DAYS)
    ]

if not daily_endpoints:
    raise ValueError(
        "No eligible market days were found."
    )

print("Eligible market days:", f"{len(daily_endpoints):,}")
print("First market day:", daily_endpoints[0][0])
print("Last market day:", daily_endpoints[-1][0])


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SHARED_IMAGE_DIRECTORY = (
    OUTPUT_ROOT / "_shared_clean_images"
)

SHARED_IMAGE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


run_metadata = {
    "market_data": str(DATA_RELATIVE_PATH),
    "first_day": daily_endpoints[0][0],
    "last_day": daily_endpoints[-1][0],
    "market_days": len(daily_endpoints),
    "strategies": list(STRATEGIES),
    "windows": list(valid_windows),
    "image_bars": IMAGE_BARS,
    "image_size": IMAGE_SIZE,
    "clean_training_images": True,
    "review_images_are_training_inputs": False,
}

(
    OUTPUT_ROOT / "run_metadata.json"
).write_text(
    json.dumps(
        run_metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    number = float(value)

    if not np.isfinite(number):
        return None

    return number


def time_at(
    position: int | None,
) -> str | None:
    if position is None:
        return None

    if position < 0 or position >= len(market):
        return None

    return pd.Timestamp(
        market.iloc[position]["time"]
    ).isoformat()


def detect_best_pattern(
    detector,
    endpoint: int,
):
    candidates = []

    for window_size in valid_windows:
        if endpoint < window_size:
            continue

        offset = endpoint - window_size

        window = (
            market.iloc[offset:endpoint]
            .reset_index(drop=True)
        )

        detection = detector(
            window,
            offset=offset,
            config=config,
        )

        if detection is None:
            continue

        detection = attach_trade_plan(
            market,
            detection,
            config=config,
        )

        candidates.append(
            (
                detection,
                window_size,
            )
        )

    if not candidates:
        return None

    # Priority:
    # 1. confirmed
    # 2. higher confidence
    # 3. longer detected structure
    # 4. larger scan window
    return max(
        candidates,
        key=lambda item: (
            item[0].status == "confirmed",
            float(item[0].confidence),
            (
                item[0].end_pos
                - item[0].start_pos
            ),
            item[1],
        ),
    )


def link_or_copy_image(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return

    try:
        # Avoid storing the same clean image 10 times
        os.link(
            source,
            destination,
        )
    except OSError:
        shutil.copy2(
            source,
            destination,
        )


def save_manifest(
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)

    if not frame.empty:
        frame = frame.sort_values(
            [
                "day_utc",
                "end_pos",
            ],
            kind="stable",
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        manifest_path,
        index=False,
    )

    return frame


# ============================================================
# GENERATE DATASETS
# ============================================================

all_strategy_summaries: list[
    dict[str, Any]
] = []

all_strategy_manifests: list[
    pd.DataFrame
] = []


print()
print(
    f"Starting {len(daily_endpoints):,} market days "
    f"× {len(STRATEGIES)} strategies."
)
print(
    "This process is resumable. "
    "Rerun the same cell after an interruption."
)
print()


for strategy_number, (
    strategy_name,
    detector,
) in enumerate(
    STRATEGIES.items(),
    start=1,
):

    strategy_directory = (
        OUTPUT_ROOT / strategy_name
    )

    manifest_path = (
        strategy_directory / "manifest.csv"
    )

    strategy_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resume previous progress
    if manifest_path.exists():
        previous_manifest = pd.read_csv(
            manifest_path
        )

        rows = previous_manifest.to_dict(
            orient="records"
        )

        completed_sample_ids = set(
            previous_manifest[
                "sample_id"
            ].astype(str)
        )

    else:
        rows = []
        completed_sample_ids = set()

    print("=" * 70)

    print(
        f"[{strategy_number}/{len(STRATEGIES)}] "
        f"Strategy: {strategy_name}"
    )

    print(
        f"Already complete: "
        f"{len(completed_sample_ids):,}/"
        f"{len(daily_endpoints):,}"
    )

    processed_since_checkpoint = 0

    for day_number, (
        day_utc,
        endpoint,
    ) in enumerate(
        daily_endpoints,
        start=1,
    ):

        end_position = endpoint - 1

        sample_id = (
            f"{strategy_name}__"
            f"{day_utc}__"
            f"{end_position:07d}"
        )

        if sample_id in completed_sample_ids:
            continue

        year, month, _ = day_utc.split("-")

        image_start_position = (
            endpoint - IMAGE_BARS
        )

        # ----------------------------------------------------
        # CLEAN IMAGE
        # Render only once per day and reuse for all strategies
        # ----------------------------------------------------

        shared_image_path = (
            SHARED_IMAGE_DIRECTORY
            / year
            / month
            / (
                f"{day_utc}__"
                f"{end_position:07d}.png"
            )
        )

        if not shared_image_path.exists():
            shared_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_market_window = (
                market.iloc[
                    image_start_position:endpoint
                ]
                .reset_index(drop=True)
            )

            render_clean_candlestick_image(
                image_market_window,
                shared_image_path,
                image_size=IMAGE_SIZE,
                include_volume=False,
            )

        # ----------------------------------------------------
        # DETECT CURRENT STRATEGY
        # ----------------------------------------------------

        result = detect_best_pattern(
            detector,
            endpoint,
        )

        if result is None:
            detection = None
            scan_window = None

            label = 0
            status = "no_pattern"
            direction = "neutral"
            confidence = 0.0
            trade_plan = None

        else:
            detection, scan_window = result

            label = 1
            status = detection.status
            direction = detection.direction

            confidence = (
                safe_float(
                    detection.confidence
                )
                or 0.0
            )

            trade_plan = (
                detection.trade_plan
            )

        # ----------------------------------------------------
        # STRATEGY-SPECIFIC IMAGE
        # ----------------------------------------------------

        relative_image_path = (
            Path(strategy_name)
            / "images"
            / status
            / year
            / month
            / f"{sample_id}.png"
        )

        absolute_image_path = (
            OUTPUT_ROOT
            / relative_image_path
        )

        link_or_copy_image(
            shared_image_path,
            absolute_image_path,
        )

        # ----------------------------------------------------
        # OPTIONAL ANNOTATED POSITIVE IMAGE
        # Human review only, never use for CNN training
        # ----------------------------------------------------

        relative_review_path = ""

        if (
            CREATE_REVIEW_IMAGES
            and detection is not None
        ):

            review_image_path = (
                strategy_directory
                / "review_images"
                / status
                / year
                / month
                / (
                    f"{sample_id}"
                    f"__annotated.png"
                )
            )

            if not review_image_path.exists():
                review_image_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                plot_chart_with_patterns(
                    market.iloc[:endpoint]
                    .reset_index(drop=True),
                    [detection],
                    output_path=review_image_path,
                    title=(
                        strategy_name
                        .replace("_", " ")
                        .title()
                        + f" | {day_utc}"
                    ),
                    show_volume=False,
                    last_n_bars=IMAGE_BARS,
                    dpi=120,
                )

            relative_review_path = (
                review_image_path
                .relative_to(OUTPUT_ROOT)
                .as_posix()
            )

        # ----------------------------------------------------
        # MANIFEST ROW
        # ----------------------------------------------------

        rows.append(
            {
                "sample_id": sample_id,
                "strategy": strategy_name,
                "label": label,
                "status": status,
                "direction": direction,
                "confidence": confidence,
                "scan_window_bars": (
                    scan_window
                ),
                "image_path": (
                    relative_image_path
                    .as_posix()
                ),
                "review_image_path": (
                    relative_review_path
                ),
                "day_utc": day_utc,
                "start_pos": (
                    image_start_position
                ),
                "end_pos": end_position,
                "start_time": time_at(
                    image_start_position
                ),
                "end_time": time_at(
                    end_position
                ),
                "pattern_start_pos": (
                    detection.start_pos
                    if detection
                    else None
                ),
                "pattern_end_pos": (
                    detection.end_pos
                    if detection
                    else None
                ),
                "pattern_start_time": (
                    time_at(
                        detection.start_pos
                    )
                    if detection
                    else None
                ),
                "pattern_end_time": (
                    time_at(
                        detection.end_pos
                    )
                    if detection
                    else None
                ),
                "breakout_pos": (
                    detection.breakout_pos
                    if detection
                    else None
                ),
                "breakout_time": (
                    time_at(
                        detection.breakout_pos
                    )
                    if detection
                    else None
                ),
                "entry_price": (
                    safe_float(
                        trade_plan.entry_price
                    )
                    if trade_plan
                    else None
                ),
                "stop_loss": (
                    safe_float(
                        trade_plan.stop_loss
                    )
                    if trade_plan
                    else None
                ),
                "target_price": (
                    safe_float(
                        trade_plan.target_price
                    )
                    if trade_plan
                    else None
                ),
                "risk_reward": (
                    safe_float(
                        trade_plan.risk_reward
                    )
                    if trade_plan
                    else None
                ),
                "breakout_level": (
                    safe_float(
                        trade_plan.breakout_level
                    )
                    if trade_plan
                    else None
                ),
                "atr_at_entry": (
                    safe_float(
                        trade_plan.atr_at_entry
                    )
                    if trade_plan
                    else None
                ),
                "reason": (
                    detection.reason
                    if detection
                    else (
                        "No rule-based "
                        "candidate at this "
                        "daily endpoint"
                    )
                ),
                "label_source": (
                    "rule_candidate"
                ),
                "review_status": "pending",
            }
        )

        completed_sample_ids.add(
            sample_id
        )

        processed_since_checkpoint += 1

        # ----------------------------------------------------
        # SAVE CHECKPOINT
        # ----------------------------------------------------

        if (
            processed_since_checkpoint
            >= CHECKPOINT_EVERY
        ):

            current_manifest = save_manifest(
                rows,
                manifest_path,
            )

            positive_count = int(
                (
                    current_manifest["label"]
                    == 1
                ).sum()
            )

            print(
                f"  {day_number:,}/"
                f"{len(daily_endpoints):,} "
                f"days | "
                f"saved="
                f"{len(current_manifest):,} | "
                f"positives="
                f"{positive_count:,}"
            )

            processed_since_checkpoint = 0

            gc.collect()

    # --------------------------------------------------------
    # FINAL STRATEGY MANIFEST
    # --------------------------------------------------------

    strategy_manifest = save_manifest(
        rows,
        manifest_path,
    )

    status_counts = (
        strategy_manifest["status"]
        .value_counts()
        .to_dict()
    )

    strategy_summary = {
        "strategy": strategy_name,
        "total_days": int(
            len(strategy_manifest)
        ),
        "positive": int(
            (
                strategy_manifest["label"]
                == 1
            ).sum()
        ),
        "confirmed": int(
            status_counts.get(
                "confirmed",
                0,
            )
        ),
        "forming": int(
            status_counts.get(
                "forming",
                0,
            )
        ),
        "no_pattern": int(
            status_counts.get(
                "no_pattern",
                0,
            )
        ),
        "positive_rate": float(
            (
                strategy_manifest["label"]
                == 1
            ).mean()
        ),
        "first_day": str(
            strategy_manifest[
                "day_utc"
            ].min()
        ),
        "last_day": str(
            strategy_manifest[
                "day_utc"
            ].max()
        ),
    }

    (
        strategy_directory
        / "summary.json"
    ).write_text(
        json.dumps(
            strategy_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_strategy_summaries.append(
        strategy_summary
    )

    all_strategy_manifests.append(
        strategy_manifest
    )

    if ZIP_EACH_STRATEGY:
        archive_directory = (
            OUTPUT_ROOT / "archives"
        )

        archive_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.make_archive(
            str(
                archive_directory
                / strategy_name
            ),
            "zip",
            root_dir=str(OUTPUT_ROOT),
            base_dir=strategy_name,
        )

    print(
        f"DONE {strategy_name}: "
        f"total="
        f"{strategy_summary['total_days']:,}, "
        f"positive="
        f"{strategy_summary['positive']:,}, "
        f"confirmed="
        f"{strategy_summary['confirmed']:,}, "
        f"forming="
        f"{strategy_summary['forming']:,}, "
        f"no-pattern="
        f"{strategy_summary['no_pattern']:,}"
    )

    print()


# ============================================================
# COMBINED OUTPUTS
# ============================================================

combined_manifest = pd.concat(
    all_strategy_manifests,
    ignore_index=True,
)

combined_manifest.to_csv(
    OUTPUT_ROOT
    / "all_strategies_manifest.csv",
    index=False,
)

summary_frame = pd.DataFrame(
    all_strategy_summaries
)

summary_frame.to_csv(
    OUTPUT_ROOT
    / "all_strategies_summary.csv",
    index=False,
)


print("=" * 70)
print("✅ ALL STRATEGIES FINISHED")
print("=" * 70)

print(
    "Combined manifest rows:",
    f"{len(combined_manifest):,}",
)

print(
    "Dataset directory:",
    OUTPUT_ROOT,
)

print()

try:
    display(summary_frame)  # noqa: F821 — Jupyter only
except NameError:
    print(summary_frame.to_string(index=False))