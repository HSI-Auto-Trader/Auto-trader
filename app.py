import json
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gold AI Trader - Dataset Explorer",
    page_icon="📈",
    layout="wide"
)

# Relative directory paths matching Message.py
OUTPUT_ROOT = Path("gold-ai-trader/research/daily_strategy_datasets_h1_11years")
MANIFEST_PATH = OUTPUT_ROOT / "all_strategies_manifest.csv"
SUMMARY_PATH = OUTPUT_ROOT / "all_strategies_summary.csv"


# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_manifest() -> pd.DataFrame | None:
    if not MANIFEST_PATH.exists():
        return None
    return pd.read_csv(MANIFEST_PATH)

@st.cache_data
def load_summary() -> pd.DataFrame | None:
    if not SUMMARY_PATH.exists():
        return None
    return pd.read_csv(SUMMARY_PATH)


# ============================================================
# MAIN APPLICATION
# ============================================================
st.title("📈 Gold AI Trader — Strategy & Pattern Visualizer")

manifest_df = load_manifest()
summary_df = load_summary()

if manifest_df is None:
    st.error(
        f"Manifest not found at `{MANIFEST_PATH}`.\n\n"
        "Please ensure `Message.py` has finished running or check your file path."
    )
    st.stop()


# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("🔍 Dataset Filters")

# Filter 1: Strategy Selection
strategies = ["All"] + sorted(manifest_df["strategy"].dropna().unique().tolist())
selected_strategy = st.sidebar.selectbox("Strategy", strategies)

# Filter 2: Pattern Status
statuses = ["All"] + sorted(manifest_df["status"].dropna().unique().tolist())
selected_status = st.sidebar.selectbox("Status", statuses)

# Filter 3: Pattern Label
selected_label = st.sidebar.radio(
    "Pattern Detection",
    options=["All", "Pattern Found (Label = 1)", "No Pattern (Label = 0)"]
)

# Apply Filtering Logic
df_filtered = manifest_df.copy()

if selected_strategy != "All":
    df_filtered = df_filtered[df_filtered["strategy"] == selected_strategy]

if selected_status != "All":
    df_filtered = df_filtered[df_filtered["status"] == selected_status]

if selected_label == "Pattern Found (Label = 1)":
    df_filtered = df_filtered[df_filtered["label"] == 1]
elif selected_label == "No Pattern (Label = 0)":
    df_filtered = df_filtered[df_filtered["label"] == 0]


# ------------------------------------------------------------
# TOP METRICS DASHBOARD
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_samples = len(df_filtered)
positive_samples = (df_filtered["label"] == 1).sum()
confirmed_breakouts = (df_filtered["status"] == "confirmed").sum()
forming_patterns = (df_filtered["status"] == "forming").sum()

col1.metric("Filtered Samples", f"{total_samples:,}")
col2.metric("Patterns Detected", f"{positive_samples:,}")
col3.metric("Confirmed Breakouts", f"{confirmed_breakouts:,}")
col4.metric("Forming Structures", f"{forming_patterns:,}")

st.divider()

# ------------------------------------------------------------
# TABBED INTERFACE
# ------------------------------------------------------------
tab_viewer, tab_summary = st.tabs(["🖼️ Image & Pattern Inspector", "📊 Strategy Summaries"])

with tab_viewer:
    if df_filtered.empty:
        st.warning("No samples match the selected filters.")
    else:
        st.subheader("Select Sample")
        
        # Selectbox to pick individual records
        sample_ids = df_filtered["sample_id"].tolist()
        selected_sample_id = st.selectbox("Sample ID", sample_ids)

        row = df_filtered[df_filtered["sample_id"] == selected_sample_id].iloc[0]

        st.markdown("---")
        
        left_col, right_col = st.columns([1, 1])

        # Left Column: Image Viewer
        with left_col:
            st.markdown("### 🖼️ CNN Input Image")
            
            # Reconstruct absolute image path
            img_relative_path = row["image_path"] 
            img_path = OUTPUT_ROOT / img_relative_path

            if img_path.exists():
                image = Image.open(img_path)
                st.image(
                    image, 
                    caption=f"CNN Input Stream (224x224 Clean Candle Chart) | Sample: {selected_sample_id}",
                    use_column_width=True
                )
            else:
                st.error(f"Image not found at path:\n`{img_path}`")

            # Review image check (if annotated charts were generated)
            review_rel_path = row.get("review_image_path")
            if pd.notna(review_rel_path) and str(review_rel_path).strip():
                review_path = OUTPUT_ROOT / str(review_rel_path)
                if review_path.exists():
                    st.markdown("### 🔍 Human Review Annotated Chart")
                    st.image(Image.open(review_path), use_column_width=True)

        # Right Column: Metadata & Trade Plan
        with right_col:
            st.markdown("### 📋 Sample Metadata & Trade Plan")

            badge_color = "green" if row["label"] == 1 else "gray"
            st.markdown(f"**Pattern Status:** :{badge_color}[{row['status'].upper()}]")
            
            st.markdown("#### Detection Details")
            st.write(f"- **Strategy:** `{row['strategy']}`")
            st.write(f"- **Direction:** `{row['direction']}`")
            st.write(f"- **Confidence:** `{row['confidence']:.2f}`")
            st.write(f"- **UTC Date:** `{row['day_utc']}`")
            st.write(f"- **Scan Window Size:** `{row['scan_window_bars']} bars`")
            st.write(f"- **Reason:** *{row['reason']}*")

            if row["label"] == 1:
                st.markdown("#### 🎯 Execution Trade Plan")
                plan_df = pd.DataFrame([
                    {"Parameter": "Entry Price", "Value": row.get("entry_price")},
                    {"Parameter": "Stop Loss", "Value": row.get("stop_loss")},
                    {"Parameter": "Target Price", "Value": row.get("target_price")},
                    {"Parameter": "Breakout Level", "Value": row.get("breakout_level")},
                    {"Parameter": "Risk / Reward", "Value": row.get("risk_reward")},
                    {"Parameter": "ATR at Entry", "Value": row.get("atr_at_entry")},
                ])
                st.table(plan_df)

# Strategy Summary Tab
with tab_summary:
    st.subheader("Strategy Level Data Distribution")
    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No overall summary file found.")

    st.markdown("---")
    st.subheader("Raw Filtered Manifest Data")
    st.dataframe(df_filtered, use_container_width=True, height=300)