# 📈 Gold AI Trader — Automated Pattern Scanner & Computer Vision Dataset Pipeline

An end-to-end algorithmic trading workspace designed for Gold (XAU/USD). This project scans multi-year historical intraday OHLC data for rule-based technical chart patterns, generates entry/exit risk-management parameters, exports clean candlestick images for Computer Vision (CNN) training, and provides an interactive Streamlit dashboard for dataset inspection.

---

## ✨ Features

* **Multi-Pattern Scanner Engine:** Detects 10 technical chart pattern strategies across variable scan window sizes (48 to 200 bars):
  * **Continuation:** Flags, Consolidations
  * **Reversal:** Cup & Handle, Double Bottom, Double Top, Inverse Head & Shoulders
  * **Trendline:** Ascending Triangle, Bullish Rectangle, Falling Wedge, Symmetrical Triangle
* **Automatic Risk Management Setup:** Calculates exact entry prices, stop-loss levels, take-profit targets, ATR-based risk parameters, and Risk-to-Reward (R:R) ratios for confirmed patterns.
* **Computer Vision (CNN) Dataset Pipeline:**
  * Generates uniform $224 \times 224$ clean candlestick charts mapped to specific daily endpoints.
  * Uses hard/soft disk linking to eliminate duplicate image storage across multiple strategies.
  * Outputs CSV manifests with ground-truth pattern labels (`label = 1` or `0`), status flags, and timestamps.
* **Interactive Streamlit Dashboard:** Filter samples by strategy, confidence, and status; inspect CNN input charts side-by-side with calculated trade plans.

---

## 📁 Repository Structure

```text
.
├── app.py                   # Streamlit visualization dashboard
├── Message.py               # Dataset generator & pattern scanning script
├── gold-ai-trader/
│   ├── chart_patterns/      # Pattern detection algorithms & trade logic
│   ├── data/raw/            # Historical market data (CSV)
│   └── research/            # Generated CNN datasets, manifests, & metadata
└── README.md
