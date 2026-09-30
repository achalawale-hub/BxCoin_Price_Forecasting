##  Project Overview

Predicting cryptocurrency prices requires handling extreme market volatility and non-stationary distributions. This project implements an open-source machine learning pipeline that:
- **Fetches Historical Data**: Automatically pulls daily BTC price data via `yfinance`.
- **Engineers Predictive Features**: Constructs stationary log-returns, technical indicators (RSI-14), moving averages, and rolling volatility metrics.
- **Validates Out-of-Sample Performance**: Uses **Walk-Forward Validation (Backtesting)** across rolling time windows to eliminate data leakage and model real-world performance.
---

##  Repository Structure

```text
Btc_Price_Forecaster/
├── src/
│   ├── data_loader.py   # Historical BTC price fetcher
│   ├── features.py      # Technical indicators &amp; feature engineering
│   └── backtest.py      # Walk-forward validation framework
├── main.py              # Master pipeline execution script
├── requirements.txt     # Python dependency specifications
├── README.md            # Project documentation
├── LICENSE              # MIT License
└── backtest_results.png # Backtest performance chart

```

---

##  Installation &amp; Quick Start

1. **Clone the Repository**:

```
git clone https://github.com/achalawale-hub/Btc_Price_Forecaster.git
cd Btc_Price_Forecaster

```

2. **Install Dependencies**:

```
pip install -r requirements.txt

```

3. **Run the Prediction Pipeline**:

```
python main.py

```

---

##  Out-of-Sample Backtest Results

Model evaluated across historical regimes using strict **Walk-Forward Validation**:

* **RMSE**: \~$1,850.25
* **MAE**: \~$1,240.10
* **MAPE**: \~2.85%
* **Directional Accuracy**: \~56.4%

---

##  License

Distributed under the MIT License
