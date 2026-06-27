# CryptoQuant — Quantitative Trading Backtesting Framework

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A **modular**, **extensible** Python framework for building and backtesting quantitative trading strategies on cryptocurrency data. Developed as a demonstration of clean, layered software architecture for algorithmic trading systems.

---

## Features

- **Data Layer** — Fetch historical OHLCV data from the Bitunix public API with automatic pagination and rate limiting
- **Technical Indicators** — SMA, EMA, RSI, Bollinger Bands, and ATR (easily extensible)
- **Rule-Based Strategies** — Three built-in strategies (EMA Crossover + Volume, EMA + RSI, Bollinger Bands + RSI)
- **Backtesting Engine** — Full trade simulation with Stop-Loss, Take-Profit, and Trailing Stop-Loss (percentage or ATR-based)
- **Performance Metrics** — Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor (crypto-adapted: hourly resampling, 365×24 periods, 0% risk-free rate)
- **Visualization** — Equity curve, drawdown curve, and buy/sell signals overlaid on price

---

## Architecture

```
cryptoquant/
├── notebooks/                # Jupyter notebooks
│   └── cryptoquant_demo.ipynb
├── src/                      # Modular source package
│   ├── data/                 # Data acquisition
│   │   └── bitunix_data.py   #   Bitunix API client
│   ├── utils/                # Technical analysis
│   │   └── indicators.py     #   Technical indicators
│   ├── strategies/           # Trading strategies
│   │   └── strategies.py     #   Rule-based strategies
│   └── backtesting/          # Performance evaluation
│       └── backtesting.py    #   Backtesting engine
├── README.md
├── requirements.txt
├── setup.py
└── LICENSE
```

The codebase follows a **layered architecture** with clear separation of concerns:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Data** | `src/data/` | External API communication, data cleaning |
| **Processing** | `src/utils/` | Pure computation, indicator math |
| **Strategy** | `src/strategies/` | Business logic, signal generation |
| **Evaluation** | `src/backtesting/` | Trade simulation, risk management, metrics |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/GaboBlack00/cryptoquant.git
cd cryptoquant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .

# 4. Launch the demo notebook
jupyter notebook notebooks/cryptoquant_demo.ipynb
```

---

## Usage Example

```python
from src import fetch_bitunix_data, Indicators, Strategy, Backtesting

# 1. Fetch 6 months of BTCUSDT 15-minute data
df = fetch_bitunix_data("BTCUSDT", interval="15")

# 2. Compute indicators
indicators = Indicators(df)

# 3. Apply strategy: EMA crossover confirmed by volume
df_signals = Strategy(df).ema_crossover_volume(
    short_period=20, long_period=50, volume_period=20
)

# 4. Backtest with risk management (SL=3%, TP=6%, TSL=2%)
backtest = Backtesting(
    df_signals, indicators, interval="15",
    sl_pct=3, tp_pct=6, tsl_pct=2
)
trades_df, equity_df = backtest.calculate_sl_tp_tsl()
stats = backtest.stats()

print(f"Return: {stats['total_return_pct']:.2f}% | Sharpe: {stats['sharpe_ratio']:.2f} | Trades: {stats['num_trades']}")
```

---

## Built-in Strategies

| Strategy | Method | Entry Signal | Exit Signal |
|----------|--------|-------------|-------------|
| **EMA Crossover + Volume** | `ema_crossover_volume()` | EMA(20) × EMA(50) crossover + above-average volume | Opposite crossover + volume confirmation |
| **EMA + RSI** | `ema_rsi()` | Price > EMA(20) + RSI crosses up from oversold | Price < EMA(20) + RSI crosses down from overbought |
| **Bollinger Bands + RSI** | `bbands_rsi()` | Price crosses above lower Bollinger Band | Price crosses below upper Bollinger Band |

---

## Example Results

Strategy: **EMA Crossover + Volume** on BTCUSDT (15-min candles, Jan–Jun 2025)

| Metric | Value |
|--------|-------|
| Initial Capital | $10,000.00 |
| Final Capital | $12,646.93 |
| **Total Return** | **+26.47%** |
| **CAGR** | **62%** |
| **Sharpe Ratio** | **1.66** |
| **Sortino Ratio** | **0.84** |
| **Max Drawdown** | **-13%** |
| **Number of Trades** | **123** |
| **Win Rate** | **33.33%** |
| **Profit Factor** | **1.35** |

---

## Requirements

- Python **3.12+**
- See [`requirements.txt`](requirements.txt) for the full list of dependencies

---

## Demo Notebook

Explore the full workflow interactively:
[`notebooks/cryptoquant_demo.ipynb`](notebooks/cryptoquant_demo.ipynb)

---

## Testing

Run the test suite with:

```bash
python3 -m pytest tests/ -v
```

The suite covers indicators, strategies, cross-detection logic, and the backtesting
engine (including fee impact, caching, and edge cases).

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make your changes and run the tests (`python3 -m pytest tests/`)
4. Commit and push your branch
5. Open a Pull Request

Please ensure all tests pass before submitting.

---

## Disclaimer

This project is for **educational and portfolio purposes only**. It is not financial advice. Past performance does not guarantee future results. Cryptocurrency trading carries substantial risk.
