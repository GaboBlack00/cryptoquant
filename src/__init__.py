"""
CryptoQuant - Modular quantitative trading toolkit.

This package provides tools for fetching market data, computing technical
indicators, implementing rule-based trading strategies, and backtesting
with professional-grade risk management and performance metrics.
"""

__version__ = "1.0.0"

from .data import fetch_bitunix_data
from .utils import Indicators
from .strategies import Strategy
from .backtesting import Backtesting

__all__ = [
    "fetch_bitunix_data",
    "Indicators",
    "Strategy",
    "Backtesting",
]
