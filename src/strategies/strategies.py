from typing import Union

import numpy as np
import pandas as pd

from ..utils.indicators import Indicators


def cross_up(series1: pd.Series, series2: Union[pd.Series, int, float]) -> pd.Series:
    """Detect when series1 crosses above series2 (Series or scalar)."""
    if isinstance(series1, pd.Series) and isinstance(series2, pd.Series):
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    elif isinstance(series1, pd.Series) and isinstance(series2, (int, float)):
        return (series1.shift(1) < series2) & (series1 > series2)
    else:
        raise ValueError(
            "Both arguments must be pandas Series, or the first must be a Series "
            "and the second a numeric value."
        )


def cross_down(series1: pd.Series, series2: Union[pd.Series, int, float]) -> pd.Series:
    """Detect when series1 crosses below series2 (Series or scalar)."""
    if isinstance(series1, pd.Series) and isinstance(series2, pd.Series):
        return (series1.shift(1) > series2.shift(1)) & (series1 < series2)
    elif isinstance(series1, pd.Series) and isinstance(series2, (int, float)):
        return (series1.shift(1) > series2) & (series1 < series2)
    else:
        raise ValueError(
            "Both arguments must be pandas Series, or the first must be a Series "
            "and the second a numeric value."
        )


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class Strategy:
    """Rule-based trading strategy builder.

    Each strategy method returns a DataFrame with the computed signals:
        1  → Buy
       -1  → Sell
        0  → Hold

    Attributes:
        df (pd.DataFrame): Market data (OHLCV).
        ind (Indicators): Technical indicator calculator.
    """

    def __init__(self, df: pd.DataFrame):
        required = REQUIRED_COLUMNS
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")
        self.df = df.copy()
        self.ind = Indicators(df)

    def ema_crossover_volume(
        self, short_period: int = 20, long_period: int = 50, volume_period: int = 20
    ) -> pd.DataFrame:
        """EMA crossover strategy confirmed by above-average volume.

        Generates a buy signal when the short EMA crosses above the long EMA
        and volume exceeds its moving average. Sell on the opposite cross.
        """
        result = self.df.copy()
        result[f"EMA_{short_period}"] = self.ind.ema(period=short_period)
        result[f"EMA_{long_period}"] = self.ind.ema(period=long_period)
        result["MA_VOLUME"] = self.ind.sma(period=volume_period, type_data="volume")

        result["signal"] = np.where(
            (cross_up(result[f"EMA_{short_period}"], result[f"EMA_{long_period}"]))
            & (result["volume"] > result["MA_VOLUME"]),
            1,
            np.where(
                (cross_down(result[f"EMA_{short_period}"], result[f"EMA_{long_period}"]))
                & (result["volume"] > result["MA_VOLUME"]),
                -1,
                0,
            ),
        )

        result = result.dropna()
        cols = ["open", "high", "low", "close", "volume",
                f"EMA_{short_period}", f"EMA_{long_period}", "MA_VOLUME", "signal"]
        return result[cols].copy()

    def ema_rsi(
        self,
        ema_period: int = 20,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
    ) -> pd.DataFrame:
        """EMA trend filter + RSI extremes strategy.

        Buy when price is above the EMA and RSI crosses up from oversold.
        Sell when price is below the EMA and RSI crosses down from overbought.
        """
        result = self.df.copy()
        result["ema"] = self.ind.ema(period=ema_period)
        result["rsi"] = self.ind.rsi(period=rsi_period)

        result["signal"] = np.where(
            (result["close"] > result["ema"]) & (cross_up(result["rsi"], rsi_oversold)),
            1,
            np.where(
                (result["close"] < result["ema"])
                & (cross_down(result["rsi"], rsi_overbought)),
                -1,
                0,
            ),
        )

        return result[["open", "high", "low", "close", "ema", "rsi", "signal"]].copy()

    def bbands_rsi(self, bbands_period: int = 20, rsi_period: int = 14) -> pd.DataFrame:
        """Bollinger Bands bounce + RSI momentum strategy.

        Buy when price crosses above the lower band. Sell when price crosses
        below the upper band.
        """
        result = self.df.copy()
        bb = self.ind.bbands(period=bbands_period)
        result["lower_band"] = bb["lower_band"]
        result["upper_band"] = bb["upper_band"]
        result["rsi"] = self.ind.rsi(period=rsi_period)

        result["signal"] = np.where(
            cross_up(result["close"], result["lower_band"]),
            1,
            np.where(cross_down(result["close"], result["upper_band"]), -1, 0),
        )

        return result[["open", "high", "low", "close", "lower_band", "upper_band", "rsi", "signal"]].copy()
