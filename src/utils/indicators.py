import pandas as pd

REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}


class Indicators:
    """Class for computing technical indicators for financial analysis.

    Attributes:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume).
    """

    def __init__(self, df: pd.DataFrame):
        """Initializes the Indicators class with a financial data DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.

        Raises:
            ValueError: If the DataFrame is empty or does not contain
                the required columns (open, high, low, close, volume).
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not REQUIRED_OHLCV.issubset(df.columns):
            missing = REQUIRED_OHLCV - set(df.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")
        self.df = df.copy()

    def sma(self, period: int = 20, type_data: str = "close") -> pd.Series:
        """Computes the Simple Moving Average (SMA).

        Args:
            period (int, optional): Number of periods for the calculation.
                Must be greater than 0. Defaults to 20.
            type_data (str, optional): DataFrame column to use for calculation.
                Defaults to 'close'.

        Returns:
            pd.Series: Series with the SMA values.
                The first (period-1) values will be NaN.
        """
        return self.df[type_data].rolling(window=period).mean()

    def ema(self, period: int = 20, type_data: str = "close") -> pd.Series:
        """Computes the Exponential Moving Average (EMA).

        Args:
            period (int, optional): Number of periods for the smoothing factor.
                Must be greater than 0. Defaults to 20.
            type_data (str, optional): DataFrame column to use for calculation.
                Defaults to 'close'.

        Returns:
            pd.Series: Series with the EMA values.
        """
        return self.df[type_data].ewm(span=period, adjust=False).mean()

    def rsi(self, period: int = 14, type_data: str = "close") -> pd.Series:
        """Computes the Relative Strength Index (RSI).

        Args:
            period (int, optional): Number of periods for the calculation.
                Defaults to 14.
            type_data (str, optional): DataFrame column to use for calculation.
                Defaults to 'close'.

        Returns:
            pd.Series: Series with the RSI values (0-100).
        """
        delta = self.df[type_data].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def bbands(
        self, period: int = 20, std_dev: int = 2, type_data: str = "close"
    ) -> pd.DataFrame:
        """Computes Bollinger Bands.

        Args:
            period (int, optional): Number of periods for the moving average.
                Defaults to 20.
            std_dev (int, optional): Number of standard deviations for the bands.
                Defaults to 2.
            type_data (str, optional): DataFrame column to use for calculation.
                Defaults to 'close'.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - 'upper_band': Upper band (SMA + std_dev * standard deviation)
                - 'middle_band': Middle band (SMA)
                - 'lower_band': Lower band (SMA - std_dev * standard deviation)
        """
        sma = self.sma(period, type_data)
        rolling_std = self.df[type_data].rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return pd.DataFrame(
            {"upper_band": upper_band, "middle_band": sma, "lower_band": lower_band}
        )

    def atr(self, period: int = 14) -> pd.Series:
        """Computes the Average True Range (ATR).

        Args:
            period (int, optional): Number of periods for the True Range moving average.
                Defaults to 14.

        Returns:
            pd.Series: Series with the ATR values.
                The first 'period' values will be NaN.
        """
        high_low = self.df["high"] - self.df["low"]
        high_prev_close = abs(self.df["high"] - self.df["close"].shift(1))
        low_prev_close = abs(self.df["low"] - self.df["close"].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
