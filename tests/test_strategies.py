import pandas as pd
import pytest

from src.strategies.strategies import cross_down, cross_up, Strategy


class TestCrossUp:
    def test_cross_up_detected(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([5, 4, 3.5, 2, 1])
        result = cross_up(a, b)
        assert result.iloc[3]
        assert not result.iloc[0]
        assert not result.iloc[4]

    def test_cross_up_scalar(self):
        a = pd.Series([1, 2, 3, 4, 5])
        result = cross_up(a, 2.5)
        assert not result.iloc[0]
        assert not result.iloc[1]
        assert result.iloc[2]

    def test_cross_up_no_cross(self):
        a = pd.Series([5, 4, 3, 2, 1])
        b = pd.Series([1, 2, 3, 4, 5])
        result = cross_up(a, b)
        assert not result.any()

    def test_cross_up_invalid_args(self):
        with pytest.raises(ValueError):
            cross_up("not_a_series", "also_not_a_series")


class TestCrossDown:
    def test_cross_down_detected(self):
        a = pd.Series([5, 4, 3.5, 2, 1])
        b = pd.Series([1, 2, 3, 4, 5])
        result = cross_down(a, b)
        assert result.iloc[3]
        assert not result.iloc[0]
        assert not result.iloc[4]

    def test_cross_down_scalar(self):
        a = pd.Series([5, 4, 3.5, 2, 1])
        result = cross_down(a, 2.5)
        assert not result.iloc[0]
        assert not result.iloc[1]
        assert result.iloc[3]

    def test_cross_down_no_cross(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([5, 4, 3, 2, 1])
        result = cross_down(a, b)
        assert not result.any()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0],
            "high": [101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 107.0, 109.0],
            "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 107.0],
            "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0],
            "volume": [1000, 1100, 1050, 1200, 1300, 1150, 1250, 1400],
        }
    )


class TestStrategy:
    def test_init_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            Strategy(df)

    def test_ema_crossover_volume_has_signal_column(self, sample_df):
        s = Strategy(sample_df)
        result = s.ema_crossover_volume(short_period=3, long_period=5, volume_period=3)
        assert "signal" in result.columns
        assert result["signal"].dropna().isin([-1, 0, 1]).all()

    def test_ema_rsi_has_signal_column(self, sample_df):
        s = Strategy(sample_df)
        result = s.ema_rsi(ema_period=3, rsi_period=3)
        assert "signal" in result.columns
        assert result["signal"].dropna().isin([-1, 0, 1]).all()

    def test_bbands_rsi_has_signal_column(self, sample_df):
        s = Strategy(sample_df)
        result = s.bbands_rsi(bbands_period=3, rsi_period=3)
        assert "signal" in result.columns
        assert result["signal"].dropna().isin([-1, 0, 1]).all()

    def test_returns_copy(self, sample_df):
        s = Strategy(sample_df)
        result = s.ema_crossover_volume()
        assert result is not sample_df
