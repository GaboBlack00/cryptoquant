import numpy as np
import pandas as pd
import pytest

from src.utils.indicators import Indicators


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


class TestIndicatorsInit:
    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            Indicators(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            Indicators(df)

    def test_valid_df(self, sample_df):
        ind = Indicators(sample_df)
        assert ind.df is not sample_df
        assert len(ind.df) == 8


class TestSMA:
    def test_sma_basic(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.sma(period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx((100 + 102 + 101) / 3)
        assert result.iloc[3] == pytest.approx((102 + 101 + 103) / 3)

    def test_sma_all_periods(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.sma(period=1)
        assert not result.isna().any()
        assert result.iloc[0] == 100.0


class TestEMA:
    def test_ema_first_value(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.ema(period=3)
        assert result.iloc[0] == 100.0

    def test_ema_length(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.ema(period=3)
        assert len(result) == 8


class TestRSI:
    def test_rsi_bounds(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.rsi(period=3)
        valid = result.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_rsi_constant_series(self):
        df = pd.DataFrame(
            {
                "open": [50.0] * 20,
                "high": [51.0] * 20,
                "low": [49.0] * 20,
                "close": [50.0] * 20,
                "volume": [1000] * 20,
            }
        )
        ind = Indicators(df)
        result = ind.rsi(period=14)
        valid = result.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()


class TestBollingerBands:
    def test_bbands_columns(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.bbands(period=3)
        assert "upper_band" in result.columns
        assert "middle_band" in result.columns
        assert "lower_band" in result.columns

    def test_bbands_upper_above_lower(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.bbands(period=3).dropna()
        assert (result["upper_band"] >= result["middle_band"]).all()
        assert (result["middle_band"] >= result["lower_band"]).all()

    def test_bbands_middle_is_sma(self, sample_df):
        ind = Indicators(sample_df)
        bbands = ind.bbands(period=3)
        sma = ind.sma(period=3)
        pd.testing.assert_series_equal(bbands["middle_band"], sma, check_names=False)


class TestATR:
    def test_atr_no_nan_after_warmup(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.atr(period=3)
        assert result.iloc[0:2].isna().all()
        assert not result.iloc[3:].isna().any()

    def test_atr_non_negative(self, sample_df):
        ind = Indicators(sample_df)
        result = ind.atr(period=3).dropna()
        assert (result >= 0).all()
