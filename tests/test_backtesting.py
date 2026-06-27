import numpy as np
import pandas as pd
import pytest

from src.utils.indicators import Indicators
from src.backtesting.backtesting import Backtesting


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(np.random.randn(200) * 0.005))
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(200) * 0.001),
            "high": close * (1 + abs(np.random.randn(200) * 0.005)),
            "low": close * (1 - abs(np.random.randn(200) * 0.005)),
            "close": close,
            "volume": np.random.randint(800, 1200, 200),
            "signal": np.random.choice([-1, 0, 1], 200, p=[0.1, 0.8, 0.1]),
        },
        index=dates,
    )


class TestBacktestingInit:
    def test_missing_signal_column_raises(self):
        df = pd.DataFrame({"close": [1.0], "open": [1.0], "high": [1.1], "low": [0.9], "volume": [100]})
        ind = Indicators(df)
        with pytest.raises(ValueError, match="missing required columns"):
            Backtesting(df, ind, interval="1")

    def test_valid_init(self, sample_df):
        ind = Indicators(sample_df)
        bt = Backtesting(
            sample_df, ind, interval="1",
            initial_capital=50000.0, fee_pct=0.2,
        )
        assert bt.initial_capital == 50000.0
        assert bt.fee_pct == 0.2
        assert bt.sl_pct == 3.0
        assert bt.tp_pct == 6.0
        assert bt.tsl_pct == 2.0
        assert bt.tsl_atr is False


class TestBacktestRun:
    def test_calculate_returns_dataframes(self, sample_df):
        ind = Indicators(sample_df)
        bt = Backtesting(sample_df, ind, interval="1")
        trades, equity = bt.calculate_sl_tp_tsl()
        assert isinstance(trades, pd.DataFrame)
        assert isinstance(equity, pd.DataFrame)

    def test_stats_contains_expected_keys(self, sample_df):
        ind = Indicators(sample_df)
        bt = Backtesting(sample_df, ind, interval="1")
        stats = bt.stats()
        expected = {
            "initial_capital", "final_capital", "total_return",
            "total_return_pct", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "num_trades", "win_rate", "profit_factor",
        }
        assert expected.issubset(stats.keys())

    def test_stats_no_trades_raises(self):
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000] * 50,
                "signal": [0] * 50,
            },
            index=pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC"),
        )
        ind = Indicators(df)
        bt = Backtesting(df, ind, interval="1")
        with pytest.raises(ValueError, match="Not enough trades"):
            bt.stats()

    def test_caching_avoids_recomputation(self, sample_df):
        ind = Indicators(sample_df)
        bt = Backtesting(sample_df, ind, interval="1")
        trades1, equity1 = bt.calculate_sl_tp_tsl()
        trades2, equity2 = bt.calculate_sl_tp_tsl()
        assert trades1 is trades2
        assert equity1 is equity2

    def test_fee_reduces_return(self, sample_df):
        ind = Indicators(sample_df)
        bt_no_fee = Backtesting(sample_df, ind, interval="1", fee_pct=0.0)
        bt_with_fee = Backtesting(sample_df, ind, interval="1", fee_pct=0.5)
        stats_no_fee = bt_no_fee.stats()
        stats_with_fee = bt_with_fee.stats()
        assert stats_with_fee["total_return"] <= stats_no_fee["total_return"]

    def test_trades_have_exit_reason_column(self, sample_df):
        ind = Indicators(sample_df)
        bt = Backtesting(sample_df, ind, interval="1")
        trades, _ = bt.calculate_sl_tp_tsl()
        if not trades.empty:
            assert "exit_reason" in trades.columns
