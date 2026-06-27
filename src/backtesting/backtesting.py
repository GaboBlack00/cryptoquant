import pandas as pd
import numpy as np
from math import sqrt
from ..utils.indicators import Indicators


class Backtesting:
    """Backtesting engine with risk management (SL, TP, trailing stop).

    Simulates trade execution over historical data and computes a full set
    of performance metrics adapted for cryptocurrency markets.

    .. note::
        Stop-loss, take-profit, and trailing-stop checks use the **close**
        price of each candle. In real trading, intra-candle price could
        trigger a stop even if the close does not, so this implementation
        may understate actual risk.

    Args:
        df: DataFrame with OHLCV data and a 'signal' column (1=buy, -1=sell, 0=hold).
        ind: Indicators instance for technical calculations (e.g. ATR).
        interval: Candle interval string ('1', '5', '15', ..., 'D', etc.).
        sl_pct: Stop-loss percentage (default 3%).
        tp_pct: Take-profit percentage (default 6%).
        tsl_pct: Trailing stop-loss percentage (default 2%).
        tsl_atr: Use ATR-based trailing stop instead of fixed percentage.
        initial_capital: Starting capital for the backtest (default 10000.0).
        fee_pct: Trading fee percentage per side (default 0.1). Total round-trip
            cost is 2 * fee_pct.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ind: Indicators,
        interval: str,
        sl_pct: float = 3.0,
        tp_pct: float = 6.0,
        tsl_pct: float = 2.0,
        tsl_atr: bool = False,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.05,
    ):
        required = {"open", "high", "low", "close", "volume", "signal"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.df = df.copy()
        self.ind = ind
        self.interval = interval
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.tsl_pct = tsl_pct
        self.tsl_atr = tsl_atr
        self._trades_df = None
        self._equity_df = None

    def _enter_position(self, signal, price, i, atr_series):
        """Open a new position and return position state dict."""
        side = "long" if signal == 1 else "short"
        if side == "long":
            sl = price * (1 - self.sl_pct / 100)
            tp = price * (1 + self.tp_pct / 100)
            tsl = self._compute_tsl_entry(price, i, atr_series, side)
        else:
            sl = price * (1 + self.sl_pct / 100)
            tp = price * (1 - self.tp_pct / 100)
            tsl = self._compute_tsl_entry(price, i, atr_series, side)

        return {
            "side": side,
            "entry_time": i,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "tsl": tsl,
        }

    def _compute_tsl_entry(self, price, i, atr_series, side):
        """Compute the initial trailing stop level on entry."""
        if not self.tsl_atr:
            factor = (
                1 - self.tsl_pct / 100 if side == "long" else 1 + self.tsl_pct / 100
            )
            return price * factor

        atr_value = self._get_atr(i, atr_series)
        if side == "long":
            return price - (atr_value * self.tsl_pct / 100)
        else:
            return price + (atr_value * self.tsl_pct / 100)

    def _update_tsl(self, price, i, atr_series, side, current_tsl):
        """Update trailing stop to lock in gains."""
        if not self.tsl_atr:
            factor = (
                1 - self.tsl_pct / 100 if side == "long" else 1 + self.tsl_pct / 100
            )
            candidate = price * factor
            return (
                max(current_tsl, candidate)
                if side == "long"
                else min(current_tsl, candidate)
            )

        atr_value = self._get_atr(i, atr_series)
        if side == "long":
            candidate = price - (atr_value * self.tsl_pct / 100)
            return max(current_tsl, candidate)
        else:
            candidate = price + (atr_value * self.tsl_pct / 100)
            return min(current_tsl, candidate)

    def _get_atr(self, i, atr_series):
        """Safely extract a scalar ATR value at index i."""
        if atr_series is None:
            return 0
        val = atr_series.get(i, np.nan)
        if isinstance(val, pd.Series):
            val = val.iloc[0] if not val.empty else np.nan
        return val if not np.isnan(val) else 0

    def _check_exit(self, price, position):
        """Determine exit reason for a position. Returns None or a reason string."""
        side = position["side"]
        if side == "long":
            if price <= position["sl"]:
                return "sl"
            if price <= position["tsl"]:
                return "tsl"
            if price >= position["tp"]:
                return "tp"
        else:
            if price >= position["sl"]:
                return "sl"
            if price >= position["tsl"]:
                return "tsl"
            if price <= position["tp"]:
                return "tp"
        return None

    def calculate_sl_tp_tsl(self):
        """Run the backtest simulation.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - trades_df: every individual trade with entry/exit details.
                - equity_df: equity curve over time.
        """
        if self._trades_df is not None and self._equity_df is not None:
            return self._trades_df, self._equity_df

        trades = []
        position = None
        capital = self.initial_capital
        equity_curve = []
        atr_series = self.ind.atr(period=14) if self.tsl_atr else None

        for i, row in self.df.iterrows():
            signal = row["signal"]
            price = row["close"]

            # --- Open new position ---
            if position is None and signal != 0:
                position = self._enter_position(signal, price, i, atr_series)
                continue

            # --- Manage existing position ---
            if position is not None:
                # Update trailing stop
                position["tsl"] = self._update_tsl(
                    price, i, atr_series, position["side"], position["tsl"]
                )

                # Check exits
                exit_reason = self._check_exit(price, position)

                # Opposite signal also triggers exit
                opposite = -1 if position["side"] == "long" else 1
                if exit_reason is None and signal == opposite:
                    exit_reason = "opposite"

                if exit_reason:
                    ret = self._compute_return(price, position)
                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": i,
                            "entry_price": position["entry_price"],
                            "exit_price": price,
                            "side": position["side"],
                            "exit_reason": exit_reason,
                            "return": ret,
                        }
                    )
                    capital *= 1 + ret
                    equity_curve.append({"time": i, "equity": capital})
                    position = None

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        self._trades_df = trades_df
        self._equity_df = equity_df
        return trades_df, equity_df

    def _compute_return(self, exit_price, position):
        if position["side"] == "long":
            ret = (exit_price - position["entry_price"]) / position["entry_price"]
        else:
            ret = (position["entry_price"] - exit_price) / position["entry_price"]
        return ret - (2 * self.fee_pct / 100)

    def stats(self):
        """Compute a full set of performance metrics.

        Returns:
            dict with keys: initial_capital, final_capital, total_return,
            total_return_pct, cagr, sharpe_ratio, sortino_ratio, max_drawdown,
            num_trades, win_rate, profit_factor.
        """
        trades_df, equity_df = self.calculate_sl_tp_tsl()

        if equity_df.empty or trades_df.empty:
            raise ValueError(
                "Not enough trades were executed. Try a different interval, "
                "strategy parameters, or a different strategy."
            )

        initial_cap = self.initial_capital
        final_cap = equity_df["equity"].iloc[-1]
        total_return = (final_cap / initial_cap) - 1
        total_return_pct = total_return * 100

        # CAGR
        start_date = equity_df["time"].iloc[0]
        end_date = equity_df["time"].iloc[-1]
        n_years = (end_date - start_date).days / 365.25
        cagr = (final_cap / initial_cap) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe & Sortino (crypto: hourly resample, 365*24 periods, 0% risk-free)
        hourly = equity_df.set_index("time").resample("1h").last().ffill()
        hourly_returns = hourly["equity"].pct_change().dropna()
        periods = 365 * 24

        if len(hourly_returns) == 0:
            raise ValueError("Not enough data for hourly return computation.")

        rf = 0.0
        std = hourly_returns.std()
        sharpe = (hourly_returns.mean() - rf) / std * sqrt(periods) if std != 0 else 0

        neg = hourly_returns[hourly_returns < 0]
        neg_std = neg.std()
        if len(neg) > 0 and neg_std != 0:
            sortino = (hourly_returns.mean() - rf) / neg_std * sqrt(periods)
        else:
            sortino = float("inf") if hourly_returns.mean() > rf else 0

        # Drawdown
        cum_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cum_max) / cum_max
        max_dd = drawdown.min()

        num_trades = len(trades_df)
        if num_trades > 0:
            winners = trades_df[trades_df["return"] > 0]
            win_rate = (len(winners) / num_trades) * 100
            gross_profit = winners["return"].sum()
            gross_loss = abs(trades_df[trades_df["return"] < 0]["return"].sum())
            profit_factor = (
                (gross_profit / gross_loss)
                if gross_loss > 0
                else (float("inf") if gross_profit > 0 else 0)
            )
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return {
            "initial_capital": initial_cap,
            "final_capital": final_cap,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }
