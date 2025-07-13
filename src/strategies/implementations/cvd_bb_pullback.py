import numpy as np
import vectorbt as vbt
from numba import njit
from typing import Dict, Any, List, Tuple
from vectorbt.portfolio.nb import order_nb, NoOrder
from vectorbt.portfolio.enums import SizeType, Direction

from ..base.strategy_base import StrategyBase
from ..base.risk_manager import _calculate_position_size


class CVDBBPullbackStrategy(StrategyBase):
    """
    Cumulative Volume Delta Bollinger Band Pullback Strategy.

    Long Signal: CVD > Lower BB after CVD < Lower BB
    Short Signal: CVD < Upper BB after CVD > Upper BB

    Stop Loss: ATR-based
    Take Profit: Risk-reward ratio based on stop loss
    """

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values for the strategy."""
        return {
            'bbands_length': 30,
            'bbands_stddev': 2.0,
            'cvd_length': 50,
            'atr_length': 14,
            'sl_coef': 2.5,
            'tpsl_ratio': 2.0
        }

    @property
    def param_ranges(self) -> Dict[str, List]:
        """Parameter ranges for optimization."""
        return {
            'bbands_length': [20, 25, 30, 35, 40],
            'bbands_stddev': [1.5, 2.0, 2.5, 3.0],
            'cvd_length': [30, 40, 50, 60, 70],
            'atr_length': [10, 14, 20, 28],
            'sl_coef': [1.5, 2.0, 2.5, 3.0, 3.5],
            'tpsl_ratio': [1.5, 2.0, 2.5, 3.0]
        }

    def create_indicator(self):
        """Create the vectorbt indicator factory for this strategy."""
        if self.indicator is None:
            self.indicator = vbt.IndicatorFactory(
                class_name="CVDBBIndicator",
                short_name="CVDBB",
                input_names=["open", "high", "low", "close", "volume"],
                param_names=["bbands_length", "bbands_stddev", "cvd_length",
                             "atr_length", "sl_coef", "tpsl_ratio"],
                output_names=["long_entries", "short_entries",
                              "long_tp_price", "long_sl_price", "short_tp_price", "short_sl_price",
                              "volume_delta", "cumulative_volume_delta", "upper_bband", "lower_bband", "atr"]
            ).from_apply_func(
                self._custom_indicator,
                **self.default_params
            )

        return self.indicator

    def get_order_func_nb(self):
        """Get the numba-compiled order function for this strategy."""
        return order_func_nb

    @staticmethod
    def _custom_indicator(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        bbands_length: int,
        bbands_stddev: float,
        cvd_length: int,
        atr_length: int,
        sl_coef: float,
        tpsl_ratio: float
    ) -> Tuple[np.ndarray, ...]:
        """
        Custom indicator calculation for CVD Bollinger Band strategy.

        Args:
            open_prices: Open prices array
            high_prices: High prices array
            low_prices: Low prices array
            close_prices: Close prices array
            volume: Volume array
            bbands_length: Bollinger Bands period
            bbands_stddev: Bollinger Bands standard deviation
            cvd_length: Cumulative Volume Delta period
            atr_length: ATR period
            sl_coef: Stop loss coefficient
            tpsl_ratio: Take profit to stop loss ratio

        Returns:
            Tuple of indicator outputs
        """
        # Initialize vectorbt indicators
        BBANDS = vbt.IndicatorFactory.from_talib("BBANDS")
        ATR = vbt.IndicatorFactory.from_talib("ATR")

        # Calculate volume delta
        volume_delta = np.where(close_prices >= open_prices, volume, -volume)

        # Calculate cumulative volume delta using rolling sum
        volume_delta_flat = np.asarray(volume_delta).flatten()
        rolling_sum = np.convolve(
            volume_delta_flat, np.ones(cvd_length), mode="valid"
        )

        cumulative_volume_delta = np.zeros_like(volume_delta_flat)
        cumulative_volume_delta[cvd_length - 1:] = rolling_sum
        cumulative_volume_delta = cumulative_volume_delta.reshape(
            volume_delta.shape)

        # Calculate Bollinger Bands on CVD
        bbands = BBANDS.run(cumulative_volume_delta,
                            bbands_length, bbands_stddev)
        upper_bband = bbands.upperband.values
        lower_bband = bbands.lowerband.values

        # Calculate ATR
        atr = ATR.run(high_prices, low_prices,
                      close_prices, atr_length).real.values

        # Generate signals
        long_entries, short_entries = CVDBBPullbackStrategy._get_signals(
            cumulative_volume_delta, upper_bband, lower_bband
        )

        # Calculate exit prices
        long_tp_price = close_prices + (atr * sl_coef * tpsl_ratio)
        long_sl_price = close_prices - (atr * sl_coef)
        short_tp_price = close_prices - (atr * sl_coef * tpsl_ratio)
        short_sl_price = close_prices + (atr * sl_coef)

        return (
            long_entries, short_entries, long_tp_price, long_sl_price,
            short_tp_price, short_sl_price, volume_delta, cumulative_volume_delta,
            upper_bband, lower_bband, atr
        )

    @staticmethod
    @njit
    def _get_signals(
        cumulative_volume_delta: np.ndarray,
        upper_bband: np.ndarray,
        lower_bband: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals based on CVD and Bollinger Bands.

        Args:
            cumulative_volume_delta: CVD values
            upper_bband: Upper Bollinger Band
            lower_bband: Lower Bollinger Band

        Returns:
            Tuple of (long_entries, short_entries) boolean arrays
        """
        # Long: CVD crosses above lower BB after being below
        long_entries = (
            (cumulative_volume_delta > lower_bband) &
            (np.roll(cumulative_volume_delta, 1) < np.roll(lower_bband, 1))
        )

        # Short: CVD crosses below upper BB after being above
        short_entries = (
            (cumulative_volume_delta < upper_bband) &
            (np.roll(cumulative_volume_delta, 1) > np.roll(upper_bband, 1))
        )

        # Prevent signals on first bar (no previous data)
        long_entries[0] = False
        short_entries[0] = False

        return long_entries, short_entries

    def _validate_params(self):
        """Validate strategy-specific parameters."""
        super()._validate_params()

        # Validate parameter ranges
        if self.params['bbands_length'] < 5:
            raise ValueError("bbands_length must be >= 5")

        if self.params['bbands_stddev'] <= 0:
            raise ValueError("bbands_stddev must be > 0")

        if self.params['cvd_length'] < 1:
            raise ValueError("cvd_length must be >= 1")

        if self.params['atr_length'] < 1:
            raise ValueError("atr_length must be >= 1")

        if self.params['sl_coef'] <= 0:
            raise ValueError("sl_coef must be > 0")

        if self.params['tpsl_ratio'] <= 0:
            raise ValueError("tpsl_ratio must be > 0")


@njit
def order_func_nb(
    c,
    last_exits_state,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_tp_price: np.ndarray,
    long_sl_price: np.ndarray,
    short_tp_price: np.ndarray,
    short_sl_price: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    fee_decimal: float,
    sizing_method: str = "Value-based",
    risk_pct: float = np.nan,
    risk_nominal: float = np.nan,
    position_size_value: float = np.nan,
    min_size_value: float = 0.0001,
    max_size_value: float = np.inf,
    size_granularity: float = 0.0001
):
    """
    Numba-compiled order function for CVD BB Pullback strategy.

    This function handles order generation based on entry/exit signals
    and manages position sizing and risk management.
    """
    col, i, pos = c.col, c.i, c.position_now
    exits_state = last_exits_state[col]
    high, low, close = high_prices[i], low_prices[i], close_prices[i]

    # Calculate minimum and maximum position sizes in shares
    min_pos_size = min_size_value / close
    max_pos_size = max_size_value / close

    # Entry logic - only when no position
    if pos == 0:
        long_signal = long_entries[i]
        short_signal = short_entries[i] if not long_signal else False

        if long_signal:
            # Set exit prices for this position
            exits_state.active_tp_price = long_tp_price[i]
            exits_state.active_sl_price = long_sl_price[i]

            # Calculate position size
            pos_size = _calculate_position_size(
                sizing_method, risk_pct, risk_nominal, position_size_value,
                c.cash_now, close, exits_state.active_sl_price, fee_decimal
            )

            return order_nb(
                pos_size, close, SizeType.Value, Direction.LongOnly,
                fee_decimal, min_size=min_pos_size, max_size=max_pos_size,
                size_granularity=size_granularity
            )

        elif short_signal:
            # Set exit prices for this position
            exits_state.active_tp_price = short_tp_price[i]
            exits_state.active_sl_price = short_sl_price[i]

            # Calculate position size
            pos_size = _calculate_position_size(
                sizing_method, risk_pct, risk_nominal, position_size_value,
                c.cash_now, close, exits_state.active_sl_price, fee_decimal
            )

            return order_nb(
                pos_size, close, SizeType.Value, Direction.ShortOnly,
                fee_decimal, min_size=min_pos_size, max_size=max_pos_size,
                size_granularity=size_granularity
            )

        return NoOrder

    # Exit logic - when position exists
    sl_price, tp_price = exits_state.active_sl_price, exits_state.active_tp_price

    if pos > 0:  # Long position
        # Check stop loss
        if low <= sl_price:
            return order_nb(
                -np.inf, sl_price, SizeType.Amount, Direction.LongOnly, fee_decimal
            )
        # Check take profit
        elif high >= tp_price:
            return order_nb(
                -np.inf, tp_price, SizeType.Amount, Direction.LongOnly, fee_decimal
            )

    else:  # Short position (pos < 0)
        # Check stop loss
        if high >= sl_price:
            return order_nb(
                -np.inf, sl_price, SizeType.Amount, Direction.ShortOnly, fee_decimal
            )
        # Check take profit
        elif low <= tp_price:
            return order_nb(
                -np.inf, tp_price, SizeType.Amount, Direction.ShortOnly, fee_decimal
            )

    return NoOrder
