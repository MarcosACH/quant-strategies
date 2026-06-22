import numpy as np
from numba import njit
import vectorbt as vbt


"""
Long: Fast EMA > Slow EMA for consecutive back candles, Close < Lower BBand
Long SL: close - (atr * sl_coef)
Long TP: close + (atr * sl_coef * tpsl_ratio)

Short: Fast EMA < Slow EMA for consecutive back candles, Close > Upper BBand
Short SL: close + (atr * sl_coef)
Short TP: close - (atr * sl_coef * tpsl_ratio)
"""


@njit
def get_signals(close, upper_bband, lower_bband, trend_confirmation):
    up_trend = trend_confirmation == 1
    down_trend = trend_confirmation == -1

    price_below_lower_bb = close < lower_bband
    price_above_upper_bb = close > upper_bband

    long_entries = up_trend & price_below_lower_bb
    short_entries = down_trend & price_above_upper_bb

    return long_entries, short_entries


def custom_indicator(high, low, close, fast_ema_length, slow_ema_length, bbands_length, bbands_stddev, backcandles, atr_length, sl_coef, tpsl_ratio):
    EMA = vbt.IndicatorFactory.from_talib("EMA")
    BBANDS = vbt.IndicatorFactory.from_talib("BBANDS")
    ATR = vbt.IndicatorFactory.from_talib("ATR")

    fast_ema = EMA.run(close, fast_ema_length).real.to_numpy()
    slow_ema = EMA.run(close, slow_ema_length).real.to_numpy()
    bbands = BBANDS.run(close, bbands_length, bbands_stddev)
    upper_bband = bbands.upperband.to_numpy()
    lower_bband = bbands.lowerband.to_numpy()
    atr = ATR.run(high, low, close, atr_length).real.to_numpy()

    trend = np.where(fast_ema > slow_ema, 1,
                     np.where(fast_ema < slow_ema, -1, 0))

    kernel = np.ones(backcandles, dtype=np.float64)

    def rolling_convolve(arr, kernel):
        if arr.ndim == 1:
            return np.convolve(arr, kernel, mode="valid")
        else:
            return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, arr)

    uptrend_conv = rolling_convolve(trend == 1, kernel)
    uptrend_confirmed = uptrend_conv == backcandles

    downtrend_conv = rolling_convolve(trend == -1, kernel)
    downtrend_confirmed = downtrend_conv == backcandles

    trend_confirmation = np.zeros_like(trend)

    start_idx = backcandles - 1
    trend_confirmation[start_idx:start_idx + len(uptrend_confirmed)] = np.where(
        uptrend_confirmed, 1, np.where(downtrend_confirmed, -1, 0)
    )

    long_entries, short_entries = get_signals(
        close,
        upper_bband,
        lower_bband,
        trend_confirmation
    )

    long_tp_price = close + (atr * sl_coef * tpsl_ratio)
    long_sl_price = close - (atr * sl_coef)

    short_tp_price = close - (atr * sl_coef * tpsl_ratio)
    short_sl_price = close + (atr * sl_coef)

    return long_entries, short_entries, long_tp_price, long_sl_price, short_tp_price, short_sl_price, fast_ema, slow_ema, upper_bband, lower_bband


indicator = vbt.IndicatorFactory(
    class_name="CustomIndicator",
    short_name="CI",
    input_names=["high", "low", "close"],
    param_names=["fast_ema_length", "slow_ema_length",
                 "bbands_length", "bbands_stddev", "backcandles", "atr_length", "sl_coef", "tpsl_ratio"],
    output_names=["long_entries", "short_entries",
                  "long_tp_price", "long_sl_price", "short_tp_price", "short_sl_price",
                  "fast_ema", "slow_ema", "upper_bband", "lower_bband"]
).from_apply_func(
    custom_indicator,
    fast_ema_length=20,
    slow_ema_length=50,
    bbands_length=14,
    bbands_stddev=2.0,
    backcandles=20,
    atr_length=14,
    sl_coef=1.5,
    tpsl_ratio=2.0
)
