import numpy as np
from numba import njit
import vectorbt as vbt


"""
Long: Close > EMA, Close > Lower BBand after Close < Lower BBand
Long SL: close - (atr * sl_coef)
Long TP: close + (atr * sl_coef * tpsl_ratio)

Short: Close < EMA, Close < Upper BBand after Close > Upper BBand
Short SL: close + (atr * sl_coef)
Short TP: close - (atr * sl_coef * tpsl_ratio)
"""


@njit
def get_signals(close, upper_bband, lower_bband, trend):
    up_trend = trend == 1
    down_trend = trend == -1

    close_crossover_lower_bb = (close > lower_bband) & (
        np.roll(close, 1) < np.roll(lower_bband, 1))
    close_crossunder_upper_bb = (close < upper_bband) & (
        np.roll(close, 1) > np.roll(upper_bband, 1))
    close_crossover_lower_bb[0] = False
    close_crossunder_upper_bb[0] = False

    long_entries = up_trend & close_crossover_lower_bb
    short_entries = down_trend & close_crossunder_upper_bb

    return long_entries, short_entries


def custom_indicator(high, low, close, ema_length, bbands_length, bbands_stddev, atr_length, sl_coef, tpsl_ratio):
    EMA = vbt.IndicatorFactory.from_talib("EMA")
    BBANDS = vbt.IndicatorFactory.from_talib("BBANDS")
    ATR = vbt.IndicatorFactory.from_talib("ATR")

    ema = EMA.run(close, ema_length).real.to_numpy()
    bbands = BBANDS.run(close, bbands_length, bbands_stddev)
    upper_bband = bbands.upperband.to_numpy()
    lower_bband = bbands.lowerband.to_numpy()
    atr = ATR.run(high, low, close, atr_length).real.to_numpy()

    trend = np.where(close > ema, 1,
                     np.where(close < ema, -1, 0))

    long_entries, short_entries = get_signals(
        close,
        upper_bband,
        lower_bband,
        trend
    )

    long_tp_price = close + (atr * sl_coef * tpsl_ratio)
    long_sl_price = close - (atr * sl_coef)

    short_tp_price = close - (atr * sl_coef * tpsl_ratio)
    short_sl_price = close + (atr * sl_coef)

    return long_entries, short_entries, long_tp_price, long_sl_price, short_tp_price, short_sl_price, ema, upper_bband, lower_bband


indicator = vbt.IndicatorFactory(
    class_name="CustomIndicator",
    short_name="CI",
    input_names=["high", "low", "close"],
    param_names=["ema_length", "bbands_length",
                 "bbands_stddev", "atr_length", "sl_coef", "tpsl_ratio"],
    output_names=["long_entries", "short_entries",
                  "long_tp_price", "long_sl_price", "short_tp_price", "short_sl_price",
                  "ema", "upper_bband", "lower_bband"]
).from_apply_func(
    custom_indicator,
    ema_length=150,
    bbands_length=30,
    bbands_stddev=2.0,
    atr_length=14,
    sl_coef=1.5,
    tpsl_ratio=2.0
)
