import numpy as np
from numba import njit
import vectorbt as vbt


"""
Fast EMA: 
- Length = 30

Slow EMA: 
- Length = 150

Buy: Fast EMA corsses above slow EMA

Sell: Fast EMA crosses below slow EMA
"""


@njit
def get_signals(fast_ema, slow_ema):
    long_entries = (fast_ema > slow_ema) & (
        np.roll(fast_ema, 1) < np.roll(slow_ema, 1))
    short_entries = (fast_ema < slow_ema) & (
        np.roll(fast_ema, 1) > np.roll(slow_ema, 1))
    long_entries[0] = False
    short_entries[0] = False

    return long_entries, short_entries


def custom_indicator(high, low, close, fast_ema_length, slow_ema_length, atr_length, sl_coef, tpsl_ratio):
    EMA = vbt.IndicatorFactory.from_talib("EMA")
    ATR = vbt.IndicatorFactory.from_talib("ATR")

    fast_ema = EMA.run(close, fast_ema_length).real.to_numpy()
    slow_ema = EMA.run(close, slow_ema_length).real.to_numpy()
    atr = ATR.run(high, low, close, atr_length).real.to_numpy()

    long_entries, short_entries = get_signals(
        fast_ema,
        slow_ema
    )

    long_tp_price = close + (atr * sl_coef * tpsl_ratio)
    long_sl_price = close - (atr * sl_coef)

    short_tp_price = close - (atr * sl_coef * tpsl_ratio)
    short_sl_price = close + (atr * sl_coef)

    return long_entries, short_entries, long_tp_price, long_sl_price, short_tp_price, short_sl_price, fast_ema, slow_ema


indicator = vbt.IndicatorFactory(
    class_name="CustomIndicator",
    short_name="CI",
    input_names=["high", "low", "close"],
    param_names=["fast_ema_length", "slow_ema_length",
                 "atr_length", "sl_coef", "tpsl_ratio"],
    output_names=["long_entries", "short_entries",
                  "long_tp_price", "long_sl_price", "short_tp_price", "short_sl_price",
                  "fast_ema", "slow_ema"]
).from_apply_func(
    custom_indicator,
    fast_ema_length=30,
    slow_ema_length=100,
    atr_length=14,
    sl_coef=1.5,
    tpsl_ratio=2.0
)
