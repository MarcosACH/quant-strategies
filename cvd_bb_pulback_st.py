import numpy as np
from numba import njit
import vectorbt as vbt


"""
Long: Cumulative Delta > Lower BBand after Cumulative Delta < Lower BBand
Long SL: close - (atr * sl_coef)
Long TP: close + (atr * sl_coef * tpsl_ratio)

Short: Cumulative Delta < Upper BBand after Cumulative Delta > Upper BBand
Short SL: close + (atr * sl_coef)
Short TP: close - (atr * sl_coef * tpsl_ratio)
"""


@njit
def get_signals(cumulative_volume_delta, upper_bband, lower_bband):
    long_entries = (cumulative_volume_delta > lower_bband) & (
        np.roll(cumulative_volume_delta, 1) < np.roll(lower_bband, 1))
    short_entries = (cumulative_volume_delta < upper_bband) & (
        np.roll(cumulative_volume_delta, 1) > np.roll(upper_bband, 1))
    long_entries[0] = False
    short_entries[0] = False

    return long_entries, short_entries


def custom_indicator(open, high, low, close, volume, bbands_length, bbands_stddev, cvd_length, atr_length, sl_coef, tpsl_ratio):
    BBANDS = vbt.IndicatorFactory.from_talib("BBANDS")
    ATR = vbt.IndicatorFactory.from_talib("ATR")

    volume_delta = np.where(close >= open, volume, -volume)

    volume_delta_flat = np.asarray(volume_delta).flatten()

    rolling_sum = np.convolve(
        volume_delta_flat, np.ones(cvd_length), mode="valid")

    cumulative_volume_delta = np.zeros_like(volume_delta_flat)
    cumulative_volume_delta[cvd_length - 1:] = rolling_sum

    cumulative_volume_delta = cumulative_volume_delta.reshape(
        volume_delta.shape)

    bbands = BBANDS.run(cumulative_volume_delta, bbands_length, bbands_stddev)
    upper_bband = bbands.upperband.to_numpy()
    lower_bband = bbands.lowerband.to_numpy()
    atr = ATR.run(high, low, close, atr_length).real.to_numpy()

    long_entries, short_entries = get_signals(
        cumulative_volume_delta,
        upper_bband,
        lower_bband
    )

    long_tp_price = close + (atr * sl_coef * tpsl_ratio)
    long_sl_price = close - (atr * sl_coef)

    short_tp_price = close - (atr * sl_coef * tpsl_ratio)
    short_sl_price = close + (atr * sl_coef)

    return long_entries, short_entries, long_tp_price, long_sl_price, short_tp_price, short_sl_price, volume_delta, cumulative_volume_delta, upper_bband, lower_bband, atr


indicator = vbt.IndicatorFactory(
    class_name="CustomIndicator",
    short_name="CI",
    input_names=["open", "high", "low", "close", "volume"],
    param_names=["bbands_length", "bbands_stddev",
                 "cvd_length", "atr_length", "sl_coef", "tpsl_ratio"],
    output_names=["long_entries", "short_entries",
                  "long_tp_price", "long_sl_price", "short_tp_price", "short_sl_price",
                  "volume_delta", "cumulative_volume_delta", "upper_bband", "lower_bband", "atr"]
).from_apply_func(
    custom_indicator,
    bbands_length=30,
    bbands_stddev=2.0,
    cvd_length=50,
    atr_length=14,
    sl_coef=2.5,
    tpsl_ratio=2.0
)
