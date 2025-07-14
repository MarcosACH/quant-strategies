"""
Numba-compiled technical analysis functions for high-performance signal detection.
These functions are designed to be used within @njit compiled strategies.
"""

import numpy as np
from numba import njit


@njit
def crossover(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Detect crossover events (series1 crosses above series2).

    Args:
        series1: First time series
        series2: Second time series

    Returns:
        Boolean array indicating crossover points
    """
    return (series1 > series2) & (np.roll(series1, 1) <= np.roll(series2, 1))


@njit
def crossunder(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Detect crossunder events (series1 crosses below series2).

    Args:
        series1: First time series
        series2: Second time series

    Returns:
        Boolean array indicating crossunder points
    """
    return (series1 < series2) & (np.roll(series1, 1) >= np.roll(series2, 1))


@njit
def above_threshold(series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Check if series crosses above threshold after being below.

    Args:
        series: Time series to check
        threshold: Threshold value

    Returns:
        Boolean array indicating threshold crossing events
    """
    return (series > threshold) & (np.roll(series, 1) <= threshold)


@njit
def below_threshold(series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Check if series crosses below threshold after being above.

    Args:
        series: Time series to check
        threshold: Threshold value

    Returns:
        Boolean array indicating threshold crossing events
    """
    return (series < threshold) & (np.roll(series, 1) >= threshold)
