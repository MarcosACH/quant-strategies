import numpy as np
from numba import njit


class TechnicalSignalUtils:
    """
    Utility class containing common technical analysis signal detection functions.

    This class provides reusable, high-performance functions for detecting
    common trading signals like crossovers, threshold breaks, etc.
    All methods are static and Numba-compiled for optimal performance.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    @njit
    def above_threshold(series: np.ndarray, threshold: float) -> np.ndarray:
        """
        Check if series is above threshold after being below.

        Args:
            series: Time series to check
            threshold: Threshold value

        Returns:
            Boolean array indicating threshold crossing events
        """
        return (series > threshold) & (np.roll(series, 1) <= threshold)

    @staticmethod
    @njit
    def below_threshold(series: np.ndarray, threshold: float) -> np.ndarray:
        """
        Check if series is below threshold after being above.

        Args:
            series: Time series to check
            threshold: Threshold value

        Returns:
            Boolean array indicating threshold crossing events
        """
        return (series < threshold) & (np.roll(series, 1) >= threshold)
