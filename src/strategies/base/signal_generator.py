from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from numba import njit


class SignalGenerator(ABC):
    """
    Abstract base class for signal generation components.

    This class defines the interface for generating trading signals
    from market data and technical indicators.
    """

    @abstractmethod
    def generate_signals(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate long and short entry signals.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (long_entries, short_entries) boolean arrays
        """
        pass

    @abstractmethod
    def calculate_exit_prices(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        """
        Calculate take profit and stop loss prices.

        Returns:
            Tuple containing TP and SL prices for long and short positions
        """
        pass


class TechnicalSignalGenerator(SignalGenerator):
    """
    Base class for technical indicator-based signal generation.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

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
