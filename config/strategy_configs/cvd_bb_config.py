from typing import Dict, Any, List
from .base_config import BaseStrategyConfig


class CVDBBPullbackConfig(BaseStrategyConfig):
    """
    Configuration class for CVD Bollinger Band Pullback Strategy.

    This class manages all parameters, validation rules, and optimization
    ranges for the CVD BB Pullback strategy.
    """

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values for CVD BB Pullback strategy."""
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
            'bbands_length': [15, 20, 25, 30, 35, 40, 45, 50],
            'bbands_stddev': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'cvd_length': [20, 30, 40, 50, 60, 70, 80, 100],
            'atr_length': [7, 10, 14, 20, 28, 35],
            'sl_coef': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'tpsl_ratio': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        }

    @property
    def param_constraints(self) -> Dict[str, Any]:
        """Parameter constraints and validation rules."""
        return {
            'bbands_length': {
                'min': 5,
                'max': 200,
                'type': int
            },
            'bbands_stddev': {
                'min': 0.1,
                'max': 5.0,
                'type': (int, float)
            },
            'cvd_length': {
                'min': 1,
                'max': 500,
                'type': int
            },
            'atr_length': {
                'min': 1,
                'max': 100,
                'type': int
            },
            'sl_coef': {
                'min': 0.1,
                'max': 10.0,
                'type': (int, float)
            },
            'tpsl_ratio': {
                'min': 0.1,
                'max': 10.0,
                'type': (int, float)
            }
        }

    def get_conservative_params(self) -> Dict[str, Any]:
        """Get conservative parameter set for low-risk trading."""
        return {
            'bbands_length': 40,
            'bbands_stddev': 2.5,
            'cvd_length': 60,
            'atr_length': 20,
            'sl_coef': 3.0,
            'tpsl_ratio': 3.0
        }

    def get_aggressive_params(self) -> Dict[str, Any]:
        """Get aggressive parameter set for higher-risk trading."""
        return {
            'bbands_length': 20,
            'bbands_stddev': 1.5,
            'cvd_length': 30,
            'atr_length': 10,
            'sl_coef': 1.5,
            'tpsl_ratio': 1.5
        }

    def get_balanced_params(self) -> Dict[str, Any]:
        """Get balanced parameter set for moderate risk."""
        return self.default_params

    def describe_strategy(self) -> str:
        """Get strategy description."""
        return """
        CVD Bollinger Band Pullback Strategy
        
        This strategy identifies pullback opportunities using Cumulative Volume Delta (CVD)
        and Bollinger Bands. The strategy enters positions when CVD crosses back into
        the Bollinger Band range after being outside it, indicating potential reversal.
        
        Entry Logic:
        - Long: CVD crosses above lower Bollinger Band after being below
        - Short: CVD crosses below upper Bollinger Band after being above
        
        Exit Logic:
        - Stop Loss: ATR-based distance from entry price
        - Take Profit: Multiple of stop loss distance (risk-reward ratio)
        
        Parameters:
        - bbands_length: Period for Bollinger Bands calculation
        - bbands_stddev: Standard deviation multiplier for bands
        - cvd_length: Rolling period for Cumulative Volume Delta
        - atr_length: Period for Average True Range calculation
        - sl_coef: Stop loss coefficient (ATR multiplier)
        - tpsl_ratio: Take profit to stop loss ratio
        """
