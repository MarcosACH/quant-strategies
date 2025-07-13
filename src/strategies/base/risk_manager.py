from typing import Literal, Dict, Any
import numpy as np
from numba import njit


class PositionSizer:
    """
    Position sizing utilities for different sizing methods.
    """

    @staticmethod
    @njit
    def calculate_position_size(
        sizing_method: str,
        risk_pct: float,
        risk_nominal: float,
        position_size_value: float,
        cash_now: float,
        close: float,
        sl_price: float,
        fee_decimal: float
    ) -> float:
        """
        Calculate position size based on the specified method.

        Args:
            sizing_method: "Value-based", "Risk percent", or "Risk nominal"
            risk_pct: Risk percentage of portfolio
            risk_nominal: Fixed risk amount in currency
            position_size_value: Fixed position size value
            cash_now: Current available cash
            close: Current close price
            sl_price: Stop loss price
            fee_decimal: Fee as decimal (e.g., 0.001 for 0.1%)

        Returns:
            Position size value
        """
        if sizing_method == "Value-based":
            return cash_now if np.isnan(position_size_value) else position_size_value

        elif sizing_method == "Risk percent":
            if np.isnan(risk_pct):
                raise ValueError(
                    "risk_percent must be provided for Risk percent sizing method")

            risk_nominal_calc = cash_now * (risk_pct / 100.0)
            sl_pct = abs(close - sl_price) / close
            entry_fee_pct = fee_decimal
            exit_fee_pct = fee_decimal * (1 - sl_pct)
            total_fee_pct = entry_fee_pct + exit_fee_pct

            return risk_nominal_calc / (sl_pct + total_fee_pct)

        elif sizing_method == "Risk nominal":
            if np.isnan(risk_nominal):
                raise ValueError(
                    "risk_nominal must be provided for Risk nominal sizing method")

            sl_pct = abs(close - sl_price) / close
            entry_fee_pct = fee_decimal
            exit_fee_pct = fee_decimal * (1 - sl_pct)
            total_fee_pct = entry_fee_pct + exit_fee_pct

            return risk_nominal / (sl_pct + total_fee_pct)

        else:
            raise ValueError(f"Invalid sizing method: {sizing_method}")


@njit
def _calculate_position_size(sizing_method, risk_pct, risk_nominal, position_size_value,
                             cash_now, close, sl_price, fee_decimal):
    """Legacy function for backward compatibility with existing numba code."""
    if sizing_method == "Value-based":
        return cash_now if np.isnan(position_size_value) else position_size_value
    elif sizing_method == "Risk percent":
        if np.isnan(risk_pct):
            raise ValueError(
                "risk_percent must be provided for Risk percent sizing method")
        risk_nominal_calc = cash_now * (risk_pct / 100.0)
        sl_pct = abs(close - sl_price) / close
        entry_fee_pct = fee_decimal
        exit_fee_pct = fee_decimal * (1 - sl_pct)
        total_fee_pct = entry_fee_pct + exit_fee_pct
        return risk_nominal_calc / (sl_pct + total_fee_pct)
    elif sizing_method == "Risk nominal":
        if np.isnan(risk_nominal):
            raise ValueError(
                "risk_nominal must be provided for Risk nominal sizing method")
        sl_pct = abs(close - sl_price) / close
        entry_fee_pct = fee_decimal
        exit_fee_pct = fee_decimal * (1 - sl_pct)
        total_fee_pct = entry_fee_pct + exit_fee_pct
        return risk_nominal / (sl_pct + total_fee_pct)
    else:
        raise ValueError(f"Invalid sizing method: {sizing_method}")


class RiskManager:
    """
    Risk management utilities and constraints.
    """

    def __init__(self, max_position_size_pct: float = 10.0, max_portfolio_risk_pct: float = 25.0):
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct

    @staticmethod
    @njit
    def validate_position_size(
        position_size: float,
        cash_available: float,
        max_position_pct: float,
        min_size_value: float,
        max_size_value: float
    ) -> float:
        """
        Validate and constrain position size within risk limits.

        Args:
            position_size: Calculated position size
            cash_available: Available cash
            max_position_pct: Maximum position size as percentage of portfolio
            min_size_value: Minimum position size
            max_size_value: Maximum position size

        Returns:
            Validated position size
        """
        max_allowed = cash_available * (max_position_pct / 100.0)

        # Apply constraints
        position_size = min(position_size, max_allowed)
        position_size = min(position_size, max_size_value)
        position_size = max(position_size, min_size_value)

        return position_size

    @staticmethod
    @njit
    def calculate_portfolio_risk(
        positions: np.ndarray,
        volatilities: np.ndarray,
        correlations: np.ndarray
    ) -> float:
        """
        Calculate portfolio-level risk (simplified volatility-based measure).

        Args:
            positions: Array of position values
            volatilities: Array of asset volatilities
            correlations: Correlation matrix

        Returns:
            Portfolio risk measure
        """
        # Simplified portfolio risk calculation
        weighted_vols = positions * volatilities
        portfolio_variance = np.sum(weighted_vols ** 2)

        # Add correlation effects (simplified)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                portfolio_variance += 2 * positions[i] * positions[j] * \
                    volatilities[i] * volatilities[j] * correlations[i, j]

        return np.sqrt(portfolio_variance)


@njit
def apply_risk_constraints(
    position_size: float,
    close_price: float,
    cash_now: float,
    min_size_value: float,
    max_size_value: float,
    max_position_pct: float = 10.0
) -> float:
    """
    Apply risk management constraints to position size.

    Args:
        position_size: Initial position size
        close_price: Current price
        cash_now: Available cash
        min_size_value: Minimum position value
        max_size_value: Maximum position value
        max_position_pct: Maximum position as percentage of portfolio

    Returns:
        Risk-adjusted position size
    """
    # Convert to shares
    shares = position_size / close_price

    # Apply minimum size constraint
    min_shares = min_size_value / close_price
    shares = max(shares, min_shares)

    # Apply maximum size constraint
    max_shares = max_size_value / close_price
    shares = min(shares, max_shares)

    # Apply maximum percentage constraint
    max_portfolio_value = cash_now * (max_position_pct / 100.0)
    max_shares_portfolio = max_portfolio_value / close_price
    shares = min(shares, max_shares_portfolio)

    return shares * close_price
