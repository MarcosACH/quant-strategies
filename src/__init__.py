"""
Quantitative Trading Strategy Development Framework

A comprehensive framework for developing, backtesting, and deploying
quantitative trading strategies using vectorbt.
"""

from .analysis.portfolio_analyzer import PortfolioAnalyzer
from .strategies.base.strategy_base import StrategyBase


def __getattr__(name):
    """Implement lazy imports for circular dependency-prone modules."""
    if name == 'VectorBTEngine':
        from .bt_engine.vectorbt_engine import VectorBTEngine
        return VectorBTEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'StrategyBase',
    'VectorBTEngine',
    'PortfolioAnalyzer'
]
