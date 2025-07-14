"""
Quantitative Trading Strategy Development Framework

A comprehensive framework for developing, backtesting, and deploying
quantitative trading strategies using vectorbt.
"""

__version__ = "1.0.0"
__author__ = "ActiveQuants"

# Core imports for easy access
from .strategies.base.strategy_base import StrategyBase
from .bt_engine.vectorbt_engine import VectorBTEngine
from .analysis.portfolio_analyzer import PortfolioAnalyzer

__all__ = [
    'StrategyBase',
    'VectorBTEngine',
    'PortfolioAnalyzer'
]
