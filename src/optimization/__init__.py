"""Optimization module with parameter selection and cross-validation capabilities."""

from .parameters_selector import ParametersSelection
from .cross_validator import (
    TimeSeriesCrossValidator,
    ValidationMethod,
    CVSplitInfo,
    CVResult,
    CVSummary,
    TimeSeriesSplitter,
    RollingWindowSplitter,
    ExpandingWindowSplitter,
    BlockedTimeSeriesSplitter,
    rolling_window_cv,
    expanding_window_cv,
    blocked_timeseries_cv
)
from .cv_integration import (
    CVIntegratedOptimizer,
    CVOptimizationResult
)

__all__ = [
    # Parameter selection
    'ParametersSelection',
    
    # Cross-validation core
    'TimeSeriesCrossValidator',
    'ValidationMethod',
    'CVSplitInfo',
    'CVResult', 
    'CVSummary',
    
    # Splitters
    'TimeSeriesSplitter',
    'RollingWindowSplitter',
    'ExpandingWindowSplitter',
    'BlockedTimeSeriesSplitter',
    
    # Convenience functions
    'rolling_window_cv',
    'expanding_window_cv', 
    'blocked_timeseries_cv',
    
    # Integration
    'CVIntegratedOptimizer',
    'CVOptimizationResult'
]