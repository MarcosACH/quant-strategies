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

# Lazy imports to avoid circular dependencies


def _get_cv_integration():
    """Lazy import of cv_integration module to avoid circular dependencies."""
    from .cv_integration import CVIntegratedOptimizer, CVOptimizationResult
    return CVIntegratedOptimizer, CVOptimizationResult


def __getattr__(name):
    """Implement lazy imports for circular dependency-prone modules."""
    if name == 'CVIntegratedOptimizer':
        CVIntegratedOptimizer, _ = _get_cv_integration()
        return CVIntegratedOptimizer
    elif name == 'CVOptimizationResult':
        _, CVOptimizationResult = _get_cv_integration()
        return CVOptimizationResult
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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

    # Integration (lazy loaded)
    'CVIntegratedOptimizer',
    'CVOptimizationResult'
]
