"""Concrete strategy implementations"""

from .cvd_bb_pullback import CVDBBPullbackStrategy

# Registry for strategy discovery
AVAILABLE_STRATEGIES = {
    'cvd_bb_pullback': CVDBBPullbackStrategy,
}

__all__ = [
    'CVDBBPullbackStrategy',
    'AVAILABLE_STRATEGIES'
]
