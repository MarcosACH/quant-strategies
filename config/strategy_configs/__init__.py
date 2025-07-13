"""Strategy configuration management"""

from .base_config import BaseStrategyConfig
from .cvd_bb_config import CVDBBPullbackConfig

# Strategy configuration registry
STRATEGY_CONFIGS = {
    'cvd_bb_pullback': CVDBBPullbackConfig,
}

__all__ = [
    'BaseStrategyConfig',
    'CVDBBPullbackConfig',
    'STRATEGY_CONFIGS'
]
