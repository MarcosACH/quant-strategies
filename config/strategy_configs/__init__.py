"""Strategy configuration management"""

from .base_config import BaseStrategyConfig
from .cvd_bb_config import CVDBBPullbackConfig

STRATEGY_CONFIGS = {
    'cvd_bb_pullback': CVDBBPullbackConfig,
}

__all__ = [
    'BaseStrategyConfig',
    'CVDBBPullbackConfig',
    'STRATEGY_CONFIGS'
]
