import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

load_dotenv()


class Settings:
    """Central configuration management for the trading framework"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables and config files"""

        # Backtesting Configuration
        self.INITIAL_CASH = 1000
        self.DEFAULT_FEE_PCT = 0.05
        self.DEFAULT_FREQUENCY = "1m"

        # Logging Configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = "logs/trading.log"
        self.LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"

        # Data Storage
        self.DATA_ROOT_PATH = self.project_root / "data/"
        self.RESULTS_ROOT_PATH = self.project_root / "results/"
        self.CACHE_ENABLED = True
        self.CACHE_EXPIRY_HOURS = 24

        # Optimization Settings
        self.MAX_PARALLEL_JOBS = -1
        self.OPTIMIZATION_TIMEOUT_HOURS = 24
        self.VALIDATION_SPLIT_RATIO = 0.2
        self.TEST_SPLIT_RATIO = 0.2

        # Risk Management
        self.MAX_POSITION_SIZE_PCT = 10
        self.MAX_PORTFOLIO_RISK_PCT = 25
        self.DEFAULT_RISK_FREE_RATE = 0.02

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_ROOT_PATH,
            self.RESULTS_ROOT_PATH,
            self.DATA_ROOT_PATH / "raw",
            self.DATA_ROOT_PATH / "processed",
            self.DATA_ROOT_PATH / "backtest_results",
            self.RESULTS_ROOT_PATH / "backtests",
            self.RESULTS_ROOT_PATH / "optimization",
            self.RESULTS_ROOT_PATH / "validation",
            Path("logs"),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.project_root / "config" / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def get_data_source_config(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific data source"""
        data_sources = self.load_yaml_config("data_sources.yaml")
        return data_sources.get("data_sources", {}).get(source_name)

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules configuration"""
        return self.load_yaml_config("validation_rules.yaml")


settings = Settings()
