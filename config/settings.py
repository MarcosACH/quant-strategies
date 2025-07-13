import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()


class Settings:
    """Central configuration management for the trading framework"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables and config files"""

        # API Configuration
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
        self.IEX_API_KEY = os.getenv("IEX_API_KEY")
        self.YAHOO_FINANCE_API_KEY = os.getenv("YAHOO_FINANCE_API_KEY")

        # Database Configuration
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL", "sqlite:///data/trading_data.db")
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Trading Configuration
        self.PAPER_TRADING = os.getenv(
            "PAPER_TRADING", "true").lower() == "true"
        self.LIVE_TRADING = os.getenv(
            "LIVE_TRADING", "false").lower() == "true"
        self.BROKER_API_KEY = os.getenv("BROKER_API_KEY")

        # Backtesting Configuration
        self.INITIAL_CASH = float(os.getenv("INITIAL_CASH", "10000"))
        self.DEFAULT_FEE_PCT = float(os.getenv("DEFAULT_FEE_PCT", "0.1"))
        self.DEFAULT_FREQUENCY = os.getenv("DEFAULT_FREQUENCY", "1D")

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/trading.log")
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}")

        # Data Storage
        self.DATA_ROOT_PATH = Path(os.getenv("DATA_ROOT_PATH", "data/"))
        self.RESULTS_ROOT_PATH = Path(
            os.getenv("RESULTS_ROOT_PATH", "results/"))
        self.CACHE_ENABLED = os.getenv(
            "CACHE_ENABLED", "true").lower() == "true"
        self.CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))

        # Optimization Settings
        self.MAX_PARALLEL_JOBS = int(os.getenv("MAX_PARALLEL_JOBS", "-1"))
        self.OPTIMIZATION_TIMEOUT_HOURS = int(
            os.getenv("OPTIMIZATION_TIMEOUT_HOURS", "24"))
        self.VALIDATION_SPLIT_RATIO = float(
            os.getenv("VALIDATION_SPLIT_RATIO", "0.2"))
        self.TEST_SPLIT_RATIO = float(os.getenv("TEST_SPLIT_RATIO", "0.2"))

        # Risk Management
        self.MAX_POSITION_SIZE_PCT = float(
            os.getenv("MAX_POSITION_SIZE_PCT", "10"))
        self.MAX_PORTFOLIO_RISK_PCT = float(
            os.getenv("MAX_PORTFOLIO_RISK_PCT", "25"))
        self.DEFAULT_RISK_FREE_RATE = float(
            os.getenv("DEFAULT_RISK_FREE_RATE", "0.02"))

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


# Global settings instance
settings = Settings()
