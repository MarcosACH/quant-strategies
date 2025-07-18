"""
Data Configuration Module

This module provides configuration management for data selection, splitting,
and preparation in the quantitative strategy development workflow.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml
from config.settings import settings


@dataclass
class DataSplitConfig:
    """Configuration for data splitting strategy."""
    train_pct: float = 0.6
    validation_pct: Optional[float] = 0.2
    test_pct: Optional[float] = 0.2
    purge_days: int = 1  # Days to purge between splits to prevent leakage

    def __post_init__(self):
        """Validate split configuration."""
        total_pct = self.train_pct + \
            (self.validation_pct or 0) + (self.test_pct or 0)
        if abs(total_pct - 1.0) > 0.001:
            raise ValueError(
                f"Split percentages must sum to 1.0, got {total_pct}")

        if self.train_pct <= 0 or self.train_pct >= 1:
            raise ValueError("Train percentage must be between 0 and 1")

        if self.validation_pct is not None and (self.validation_pct <= 0 or self.validation_pct >= 1):
            raise ValueError("Validation percentage must be between 0 and 1")

        if self.test_pct is not None and (self.test_pct <= 0 or self.test_pct >= 1):
            raise ValueError("Test percentage must be between 0 and 1")


@dataclass
class DataConfig:
    """
    Configuration for data selection and preparation.

    This class manages all parameters needed for data collection,
    processing, and splitting according to the development workflow.
    """
    symbol: str
    exchange: str = "OKX"
    start_date: datetime = field(default_factory=lambda: datetime.now(
        timezone.utc) - timedelta(days=365))
    end_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    timeframe: str = "1h"

    split_config: DataSplitConfig = field(default_factory=DataSplitConfig)

    min_data_points: int = 1000
    max_gap_minutes: int = 60
    outlier_std_threshold: float = 5.0

    apply_data_cleaning: bool = True
    # "forward", "backward", "interpolate", "drop"
    fill_missing_method: str = "forward"

    config_name: str = field(default="default_config")
    description: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

        if not self.exchange:
            raise ValueError("Exchange cannot be empty")

        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

        if self.timeframe not in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        if self.min_data_points < 100:
            raise ValueError("Minimum data points should be at least 100")

        if self.fill_missing_method not in ["forward", "backward", "interpolate", "drop"]:
            raise ValueError(
                f"Unsupported fill method: {self.fill_missing_method}")

    def get_expected_data_points(self) -> int:
        """Calculate expected number of data points based on timeframe and date range."""
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }

        minutes_in_range = int(
            (self.end_date - self.start_date).total_seconds() / 60)
        return minutes_in_range // timeframe_minutes[self.timeframe]

    def get_split_dates(self) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate split dates based on configuration.

        Returns:
            Dictionary with 'train', 'validation', 'test' keys and date ranges
        """
        total_duration = self.end_date - self.start_date
        purge_duration = timedelta(days=self.split_config.purge_days)

        train_end = self.start_date + \
            (total_duration * self.split_config.train_pct)

        splits = {
            "train": (self.start_date, train_end)
        }

        if self.split_config.validation_pct:
            validation_start = train_end + purge_duration
            validation_end = validation_start + \
                (total_duration * self.split_config.validation_pct)
            splits["validation"] = (validation_start, validation_end)

            if self.split_config.test_pct:
                test_start = validation_end + purge_duration
                splits["test"] = (test_start, self.end_date)
        elif self.split_config.test_pct:
            test_start = train_end + purge_duration
            splits["test"] = (test_start, self.end_date)

        return splits

    def save_config(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration to YAML file.

        Args:
            filepath: Optional path to save file. If None, uses default path.

        Returns:
            Path where configuration was saved
        """
        if filepath is None:
            config_dir = settings.project_root / "config" / "data_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            filepath = config_dir / f"{self.config_name}.yaml"

        config_data = {
            "data_config": {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "timeframe": self.timeframe,
                "config_name": self.config_name,
                "description": self.description,
                "created_at": self.created_at.isoformat()
            },
            "split_config": {
                "train_pct": self.split_config.train_pct,
                "validation_pct": self.split_config.validation_pct,
                "test_pct": self.split_config.test_pct,
                "purge_days": self.split_config.purge_days
            },
            "quality_config": {
                "min_data_points": self.min_data_points,
                "max_gap_minutes": self.max_gap_minutes,
                "outlier_std_threshold": self.outlier_std_threshold,
                "apply_data_cleaning": self.apply_data_cleaning,
                "fill_missing_method": self.fill_missing_method
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        return str(filepath)

    @classmethod
    def load_config(cls, filepath: str) -> 'DataConfig':
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to configuration file

        Returns:
            DataConfig instance
        """
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        data_config = config_data["data_config"]
        split_config = config_data["split_config"]
        quality_config = config_data["quality_config"]

        start_date = datetime.fromisoformat(data_config["start_date"])
        end_date = datetime.fromisoformat(data_config["end_date"])
        created_at = datetime.fromisoformat(data_config["created_at"])

        split_cfg = DataSplitConfig(
            train_pct=split_config["train_pct"],
            validation_pct=split_config.get("validation_pct"),
            test_pct=split_config.get("test_pct"),
            purge_days=split_config["purge_days"]
        )

        return cls(
            symbol=data_config["symbol"],
            exchange=data_config["exchange"],
            start_date=start_date,
            end_date=end_date,
            timeframe=data_config["timeframe"],
            split_config=split_cfg,
            min_data_points=quality_config["min_data_points"],
            max_gap_minutes=quality_config["max_gap_minutes"],
            outlier_std_threshold=quality_config["outlier_std_threshold"],
            apply_data_cleaning=quality_config["apply_data_cleaning"],
            fill_missing_method=quality_config["fill_missing_method"],
            config_name=data_config["config_name"],
            description=data_config["description"],
            created_at=created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timeframe": self.timeframe,
            "split_config": {
                "train_pct": self.split_config.train_pct,
                "validation_pct": self.split_config.validation_pct,
                "test_pct": self.split_config.test_pct,
                "purge_days": self.split_config.purge_days
            },
            "min_data_points": self.min_data_points,
            "max_gap_minutes": self.max_gap_minutes,
            "outlier_std_threshold": self.outlier_std_threshold,
            "apply_data_cleaning": self.apply_data_cleaning,
            "fill_missing_method": self.fill_missing_method,
            "config_name": self.config_name,
            "description": self.description,
            "created_at": self.created_at.isoformat()
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        split_dates = self.get_split_dates()
        return f"""DataConfig: {self.config_name}
Symbol: {self.symbol} ({self.exchange})
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Timeframe: {self.timeframe}
Expected Data Points: {self.get_expected_data_points():,}
Splits: Train {self.split_config.train_pct:.0%} | Val {self.split_config.validation_pct or 0:.0%} | Test {self.split_config.test_pct or 0:.0%}
Train: {split_dates['train'][0].strftime('%Y-%m-%d')} to {split_dates['train'][1].strftime('%Y-%m-%d')}"""
