"""
Data Preparation Pipeline

This module provides a complete pipeline for data preparation in the
quantitative strategy development workflow.
"""

from typing import Dict, List
from datetime import datetime
import polars as pl
import json

from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
from src.data.config.data_config import DataConfig
from src.data.config.data_validator import DataValidator
from config.settings import settings


class DataPreparationPipeline:
    """
    Complete data preparation pipeline for strategy development.

    This class coordinates the entire data preparation process:
    1. Data collection from QuestDB
    2. Data quality validation
    3. Data cleaning and preprocessing
    4. Train/validation/test splitting
    5. Data export for downstream use
    """

    def __init__(self, config: DataConfig):
        """
        Initialize the data preparation pipeline.

        Args:
            config: DataConfig instance with all preparation parameters
        """
        self.config = config
        self.query_service = QuestDBMarketDataQuery()
        self.validator = DataValidator(config)
        self.results = {}

    def prepare_data(self, save_to_disk: bool = True) -> Dict[str, pl.DataFrame]:
        """
        Execute the complete data preparation pipeline.

        Args:
            save_to_disk: Whether to save prepared datasets to disk

        Returns:
            Dictionary containing prepared datasets
        """
        print(f"Starting data preparation for {self.config.symbol}")
        print(f"Configuration: {self.config.config_name}")
        print(
            f"Period: {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')}")
        print(f"Timeframe: {self.config.timeframe}")

        print("\nStep 1: Collecting raw data from QuestDB...")
        raw_data = self._collect_raw_data()

        print("\nStep 2: Validating data quality...")
        validation_results = self.validator.validate_data_quality(raw_data)
        self.results['validation'] = validation_results

        if not validation_results['is_valid']:
            print("Data validation failed!")
            for issue in validation_results['issues']:
                print(f"   • {issue}")
            raise ValueError("Data quality validation failed")
        else:
            print("Data quality validation passed")

        print("\nStep 3: Cleaning data...")
        cleaned_data = self.validator.clean_data(raw_data)
        print(f"   • Original records: {len(raw_data):,}")
        print(f"   • Cleaned records: {len(cleaned_data):,}")

        print("\nStep 4: Splitting data...")
        split_datasets = self._split_data(cleaned_data)

        for split_name, split_data in split_datasets.items():
            print(
                f"   • {split_name.capitalize()}: {len(split_data):,} records")

        if save_to_disk:
            print("\nStep 5: Saving prepared datasets...")
            self._save_datasets(split_datasets)

        print("\nStep 6: Generating summary report...")
        self._generate_summary_report(split_datasets)

        print("\nData preparation completed successfully!")
        return split_datasets

    def _collect_raw_data(self) -> pl.DataFrame:
        """Collect raw data from QuestDB."""
        try:
            data = self.query_service.get_market_data(
                symbol=self.config.symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                timeframe=self.config.timeframe,
                exchange=self.config.exchange
            )

            if len(data) == 0:
                raise ValueError("No data returned from QuestDB")

            return data

        except Exception as e:
            print(f"Error collecting data: {e}")
            raise

    def _split_data(self, data: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Split data into train/validation/test sets."""
        split_dates = self.config.get_split_dates()
        split_datasets = {}

        for split_name, (start_date, end_date) in split_dates.items():
            split_data = data.filter(
                (pl.col("timestamp") >= start_date) &
                (pl.col("timestamp") <= end_date)
            )

            if len(split_data) == 0:
                print(f"Warning: No data in {split_name} split")

            split_datasets[split_name] = split_data

        return split_datasets

    def _save_datasets(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Save datasets to disk."""
        output_dir = settings.DATA_ROOT_PATH / "processed" / self.config.config_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, dataset in datasets.items():
            if len(dataset) > 0:
                filename = f"{self.config.symbol}_{self.config.exchange}_{split_name}_{self.config.timeframe}.parquet"
                filepath = output_dir / filename
                dataset.write_parquet(filepath)
                print(f"   • Saved {split_name}: {filepath}")

        config_path = output_dir / "config.yaml"
        self.config.save_config(str(config_path))
        print(f"   • Saved config: {config_path}")

        metadata = {
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "config_name": self.config.config_name,
            "prepared_at": datetime.now().isoformat(),
            "datasets": {
                split_name: {
                    "filename": f"{self.config.symbol}_{self.config.exchange}_{split_name}_{self.config.timeframe}.parquet",
                    "records": len(dataset),
                    "start_date": dataset["timestamp"].min().strftime('%Y-%m-%d %H:%M:%S') if len(dataset) > 0 else None,
                    "end_date": dataset["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S') if len(dataset) > 0 else None
                }
                for split_name, dataset in datasets.items()
            },
            "validation_results": self.results.get('validation', {})
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"   • Saved metadata: {metadata_path}")

    def _generate_summary_report(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Generate and display summary report."""
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)

        print(f"Symbol: {self.config.symbol} ({self.config.exchange})")
        print(f"Timeframe: {self.config.timeframe}")
        print(
            f"Period: {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')}")
        print(f"Configuration: {self.config.config_name}")

        print("\nDATASET SUMMARY")
        print("-" * 40)

        total_records = 0
        for split_name, dataset in datasets.items():
            records = len(dataset)
            total_records += records

            if records > 0:
                start_date = dataset["timestamp"].min().strftime('%Y-%m-%d')
                end_date = dataset["timestamp"].max().strftime('%Y-%m-%d')
                percentage = (records / sum(len(d)
                              for d in datasets.values())) * 100

                print(
                    f"{split_name.capitalize()}: {records:,} records ({percentage:.1f}%)")
                print(f"   Period: {start_date} to {end_date}")
            else:
                print(f"{split_name.capitalize()}: No data")

        print(f"\nTotal Records: {total_records:,}")

        if 'validation' in self.results:
            val_results = self.results['validation']
            print("\nVALIDATION SUMMARY")
            print("-" * 40)
            print(
                f"Status: {'PASSED' if val_results['is_valid'] else 'FAILED'}")

            if val_results['issues']:
                print("Issues:")
                for issue in val_results['issues']:
                    print(f"   • {issue}")

            if val_results['warnings']:
                print("Warnings:")
                for warning in val_results['warnings']:
                    print(f"   • {warning}")

    def get_dataset_info(self, config_name: str) -> Dict:
        """
        Get information about a previously prepared dataset.

        Args:
            config_name: Name of the configuration

        Returns:
            Dictionary with dataset information
        """
        dataset_dir = settings.DATA_ROOT_PATH / "processed" / config_name
        metadata_path = dataset_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset '{config_name}' not found")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata

    def load_prepared_dataset(self, config_name: str, split: str = "train") -> pl.DataFrame:
        """
        Load a previously prepared dataset.

        Args:
            config_name: Name of the configuration
            split: Dataset split to load ("train", "validation", "test")

        Returns:
            Loaded dataset
        """
        dataset_dir = settings.DATA_ROOT_PATH / "processed" / config_name

        metadata = self.get_dataset_info(config_name)

        if split not in metadata["datasets"]:
            raise ValueError(
                f"Split '{split}' not found in dataset '{config_name}'")

        filename = metadata["datasets"][split]["filename"]
        filepath = dataset_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        return pl.read_parquet(filepath)

    @staticmethod
    def list_prepared_datasets() -> List[str]:
        """
        List all prepared datasets.

        Returns:
            List of dataset configuration names
        """
        processed_dir = settings.DATA_ROOT_PATH / "processed"

        if not processed_dir.exists():
            return []

        datasets = []
        for item in processed_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                datasets.append(item.name)

        return sorted(datasets)

    def print_validation_report(self, data: pl.DataFrame) -> None:
        """Print detailed validation report."""
        report = self.validator.generate_data_report(data)
        print(report)
