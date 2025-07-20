#!/usr/bin/env python3
"""
Data Configuration Script

Command-line interface for configuring and preparing market data
for quantitative strategy development.

Usage:
    python scripts/prepare_data.py --symbol BTC-USDT-SWAP --start 2023-01-01 --end 2024-06-30
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.data.pipeline.data_preparation import DataPreparationPipeline
    from src.data.config.data_config import DataConfig, DataSplitConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def main(
        start_date: datetime,
        end_date: datetime,
        symbol: str = "BTC-USDT-SWAP",
        exchange: str = "OKX",
        timeframe: str = "1h",
        train_pct: float = 0.6,
        validation_pct: float = 0.2,
        test_pct: float = 0.2,
        config_name: str = None,
        description: str = None,
        dry_run: bool = False
):
    """Main execution function."""
    try:
        start_date = datetime.strptime(
            start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(
            end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    if start_date >= end_date:
        print("Start date must be before end date")
        sys.exit(1)

    config_name = config_name or f"{symbol.lower().replace('-', '_')}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"

    description = description or f"{symbol} {timeframe} data from {start_date} to {end_date}"

    split_config = DataSplitConfig(
        train_pct=train_pct,
        validation_pct=validation_pct,
        test_pct=test_pct,
        purge_days=1
    )

    data_config = DataConfig(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        split_config=split_config,
        config_name=config_name,
        description=description
    )

    print("DATA CONFIGURATION")
    print("=" * 50)
    print(str(data_config))

    split_dates = data_config.get_split_dates()
    print(f"\nSplit Dates:")
    for split_name, (start, end) in split_dates.items():
        duration = end - start
        print(f"   {split_name.upper():12} {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({duration.days} days)")

    print(
        f"\nExpected data points: {data_config.get_expected_data_points():,}")

    if dry_run:
        print("\nDRY RUN - No data will be prepared")
        return

    print("\nREADY TO PREPARE DATA")
    print("This will download and process the full dataset.")
    response = input("Continue? (y/N): ").strip().lower()

    if response != 'y':
        print("Operation cancelled.")
        return

    pipeline = DataPreparationPipeline(data_config)

    try:
        print("\nStarting data preparation...")
        prepared_datasets = pipeline.prepare_data(save_to_disk=False)

        print("\nData preparation completed successfully!")

        for split_name, dataset in prepared_datasets.items():
            if len(dataset) > 0:
                print(f"\n{split_name.upper()} SET:")
                print(f"   Records: {len(dataset):,}")
                print(
                    f"   Period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")

        print(f"\nReady to proceed to Phase 2: Strategy Development!")
        print(
            f"\nUse this configuration name in your notebooks: '{config_name}'")

    except Exception as e:
        print(f"\nError during preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main("2022-01-01", "2022-12-31", "BTC-USDT-SWAP", "OKX", "1h",
         train_pct=0.6, validation_pct=0.2, test_pct=0.2,
         config_name="btc_usdt_swap_1h_2022", description="BTC-USDT-SWAP 1h data for strategy development")
