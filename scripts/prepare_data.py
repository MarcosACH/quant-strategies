#!/usr/bin/env python3
"""
Data Configuration Script

Command-line interface for configuring and preparing market data
for quantitative strategy development.

Usage:
    python scripts/prepare_data.py --symbol BTC-USDT-SWAP --start 2023-01-01 --end 2024-06-30
"""

from src.data.pipeline.data_preparation import DataPreparationPipeline
from src.data.config.data_config import DataConfig, DataSplitConfig
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure and prepare market data for strategy development"
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g., BTC-USDT-SWAP)"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--exchange",
        default="OKX",
        help="Exchange name (default: OKX)"
    )

    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Timeframe (default: 1h)"
    )

    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.6,
        help="Training set percentage (default: 0.6)"
    )

    parser.add_argument(
        "--validation-pct",
        type=float,
        default=0.2,
        help="Validation set percentage (default: 0.2)"
    )

    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.2,
        help="Test set percentage (default: 0.2)"
    )

    parser.add_argument(
        "--config-name",
        help="Configuration name (auto-generated if not provided)"
    )

    parser.add_argument(
        "--description",
        help="Configuration description"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without preparing data"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    try:
        start_date = datetime.strptime(
            args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(
            args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    if start_date >= end_date:
        print("Start date must be before end date")
        sys.exit(1)

    config_name = args.config_name or f"{args.symbol.lower().replace('-', '_')}_{args.timeframe}_{args.start.replace('-', '')}_{args.end.replace('-', '')}"

    description = args.description or f"{args.symbol} {args.timeframe} data from {args.start} to {args.end}"

    split_config = DataSplitConfig(
        train_pct=args.train_pct,
        validation_pct=args.validation_pct,
        test_pct=args.test_pct,
        purge_days=1
    )

    data_config = DataConfig(
        symbol=args.symbol,
        exchange=args.exchange,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        split_config=split_config,
        config_name=config_name,
        description=description
    )

    print("DATA CONFIGURATION")
    print("=" * 50)
    print(str(data_config))

    split_dates = data_config.get_split_dates()
    print(f"\\nSplit Dates:")
    for split_name, (start, end) in split_dates.items():
        duration = end - start
        print(f"   {split_name.upper():12} {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({duration.days} days)")

    print(
        f"\\nExpected data points: {data_config.get_expected_data_points():,}")

    if args.dry_run:
        print("\\nDRY RUN - No data will be prepared")
        return

    print("\\nREADY TO PREPARE DATA")
    print("This will download and process the full dataset.")
    response = input("Continue? (y/N): ").strip().lower()

    if response != 'y':
        print("Operation cancelled.")
        return

    pipeline = DataPreparationPipeline(data_config)

    try:
        print("\\nStarting data preparation...")
        prepared_datasets = pipeline.prepare_data(save_to_disk=True)

        print("\\nData preparation completed successfully!")
        print(f"\\nDatasets saved to: data/processed/{config_name}/")

        for split_name, dataset in prepared_datasets.items():
            if len(dataset) > 0:
                print(f"\\n{split_name.upper()} SET:")
                print(f"   Records: {len(dataset):,}")
                print(
                    f"   Period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")

        print(f"\\nReady to proceed to Phase 2: Strategy Development!")
        print(
            f"\\nUse this configuration name in your notebooks: '{config_name}'")

    except Exception as e:
        print(f"\\nError during preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
