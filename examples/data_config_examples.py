"""
Example Data Configuration

This file shows how to create data configurations programmatically.
"""

from datetime import datetime, timezone, timedelta
from src.data.config.data_config import DataConfig, DataSplitConfig
from src.data.pipeline.data_preparation import DataPreparationPipeline


def create_btc_config():
    """Create a configuration for BTC-USDT-SWAP data."""
    split_config = DataSplitConfig(
        train_pct=0.6,
        validation_pct=0.2,
        test_pct=0.2,
        purge_days=1
    )

    return DataConfig(
        symbol="BTC-USDT-SWAP",
        exchange="OKX",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        timeframe="1h",
        split_config=split_config,
        config_name="btc_usdt_swap_1h_2023_2024",
        description="BTC-USDT-SWAP 1h data for 2023-2024 strategy development"
    )


def create_eth_config():
    """Create a configuration for ETH-USDT-SWAP data."""
    split_config = DataSplitConfig(
        train_pct=0.7,
        validation_pct=0.15,
        test_pct=0.15,
        purge_days=2
    )

    return DataConfig(
        symbol="ETH-USDT-SWAP",
        exchange="OKX",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        timeframe="4h",
        split_config=split_config,
        config_name="eth_usdt_swap_4h_2023_2024",
        description="ETH-USDT-SWAP 4h data for 2023-2024 strategy development"
    )


def create_short_term_config():
    """Create a configuration for short-term trading data."""
    split_config = DataSplitConfig(
        train_pct=0.5,
        validation_pct=0.25,
        test_pct=0.25,
        purge_days=1
    )

    return DataConfig(
        symbol="BTC-USDT-SWAP",
        exchange="OKX",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        timeframe="15m",
        split_config=split_config,
        config_name="btc_usdt_swap_15m_2024_short",
        description="BTC-USDT-SWAP 15m data for short-term strategy development"
    )


def prepare_all_configs():
    """Prepare data for all example configurations."""
    configs = [
        create_btc_config(),
        create_eth_config(),
        create_short_term_config()
    ]

    for config in configs:
        print(f"\\nPreparing data for: {config.config_name}")
        print(f"   Symbol: {config.symbol}")
        print(
            f"   Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
        print(f"   Timeframe: {config.timeframe}")

        try:
            pipeline = DataPreparationPipeline(config)
            prepared_datasets = pipeline.prepare_data(save_to_disk=True)

            print(f"Successfully prepared {config.config_name}")

        except Exception as e:
            print(f"Error preparing {config.config_name}: {e}")


if __name__ == "__main__":
    print("Example Data Configurations")
    print("=" * 50)

    btc_config = create_btc_config()
    eth_config = create_eth_config()
    short_config = create_short_term_config()

    print("\\n1. BTC Configuration:")
    print(str(btc_config))

    print("\\n2. ETH Configuration:")
    print(str(eth_config))

    print("\\n3. Short-term Configuration:")
    print(str(short_config))

    print("\\nTo prepare data for these configurations, uncomment the line below:")
    print("# prepare_all_configs()")

    # Uncomment to actually prepare the data
    # prepare_all_configs()
