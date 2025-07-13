#!/usr/bin/env python3
"""
OKX Data Collection Script

This script demonstrates how to collect data from OKX exchange
using the integrated data collection framework.
"""

import asyncio
import argparse

from src.data.collectors import OKXDataCollector
from src.data.storage import FileManager, DatabaseManager


async def collect_okx_data(
    symbol: str = "BTC-USDT-SWAP",
    timeframe: str = "1m",
    count: int = 100,
    save_to_db: bool = False,
    save_to_file: bool = True
):
    """
    Collect data from OKX exchange and save to storage.

    Args:
        symbol: Trading pair symbol
        timeframe: Data timeframe  
        count: Number of 100-candle batches to fetch
        save_to_db: Whether to save to database
        save_to_file: Whether to save to file
    """
    print(f"🚀 Starting OKX data collection for {symbol} ({timeframe})")

    # Initialize collector
    collector = OKXDataCollector()

    try:
        # Fetch data
        print(f"📥 Fetching {count * 100} candles...")
        df = await collector.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            count=count
        )

        print(f"✅ Successfully fetched {len(df)} candles")
        print(
            f"📊 Data range: {df['datetime'].min()} to {df['datetime'].max()}")

        # Save to file if requested
        if save_to_file:
            file_manager = FileManager()

            # Convert to pandas for compatibility
            pandas_df = collector.to_pandas(df)

            filepath = file_manager.save_market_data(
                data=pandas_df,
                symbol=symbol,
                timeframe=timeframe,
                source="okx",
                format="parquet",
                compress=True
            )

            print(f"💾 Data saved to file: {filepath}")

        # Save to database if requested
        if save_to_db:
            try:
                db_manager = DatabaseManager()

                # Convert to pandas for database storage
                pandas_df = collector.to_pandas(df)

                db_manager.store_market_data(
                    data=pandas_df,
                    symbol=symbol,
                    timeframe=timeframe,
                    source="okx"
                )

                print(f"🗄️ Data saved to database")

            except Exception as e:
                print(f"⚠️ Database save failed: {e}")

        # Display sample data
        print(f"\n📈 Sample data (first 5 rows):")
        print(df.head())

        return df

    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Collect data from OKX exchange")

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USDT-SWAP",
        help="Trading pair symbol (default: BTC-USDT-SWAP)"
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
        help="Data timeframe (default: 1m)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of 100-candle batches to fetch (default: 100)"
    )

    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save data to database"
    )

    parser.add_argument(
        "--no-save-file",
        action="store_true",
        help="Don't save data to file"
    )

    args = parser.parse_args()

    # Run data collection
    asyncio.run(collect_okx_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        count=args.count,
        save_to_db=args.save_to_db,
        save_to_file=not args.no_save_file
    ))


if __name__ == "__main__":
    main()
