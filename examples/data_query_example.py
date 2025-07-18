"""
Example: How to use the QuestDB Market Data Query Service

This script demonstrates how to fetch and analyze market data from QuestDB
for strategy development.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def main():
    """Example usage of the QuestDB Market Data Query Service."""

    print("QuestDB Market Data Query Example\n")

    query_service = QuestDBMarketDataQuery()

    print("1. Getting available symbols...")
    try:
        symbols = query_service.get_available_symbols()
        print(f"   Available symbols: {symbols}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        return

    if not symbols:
        print("   No symbols found. Please run data ingestion first.\n")
        return

    print("2. Getting data range for BTC-USDT-SWAP...")
    try:
        btc_range = query_service.get_data_range("BTC-USDT-SWAP")
        print(f"   Data range: {btc_range}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    print("3. Getting recent BTC data (last 24 hours)...")
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)

        df = query_service.get_market_data(
            symbol="BTC-USDT-SWAP",
            start_date=start_date,
            end_date=end_date,
            timeframe="5m"
        )

        print(f"   Retrieved {len(df)} records")
        print(f"   Data preview:")
        print(df.head())
        print()

        print("4. Verifying data continuity...")
        verification = query_service.verify_data_continuity(df, "5m")

        print(f"   Continuous data: {verification['is_continuous']}")
        print(f"   Total records: {verification['total_records']}")
        print(f"   Gaps found: {verification['gaps_found']}")

        if verification['gaps_found'] > 0:
            print("   Gap details:")
            for gap in verification['gap_details'][:5]:  # Show first 5 gaps
                print(f"     - {gap['timestamp']}: Expected {gap['expected_interval_ms']}ms, "
                      f"got {gap['actual_interval_ms']}ms (x{gap['gap_multiple']:.1f})")
        print()

    except Exception as e:
        print(f"   Error: {e}\n")

    print("5. Getting raw data (no sampling)...")
    try:
        raw_df = query_service.get_market_data(
            symbol="BTC-USDT-SWAP",
            start_date=end_date - timedelta(hours=2),
            end_date=end_date,
            timeframe=None  # Raw data
        )

        print(f"   Retrieved {len(raw_df)} raw records")
        print(
            f"   Time range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
        print()

    except Exception as e:
        print(f"   Error: {e}\n")


if __name__ == "__main__":
    main()
