"""
Example: How to use the QuestDB Market Data Query Service

This script demonstrates how to fetch and analyze market data from QuestDB
for strategy development.
"""

from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery, verify_data_quality, get_btc_data
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    """Example usage of the QuestDB Market Data Query Service."""

    print("🚀 QuestDB Market Data Query Example\n")

    # Initialize query service
    query_service = QuestDBMarketDataQuery()

    # 1. Get available symbols
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

    # 2. Get data range for BTC
    print("2. Getting data range for BTC-USDT-SWAP...")
    try:
        btc_range = query_service.get_data_range("BTC-USDT-SWAP")
        print(f"   Data range: {btc_range}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # 3. Get recent BTC data (last 24 hours)
    print("3. Getting recent BTC data (last 24 hours)...")
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)

        # Get 5-minute sampled data
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

        # 4. Verify data continuity
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

    # 5. Get raw data (no sampling)
    print("5. Getting raw data (no sampling)...")
    try:
        # Get last 1000 raw records
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

    # 6. Demonstrate convenience function
    print("6. Using convenience function...")
    try:
        # Use the convenience function
        btc_data = get_btc_data(
            start_date=end_date - timedelta(hours=6),
            end_date=end_date,
            timeframe="15m"
        )

        print(f"   BTC 15m data: {len(btc_data)} records")
        print(
            f"   Price range: ${btc_data['low'].min():.2f} - ${btc_data['high'].max():.2f}")
        print()

    except Exception as e:
        print(f"   Error: {e}\n")

    # 7. Data quality verification
    print("7. Data quality verification...")
    try:
        quality_check = verify_data_quality(
            symbol="BTC-USDT-SWAP",
            start_date=end_date - timedelta(hours=12),
            end_date=end_date,
            timeframe="1m"
        )

        print(f"   Quality check results:")
        print(f"   - Continuous: {quality_check['is_continuous']}")
        print(f"   - Records: {quality_check['total_records']}")
        print(f"   - Gaps: {quality_check['gaps_found']}")
        print(f"   - Expected interval: {quality_check['expected_interval']}")
        print()

    except Exception as e:
        print(f"   Error: {e}\n")


if __name__ == "__main__":
    main()
