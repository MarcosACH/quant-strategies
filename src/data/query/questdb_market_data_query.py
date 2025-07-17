"""
QuestDB Market Data Query using Polars

This module provides a simplified interface to query market data from QuestDB
using only Polars for data processing.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List
import polars as pl
from polars import col


class QuestDBMarketDataQuery:
    """
    QuestDB market data query service using Polars.

    This class provides methods to query market data from QuestDB
    for strategy development and backtesting using only Polars.
    """

    def __init__(self,
                 host: str = "ec2-44-202-48-168.compute-1.amazonaws.com",
                 port: int = 8812
                 ):
        """
        Initialize QuestDB query service.

        Args:
            host: QuestDB host address
            port: QuestDB PostgreSQL port (default: 8812)
        """
        self.connection_uri = f"redshift://admin:quest@{host}:{port}/qdb"

    def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str] = None,
        exchange: str = "OKX"
    ) -> pl.DataFrame:
        """
        Get market data from QuestDB.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USDT-SWAP")
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            timeframe: Timeframe for sampling (1m, 5m, 15m, 30m, 1h, 4h, 1d, None for raw)
            exchange: Exchange name (default: "OKX")

        Returns:
            Polars DataFrame with OHLCV data
        """
        if timeframe is None:
            query = f"""
                SELECT timestamp, symbol, exchange, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}' 
                AND exchange = '{exchange}'
                AND timestamp >= '{start_date.isoformat()}'
                AND timestamp <= '{end_date.isoformat()}'
                ORDER BY timestamp ASC
            """
        else:
            query = f"""
                SELECT 
                    timestamp,
                    symbol,
                    exchange,
                    first(open) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close) as close,
                    sum(volume) as volume
                FROM ohlcv
                WHERE symbol = '{symbol}' 
                AND exchange = '{exchange}'
                AND timestamp >= '{start_date.isoformat()}'
                AND timestamp <= '{end_date.isoformat()}'
                SAMPLE BY {timeframe} 
                ORDER BY timestamp ASC
            """

        try:
            df = pl.read_database_uri(query, self.connection_uri)

            df = df.with_columns(
                col("timestamp").cast(pl.Datetime(
                    time_unit="ns", time_zone="UTC"))
            )

            print(
                f"Retrieved {len(df)} records for {symbol} ({timeframe or 'raw'})")
            return df

        except Exception as e:
            print(f"Error querying market data: {e}")
            raise

    def verify_data_continuity(
        self,
        df: pl.DataFrame,
        expected_timeframe: str
    ) -> dict:
        """
        Verify data continuity by checking timestamp gaps.

        Args:
            df: DataFrame with timestamp column
            expected_timeframe: Expected timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)

        Returns:
            Dictionary with verification results
        """
        if len(df) < 2:
            return {
                "is_continuous": True,
                "total_records": len(df),
                "expected_interval": expected_timeframe,
                "gaps_found": 0,
                "gap_details": []
            }

        timeframe_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000
        }

        expected_interval_ms = timeframe_ms.get(expected_timeframe)
        if expected_interval_ms is None:
            raise ValueError(f"Unsupported timeframe: {expected_timeframe}")

        df_with_diff = df.with_columns([
            col("timestamp").diff().alias("time_diff")
        ])

        df_with_diff = df_with_diff.with_columns([
            col("time_diff").dt.total_milliseconds().alias("diff_ms")
        ])

        gaps = df_with_diff.filter(
            (col("diff_ms") != expected_interval_ms) &
            (col("diff_ms").is_not_null())
        )

        gap_details = []
        if len(gaps) > 0:
            gap_details = [
                {
                    "timestamp": row["timestamp"],
                    "expected_interval_ms": expected_interval_ms,
                    "actual_interval_ms": row["diff_ms"],
                    "gap_multiple": row["diff_ms"] / expected_interval_ms if expected_interval_ms > 0 else 0
                }
                for row in gaps.to_dicts()
            ]

        return {
            "is_continuous": len(gaps) == 0,
            "total_records": len(df),
            "expected_interval": expected_timeframe,
            "expected_interval_ms": expected_interval_ms,
            "gaps_found": len(gaps),
            "gap_details": gap_details,
            "data_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max()
            }
        }

    def get_available_symbols(self, exchange: str = "OKX") -> List[str]:
        """
        Get list of available symbols in the database.

        Args:
            exchange: Exchange name

        Returns:
            List of available symbols
        """
        query = f"""
            SELECT DISTINCT symbol
            FROM ohlcv
            WHERE exchange = '{exchange}'
            ORDER BY symbol
        """

        try:
            df = pl.read_database_uri(query, self.connection_uri)
            symbols = df["symbol"].to_list()

            print(f"Found {len(symbols)} symbols for {exchange}")
            return symbols

        except Exception as e:
            print(f"Error querying available symbols: {e}")
            raise

    def get_data_range(self, symbol: str, exchange: str = "OKX") -> dict:
        """
        Get data range information for a symbol.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            Dictionary with data range information
        """
        query = f"""
            SELECT 
                min(timestamp) as start_time,
                max(timestamp) as end_time,
                count(*) as total_records
            FROM ohlcv
            WHERE symbol = '{symbol}' AND exchange = '{exchange}'
        """

        try:
            df = pl.read_database_uri(query, self.connection_uri)

            if len(df) > 0:
                result = df.to_dicts()[0]
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "total_records": result["total_records"]
                }
            else:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "start_time": None,
                    "end_time": None,
                    "total_records": 0
                }

        except Exception as e:
            print(f"Error querying data range: {e}")
            raise


if __name__ == "__main__":
    query_service = QuestDBMarketDataQuery()

    symbols = query_service.get_available_symbols()
    print(f"Available symbols: {symbols}")

    if symbols:
        data_range = query_service.get_data_range(symbols[0])
        print(f"Data range for {symbols[0]}: {data_range}")

        end_date = datetime.now(timezone.utc)
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

        df = query_service.get_market_data(
            symbol=symbols[0],
            start_date=start_date,
            end_date=end_date,
            timeframe="5m"
        )

        print(f"Sample data:")
        print(df.head())

        verification = query_service.verify_data_continuity(df, "5m")
        print(f"Data continuity check: {verification}")
