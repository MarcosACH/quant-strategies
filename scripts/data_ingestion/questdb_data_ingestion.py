"""QuestDB Market Data Ingestion Script using Candlestream."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from questdb.ingress import Sender, TimestampNanos

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.data.candlestream.exchanges.okx import OKX
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


async def ingest_market_data(
    symbol: str = "BTC-USDT-SWAP",
    from_datetime: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc),
    to_datetime: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
    batch_size: int = 10,
    questdb_host: str = "ec2-184-72-69-46.compute-1.amazonaws.com",
    questdb_port: int = 9000,
    show_progress: bool = True
):
    """
    Ingest market data from OKX into QuestDB.

    Args:
        symbol: Trading pair symbol
        from_datetime: Start datetime
        to_datetime: End datetime
        batch_size: Number of batches to process
        questdb_host: QuestDB host address
        questdb_port: QuestDB port
        show_progress: Show progress information
    """
    okx = OKX()

    print(f"Starting data ingestion for {symbol}")
    print(f"From: {from_datetime} to {to_datetime}")
    print(f"QuestDB: {questdb_host}:{questdb_port}")

    conf = f"http::addr={questdb_host}:{questdb_port};"
    total_records = 0

    async for candles in okx.get_candlesticks(
        symbol,
        from_datetime=from_datetime,
        to_datetime=to_datetime,
        batch_size=batch_size,
        show_progress=show_progress,
    ):
        df = pl.DataFrame([c.to_dict() for c in candles])

        df = df.with_columns(
            pl.col("timestamp").cast(
                pl.Datetime(time_unit="ns", time_zone=timezone.utc)
            )
        )

        df = df.with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit("OKX").alias("exchange"),
        )

        with Sender.from_conf(conf) as sender:
            sender.dataframe(
                df.to_pandas(), table_name="ohlcv", at="timestamp"
            )

        total_records += len(df)

    print(f"Data ingestion completed! Total records: {total_records}")


async def main(
    symbol: str = "BTC-USDT-SWAP",
    from_date: str = "2020-01-01",
    to_date: str = None,
    batch_size: int = 15,
    questdb_host: str = "ec2-184-72-69-46.compute-1.amazonaws.com",
    questdb_port: int = 9000,
    show_progress: bool = True
):
    """Main function with command line interface."""
    from_datetime = datetime.strptime(
        from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_datetime = datetime.strptime(
        to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if to_date else to_date

    try:
        await ingest_market_data(
            symbol=symbol,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
            batch_size=batch_size,
            questdb_host=questdb_host,
            questdb_port=questdb_port,
            show_progress=show_progress
        )
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main(from_date="2022-07-25",
                to_date="2023-01-01", batch_size=10))
