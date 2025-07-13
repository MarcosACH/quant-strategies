import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import httpx
import polars as pl
import pandas as pd
from pathlib import Path

from config.settings import settings


class OKXDataCollector:
    """
    OKX exchange data collector for candlestick data.

    This class handles fetching OHLCV data from OKX exchange
    with proper rate limiting and error handling.
    """

    def __init__(self, base_url: str = "https://www.okx.com/api/v5"):
        """
        Initialize OKX data collector.

        Args:
            base_url: Base URL for OKX API
        """
        self.base_url = base_url
        self.rate_limit_delay = 0.5  # seconds between requests
        self.max_retries = 3

    async def get_candles(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> List:
        """
        Fetch candlestick data from OKX API.

        Args:
            client: HTTP client
            url: API endpoint URL
            params: Request parameters

        Returns:
            List of candlestick data
        """
        for attempt in range(self.max_retries):
            try:
                resp = await client.get(url, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") == "0":  # OKX success code
                    return data.get("data", [])
                else:
                    raise Exception(
                        f"OKX API error: {data.get('msg', 'Unknown error')}")

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        return []

    async def fetch_historical_data(
        self,
        symbol: str = "BTC-USDT-SWAP",
        timeframe: str = "1m",
        count: int = 100,
        save_path: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch historical candlestick data from OKX.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1H, 4H, 1D)
            count: Number of 100-candle batches to fetch
            save_path: Optional path to save the data

        Returns:
            Polars DataFrame with OHLCV data
        """
        start = int(datetime.now().timestamp() * 1000)
        start = start - (start % 60000)  # Round to minute

        # Calculate time interval based on timeframe
        timeframe_ms = self._get_timeframe_ms(timeframe)

        async with httpx.AsyncClient() as client:
            tasks = []

            for offset in range(0, timeframe_ms * count, timeframe_ms * 100):
                params = {
                    "instId": symbol,
                    "after": start - offset,
                    "bar": timeframe,
                    "limit": 100,
                }

                url = f"{self.base_url}/market/history-candles"
                tasks.append(asyncio.create_task(
                    self.get_candles(client, url, params)
                ))

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            # Gather all results
            candles_batches = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and flatten results
            candles = []
            for batch in candles_batches:
                if isinstance(batch, Exception):
                    print(f"Error fetching batch: {batch}")
                    continue
                candles.extend(batch)

        # Convert to DataFrame
        df = self._process_candles_data(candles)

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(save_path)
            print(f"Data saved to: {save_path}")

        return df

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        timeframe_map = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1H": 3600000,
            "4H": 14400000,
            "1D": 86400000,
            "1W": 604800000,
        }
        return timeframe_map.get(timeframe, 60000)

    def _process_candles_data(self, candles: List) -> pl.DataFrame:
        """
        Process raw candles data into structured DataFrame.

        Args:
            candles: Raw candles data from API

        Returns:
            Processed Polars DataFrame
        """
        if not candles:
            return pl.DataFrame()

        df = pl.DataFrame({
            "timestamp": [int(candle[0]) for candle in candles],
            "open": [float(candle[1]) for candle in candles],
            "high": [float(candle[2]) for candle in candles],
            "low": [float(candle[3]) for candle in candles],
            "close": [float(candle[4]) for candle in candles],
            "volume": [float(candle[5]) for candle in candles],
        })

        # Convert timestamp and optimize data types
        df = df.with_columns([
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime"),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float64),
        ])

        # Sort by datetime and remove duplicates
        df = df.sort("datetime", descending=False)
        df = df.unique(subset=["datetime"], keep="first")

        return df

    def to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        Convert Polars DataFrame to Pandas for vectorbt compatibility.

        Args:
            df: Polars DataFrame

        Returns:
            Pandas DataFrame with datetime index
        """
        pandas_df = df.to_pandas()
        pandas_df.set_index("datetime", inplace=True)
        return pandas_df

    async def fetch_latest_data(
        self,
        symbol: str = "BTC-USDT-SWAP",
        timeframe: str = "1m",
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Fetch latest candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of latest candles

        Returns:
            Polars DataFrame with latest data
        """
        async with httpx.AsyncClient() as client:
            params = {
                "instId": symbol,
                "bar": timeframe,
                "limit": limit,
            }

            url = f"{self.base_url}/market/candles"
            candles = await self.get_candles(client, url, params)

            return self._process_candles_data(candles)


# Convenience functions for backward compatibility
async def data_fetcher(count: int = 100, symbol: str = "BTC-USDT-SWAP", timeframe: str = "1m"):
    """
    Legacy function for backward compatibility.

    Args:
        count: Number of batches to fetch
        symbol: Trading pair symbol
        timeframe: Data timeframe
    """
    collector = OKXDataCollector()
    save_path = settings.DATA_ROOT_PATH / "raw" / \
        f"okx_{symbol.replace('-', '_').lower()}_{timeframe}_candles.parquet"

    df = await collector.fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        count=count,
        save_path=save_path
    )

    return df


# Example usage
if __name__ == "__main__":
    # Fetch BTC-USDT-SWAP 1-minute data
    asyncio.run(data_fetcher(100, "BTC-USDT-SWAP", "1m"))

    # Or use the class directly
    # collector = OKXDataCollector()
    # df = asyncio.run(collector.fetch_historical_data("BTC-USDT-SWAP", "1H", 50))
    # print(df)
