import asyncio
from datetime import datetime
import httpx
import polars as pl


async def get_candles(client, url, params):
    resp = await client.get(url, params=params)
    return resp.json()["data"]


async def data_fetcher(count):
    start = int(datetime.now().timestamp() * 1000)
    start = start - (start % 60000)

    async with httpx.AsyncClient() as client:
        tasks = []
        for offset in range(0, 60000 * count, 60000 * 100):
            params = {
                "instId": "BTC-USDT-SWAP",
                "after": start - offset,
                "bar": "1m",
                "limit": 100,
            }
            url = f"https://www.okx.com/api/v5/market/history-candles"
            tasks.append(asyncio.ensure_future(
                get_candles(client, url, params)))

            await asyncio.sleep(0.5)

        candles = await asyncio.gather(*tasks)

        candles = [item for sublist in candles for item in sublist]

        df = pl.DataFrame(
            {
                "Datetime": [int(candle[0]) for candle in candles],
                "Open": [float(candle[1]) for candle in candles],
                "High": [float(candle[2]) for candle in candles],
                "Low": [float(candle[3]) for candle in candles],
                "Close": [float(candle[4]) for candle in candles],
                "Volume": [float(candle[5]) for candle in candles],
            }
        )

        df = df.with_columns(
            pl.from_epoch("Datetime", time_unit="ms"),
            pl.col("Open").cast(pl.Float32),
            pl.col("High").cast(pl.Float32),
            pl.col("Low").cast(pl.Float32),
            pl.col("Close").cast(pl.Float32),
            pl.col("Volume").cast(pl.Float64),
        )

        df = df.sort("Datetime", descending=False)
        df.write_parquet("data/raw/okx_candles.parquet")


# asyncio.run(data_fetcher(100))


# df = pl.read_parquet("candles.parquet")
# print(df)
# df = df.with_columns(
#     (pl.col("datetime") - pl.col("datetime").shift(1)).alias("diff"))

# print(df.select("diff").unique())
