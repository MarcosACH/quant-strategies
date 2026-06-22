from utils.okx_candlestick_data_fetcher import data_fetcher
import asyncio
from time import time
import pandas as pd


if __name__ == "__main__":
    start = time()
    asyncio.run(data_fetcher(10000))

    data = pd.read_parquet("data/raw/okx_candles.parquet")
    print(data)
    print(data.info())
    print(data.describe())

    data["diff"] = data["Datetime"] - data["Datetime"].shift(1)
    print(data["diff"].unique())

    end = time()
    print(f"Data fetched in {end - start:.2f} seconds.")
