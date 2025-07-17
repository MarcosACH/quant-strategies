"""OKX exchange implementation."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

from ..candle import Candle
from .base import Exchange


class OKX(Exchange):
    BASE_URL = "https://www.okx.com"
    API_VERSION = "v5"

    def __init__(self):
        super().__init__("OKX", "20/2s",
                         f"{self.BASE_URL}/api/{self.API_VERSION}")

    def _parse_candlesticks(self, responses: List[dict]) -> List[Candle]:
        if not responses:
            return []

        # Concatenate all JSON arrays into a single 2-D NumPy array.
        data = np.concatenate(
            [np.asarray(r["data"], dtype=float, order="C") for r in responses], axis=0
        )

        if not data.size:
            return []

        # Ensure chronological order (ascending timestamp).
        data = data[np.argsort(data[:, 0])]

        # Column slicing: vectorised and cache-friendly.
        timestamps = (data[:, 0] * 0.001).astype(np.int64)  # ms → s as int64
        ohlcv = data[:, 1:6]  # shape (n, 5)

        # Build the domain objects.
        return [Candle(ts, *row) for ts, row in zip(timestamps, ohlcv)]

    def _get_candlesticks_urls(
        self,
        symbol: str,
        from_datetime: datetime,  # before
        to_datetime: Optional[datetime] = None,  # after
        limit: int = 100,
    ) -> List[str]:

        from_datetime = from_datetime.replace(second=0, microsecond=0) - timedelta(
            minutes=1
        )
        to_datetime = (
            to_datetime.replace(second=0, microsecond=0) + timedelta(minutes=1)
            if to_datetime
            else datetime.now(timezone.utc)
        )

        from_timestamp = int(from_datetime.timestamp() * 1000)
        to_timestamp = int(to_datetime.timestamp() * 1000)

        if from_timestamp > to_timestamp:
            raise ValueError("from_datetime is after to_datetime")

        # One request can return up to ``limit`` minutes → step size in ms.
        step_ms = limit * 60_000  # 60 000 ms = 1 minute

        # Vectorized generation of the `[before, after]` windows.
        befores = np.arange(from_timestamp, to_timestamp,
                            step_ms, dtype=np.int64)
        afters = np.minimum(
            befores + step_ms + 1, to_timestamp
        )  # +1 ms to make it inclusive

        return [
            "/market/history-candles"
            f"?instId={symbol}&before={before}&after={after}&limit={limit}"
            for before, after in zip(befores, afters)
        ]
