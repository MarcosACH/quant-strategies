from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

import numpy as np

from ..candle import Candle
from ..fetcher import Fetcher


class Exchange(Fetcher):
    def __init__(self, name: str, rate_limit: str, base_url: str):
        super().__init__(rate_limit, base_url)
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Exchange(name={self.name})"

    def _parse_candlesticks(self, responses: List[dict]) -> List[Candle]:
        raise NotImplementedError("parse_candlesticks is not implemented")

    def _get_candlesticks_urls(
        self,
        symbol: str,
        from_datetime: datetime,
        to_datetime: Optional[datetime] = None,
    ) -> List[str]:
        raise NotImplementedError("get_candlesticks_urls is not implemented")

    def _checksum_candlesticks(self, candles: List[Candle]) -> bool:

        n = len(candles)
        if n == 0:
            return True

        # Convert aware datetimes to epoch seconds (int64) in one vectorised pass.
        ts = np.fromiter(
            (c.timestamp.timestamp() for c in candles), dtype=np.int64, count=n
        )

        # ``np.unique`` both sorts and deduplicates – a single pass.
        uniq_ts = np.unique(ts)

        # Fail fast on duplicates.
        if uniq_ts.size != n:
            return False

        # Check that consecutive timestamps differ by exactly 60 seconds.
        return bool(np.all(np.diff(uniq_ts) == 60))

    async def get_candlesticks(
        self,
        symbol: str,
        from_datetime: datetime,
        to_datetime: Optional[datetime] = None,
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> AsyncGenerator[List[Candle], None]:
        """Fetch candlesticks for *symbol*.

        If *batch_size* is provided, responses are parsed every ``batch_size``
        finished requests instead of waiting for the full set.  This can reduce
        latency and memory footprint when the time range spans many API calls.
        """

        urls = self._get_candlesticks_urls(symbol, from_datetime, to_datetime)

        # ------------------------------------------------------------------
        # Progress-bar setup (candle-level)
        # ------------------------------------------------------------------

        expected_candles: Optional[int] = None

        if show_progress:
            # Compute expected number of 1-minute candles in the requested
            # range so the bar reflects true data progress rather than number
            # of HTTP calls.

            end_dt = to_datetime or datetime.now(timezone.utc)
            start_dt = from_datetime

            # Normalise to exact minute boundaries (UTC) to avoid off-by-one.
            start_dt = start_dt.replace(second=0, microsecond=0)
            end_dt = end_dt.replace(second=0, microsecond=0)

            # +1 because both endpoints are inclusive (one candle per minute).
            expected_candles = int((end_dt - start_dt).total_seconds() // 60) + 1

        # Lazily import tqdm so it is an *optional* dependency.
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(
                    total=expected_candles,
                    desc=f"{symbol} candles",
                    unit="candle",
                )
            except ModuleNotFoundError:
                print(
                    "tqdm not installed - progress bar disabled.  Install with 'pip install tqdm'."
                )
                show_progress = False  # disable for remainder

        if batch_size is None or batch_size <= 0:
            # Original behaviour—wait for every request to finish first.
            responses = await self.fetch_all(urls)

            _candles = self._parse_candlesticks(responses)

            if not self._checksum_candlesticks(_candles):
                raise ValueError("Candlesticks are not valid")

            yield _candles
        else:
            # Process the requests in consecutive *batches* of at most
            # ``batch_size`` URLs.  This keeps the number of concurrent tasks
            # bounded and therefore works more naturally with the internal
            # rate-limiter – reducing the likelihood of 429 responses from the
            # exchange.

            for start in range(0, len(urls), batch_size):
                sub_urls = urls[start : start + batch_size]

                # The inherited ``fetch_all`` still applies the per-request
                # _acquire() guard, so we remain within the configured
                # rate-limit even inside each batch.
                responses = await self.fetch_all(sub_urls)

                _candles = self._parse_candlesticks(responses)

                if not self._checksum_candlesticks(_candles):
                    raise ValueError("Candlesticks are not valid")

                yield _candles

                if show_progress and expected_candles:
                    pbar.update(len(_candles))

        if show_progress:
            pbar.close()
