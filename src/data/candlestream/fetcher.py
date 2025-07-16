import asyncio
import collections
import re
import time
from typing import Any, AsyncIterator, Deque, Optional

import httpx


class Fetcher:
    def __init__(self, rate_limit: str, base_url: str):
        """Create a new Fetcher.

        Parameters
        ----------
        rate_limit : str
            A string in the form "<max>/<window>" e.g. "20/2s" meaning at most 20
            requests every 2 seconds. Supported time suffixes: ``ms``, ``s``, ``m``, ``h``.
        base_url : str
            The base URL that will be prefixed to every request.
        """

        # Parse the rate-limit specification.
        self._limit, self._window_seconds = self._parse_rate_limit(rate_limit)

        # Time stamps of the most recent requests (monotonic seconds).
        self._timestamps: Deque[float] = collections.deque()

        # Async lock protecting the timestamp deque so we can safely access it
        # from multiple coroutines.
        self._lock = asyncio.Lock()

        # Semaphore to bound the number of *concurrent* in-flight requests.  We
        # cap it at the same value as the throughput limit so we never exceed
        # the exchange's burst limits (helps avoid HTTP 429).
        self._semaphore = asyncio.Semaphore(
            int(self._limit) if self._limit != float("inf") else 1_000
        )

        self.client = httpx.AsyncClient(
            http2=True,
            base_url=base_url,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    # ---------------------------------------------------------------------
    # Rate-limit helpers
    # ---------------------------------------------------------------------

    TIME_MULTIPLIERS = {
        "ms": 1 / 1000,
        "s": 1,
        "m": 60,
        "h": 3600,
    }

    @classmethod
    def _parse_rate_limit(cls, spec: Optional[str]) -> tuple[float, float]:
        """Return `(limit, window_seconds)` parsed from *spec*.

        If *spec* is falsy, we return ``(float("inf"), 0.0)`` indicating no
        effective limit.
        """
        if not spec:
            # No rate limit specified
            return (float("inf"), 0.0)

        match = re.fullmatch(r"(\d+)\/(\d+)(ms|s|m|h)", spec.strip())
        if not match:
            raise ValueError(
                "rate_limit must be in the form '<count>/<window>' e.g. '20/2s'"
            )

        count = int(match.group(1))
        window_value = int(match.group(2))
        unit = match.group(3)
        seconds = window_value * cls.TIME_MULTIPLIERS[unit]
        return (count, seconds)

    async def _acquire(self) -> None:
        """Wait until sending a request would respect the configured rate-limit."""

        # Unbounded limit → no waiting necessary.
        if self._limit == float("inf"):
            return

        while True:
            async with self._lock:
                now = time.monotonic()

                # Remove timestamps that are no longer inside the window.
                while (
                    self._timestamps
                    and (now - self._timestamps[0]) >= self._window_seconds
                ):
                    self._timestamps.popleft()

                if len(self._timestamps) < self._limit:
                    # We have capacity for a new request.
                    self._timestamps.append(now)
                    return

                # Need to wait: compute time until the oldest timestamp exits the window.
                wait_seconds = self._window_seconds - (now - self._timestamps[0])

            # Sleep *outside* the lock so that other coroutines can proceed.
            await asyncio.sleep(wait_seconds)

    async def fetch_url(self, url: str) -> Any:
        try:
            # Ensure we do not exceed both the *throughput* (rate) and the
            # *concurrency* limits.
            async with self._semaphore:
                await self._acquire()

                response = await self.client.get(url)
                response.raise_for_status()

                return response.json()
        except httpx.RequestError as e:
            raise e

    async def close(self):
        await self.client.aclose()

    async def fetch_all(self, urls: list[str]) -> list[Any]:
        # We rely on the per-request _acquire() guard to rate-limit concurrent
        # tasks, so we can launch them all simultaneously.
        tasks = [self.fetch_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def fetch_stream(self, urls: list[str]) -> AsyncIterator[Any]:
        """Yield each response as soon as the corresponding request completes.

        This lets callers process responses incrementally (e.g. parse every *x*
        results) without having to wait for **all** requests to finish first.
        """

        # Create the tasks eagerly so they all start right away.
        tasks = [asyncio.create_task(self.fetch_url(url)) for url in urls]

        # ``asyncio.as_completed`` yields tasks in the order they finish.
        for coro in asyncio.as_completed(tasks):
            yield await coro
