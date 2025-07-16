from datetime import datetime, timezone


class Candle:
    def __init__(
        self,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ):
        self.timestamp = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)

    def __str__(self):
        return f"{self.timestamp} {self.open} {self.high} {self.low} {self.close} {self.volume}"

    def __repr__(self):
        return f"Candle(timestamp={self.timestamp}, open={self.open}, high={self.high}, low={self.low}, close={self.close}, volume={self.volume})"

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
