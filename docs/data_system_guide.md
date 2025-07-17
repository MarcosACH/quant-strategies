# QuestDB Market Data System

This system provides a complete solution for ingesting and querying market data from OKX exchange into QuestDB for quantitative strategy development.

## Components

### 1. Data Ingestion (`scripts/data_ingestion/questdb_data_ingestion.py`)

Fetches market data from OKX exchange and stores it in QuestDB using the candlestream module.

**Features:**

- Fetches historical OHLCV data from OKX
- Stores data in QuestDB with proper timestamps
- Command-line interface for easy automation
- Progress tracking and error handling

**Usage:**

```bash
# Basic usage - fetch BTC data for default date range
python scripts/data_ingestion/questdb_data_ingestion.py

# Custom symbol and date range
python scripts/data_ingestion/questdb_data_ingestion.py --symbol ETH-USDT-SWAP --from-date 2023-01-01 --to-date 2023-12-31

# Custom QuestDB connection
python scripts/data_ingestion/questdb_data_ingestion.py --questdb-host localhost --questdb-port 9000
```

**Arguments:**

- `--symbol`: Trading pair symbol (default: BTC-USDT-SWAP)
- `--from-date`: Start date in YYYY-MM-DD format
- `--to-date`: End date in YYYY-MM-DD format
- `--batch-size`: Number of batches to process
- `--questdb-host`: QuestDB host address
- `--questdb-port`: QuestDB port
- `--no-progress`: Disable progress output

### 2. Data Query (`src/data/query/questdb_market_data_query.py`)

Provides a Polars-based interface to query market data from QuestDB.

**Features:**

- Pure Polars implementation (no pandas dependency)
- Support for raw and sampled data queries
- Data continuity verification
- Automatic gap detection in time series
- Convenience functions for common queries

**Key Methods:**

- `get_market_data()`: Fetch OHLCV data with optional sampling
- `verify_data_continuity()`: Check for gaps in time series data
- `get_available_symbols()`: List available trading pairs
- `get_data_range()`: Get data availability information

**Usage Examples:**

```python
from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
from datetime import datetime, timezone, timedelta

# Initialize query service
query_service = QuestDBMarketDataQuery()

# Get raw data
df = query_service.get_market_data(
    symbol="BTC-USDT-SWAP",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
    timeframe=None  # Raw data
)

# Get 5-minute sampled data
df_5m = query_service.get_market_data(
    symbol="BTC-USDT-SWAP",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
    timeframe="5m"
)

# Verify data continuity
verification = query_service.verify_data_continuity(df_5m, "5m")
print(f"Data is continuous: {verification['is_continuous']}")
print(f"Gaps found: {verification['gaps_found']}")
```

### 3. Example Usage (`examples/data_query_example.py`)

Complete example demonstrating all features of the query system.

**Run the example:**

```bash
python examples/data_query_example.py
```

## Supported Timeframes

The system supports the following timeframes for data sampling:

- `1m` - 1 minute
- `5m` - 5 minutes  
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day
- `None` - Raw data (no sampling)

## Data Quality Verification

The system includes built-in data quality checks:

1. **Continuity Verification**: Checks if timestamps are evenly spaced according to the expected timeframe
2. **Gap Detection**: Identifies missing data points in time series
3. **Gap Analysis**: Provides detailed information about detected gaps including:
   - Timestamp of gap
   - Expected vs actual interval
   - Gap size as multiple of expected interval

## QuestDB Connection

The system connects to QuestDB using:

- **HTTP Ingestion**: Port 9000 (for data ingestion)
- **PostgreSQL Wire Protocol**: Port 8812 (for data queries)

**Default Connection:**

- Host: `ec2-44-202-48-168.compute-1.amazonaws.com`
- Ingestion Port: 9000
- Query Port: 8812
- Database: `qdb`
- Username: `admin`
- Password: `quest`

## Table Schema

Data is stored in QuestDB with the following schema:

```sql
CREATE TABLE ohlcv (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    exchange SYMBOL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE
) timestamp(timestamp) PARTITION BY DAY;
```

## Dependencies

Key dependencies for the market data system:

- `polars>=0.20.0` - Primary data processing library
- `questdb>=1.0.0` - QuestDB client library
- `httpx>=0.25.0` - HTTP client for API requests
- `connectorx>=0.3.0` - High-performance database connector

## Best Practices

1. **Data Ingestion:**
   - Run ingestion during off-peak hours
   - Use appropriate batch sizes to avoid rate limits
   - Monitor ingestion progress and handle errors gracefully

2. **Data Queries:**
   - Use sampled data when possible to reduce query time
   - Leverage QuestDB's time-series optimizations
   - Verify data continuity before running strategies

3. **Data Quality:**
   - Regularly check for data gaps
   - Validate data after ingestion
   - Use the continuity verification before backtesting

## Troubleshooting

**Common Issues:**

1. **Connection Errors**: Check QuestDB server status and network connectivity
2. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
3. **Data Gaps**: Use the verification tools to identify and analyze gaps
4. **Performance**: Use appropriate timeframes and date ranges for queries

**Debugging:**

- Enable progress output during ingestion
- Check QuestDB logs for ingestion errors
- Use the example script to test connectivity
