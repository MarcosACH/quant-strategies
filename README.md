## Quantitative Trading Strategy Framework (WIP)

Status: Work in progress. This framework is under active development and not finished yet. Interfaces, APIs, and behavior may change at any time.

### What this is

Tools to research, backtest, and analyze quantitative trading strategies (built around Python and vectorized workflows).

### Current highlights

- Modular config-first structure (config/, src/)
- Data prep and notebooks for exploration (notebooks/)
- Early strategy prototypes and backtesting utilities

### Run the example end-to-end

The bundled example backtests a **CVD + Bollinger Band pullback** strategy on BTC-USDT-SWAP
and sweeps its parameters. The steps below take you from a fresh clone to results — no prior
setup assumed. Everything runs locally against public OKX data; no exchange API keys required.

**Requirements:** Python 3.10+ (3.12 recommended) and Docker (for the local QuestDB).

#### 1. Clone and install

```bash
git clone https://github.com/MarcosACH/quant-strategies.git
cd quant-strategies

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Native dependency — TA-Lib.** The strategies compute indicators via TA-Lib, which needs its
> C library installed *before* `pip install` can build the Python wrapper:
> - **Windows:** `conda install -c conda-forge ta-lib`, or install a prebuilt wheel from
>   [TA-Lib/ta-lib-python releases](https://github.com/TA-Lib/ta-lib-python#windows).
> - **macOS:** `brew install ta-lib`
> - **Debian/Ubuntu:** `sudo apt-get install ta-lib` (or build from source).

#### 2. Configure environment

```bash
# Windows (PowerShell):
Copy-Item .env.example .env
# macOS / Linux:
cp .env.example .env
```

The defaults already point at the local QuestDB you start in the next step, so **no edits are
needed for a local run**. (To use a remote QuestDB instead, set `QUESTDB_HOST` / `QUESTDB_PG_PORT`
/ `QUESTDB_ILP_PORT` in `.env`.)

#### 3. Start QuestDB

```bash
docker compose -f deployment/docker/docker-compose.yml up -d
```

This launches QuestDB locally (web console at http://localhost:9000). Data ingestion writes to
port `9000`; backtest queries read from the PostgreSQL-wire port `8812`.

#### 4. Ingest market data

Pull historical 1-minute candles from OKX into QuestDB. From the project root, this ingests
2022 Q1 (~130k candles, a couple of minutes) — enough for the example below:

```bash
python -c "import asyncio; from scripts.data_ingestion.questdb_data_ingestion import main; asyncio.run(main(symbol='BTC-USDT-SWAP', from_date='2022-01-01', to_date='2022-04-01'))"
```

Candles land in the `ohlcv` table. (CVD is derived from OHLCV at backtest time, so no extra data
is needed.)

> **Rate limits.** OKX's public `history-candles` endpoint throttles aggressively. Ingesting in
> windows of a few months at a time is reliable; very large pulls (e.g. a full year in one call)
> can trip HTTP 429. Run the command again with later dates to extend coverage.

#### 5. Run the backtest

```bash
python scripts/backtesting/run_cvd_bb_backtest.py
```

It runs a grid search and prints the top strategies by Sharpe ratio; the best parameters are saved
to `results/best_params/`. Edit the `__main__` block of that script to switch between grid / random
/ Bayesian search.

> **Note.** Grid search runs single-threaded by default (`MAX_PARALLEL_JOBS=1`) to stay memory-safe
> during numba compilation. On a machine with spare RAM, raise it in `.env` to parallelize. Also
> ensure the script's date range falls within the window you ingested in step 4.

#### Or call it from your own code

```python
import numpy as np
from src.bt_engine.backtest_runner import BacktestRunner
from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery

# 1. Parameter grid to sweep
param_ranges = {
    "bbands_length": np.arange(25, 150, 10),
    "bbands_stddev": np.arange(2.0, 6.0, 0.5),
    "cvd_length":    [40],
    "atr_length":    [10],
    "sl_coef":       [2.0],
    "tpsl_ratio":    [2.5],
}

# 2. Wire a strategy + data source into the runner
runner = BacktestRunner(
    CVDBBPullbackStrategy(),
    QuestDBMarketDataQuery(),
    symbol="BTC-USDT-SWAP",
    start_date="2022-01-01",
    end_date="2022-04-01",   # match the window you ingested in step 4
    timeframe="1h",
    initial_cash=1000,
    fee_pct=0.05,
    risk_pct=1.0,
)

# 3. Optimize over the grid — method can be "grid", "random" or "bayesian"
results = runner.run_backtest(
    data_type="train",
    param_ranges=param_ranges,
    method="grid",
    optimization_metric="sharpe_ratio",
    auto_confirm=True,  # skip the interactive data-prep prompt
)

# results: a Polars DataFrame, one row per parameter combination
print(results.sort("sharpe_ratio", descending=True).head())
```

Each row carries `sharpe_ratio`, `total_return_pct`, `max_drawdown_pct`, `win_rate_pct`,
`total_trades` and the parameter values; the best combination is also saved to
`results/best_params/`.

### Roadmap (short)

- Stabilize public APIs and configs
- Expand tests and docs
- Harden data ingestion and backtesting engine

### Contributing

While the project is still evolving, feedback and issues are welcome. Major contributions may be deferred until the API stabilizes.

### License

Non-commercial, no-redistribution. See [LICENSE](LICENSE) for the full terms.

### Disclaimer

For research and educational use only. Trading involves risk; no warranty or guarantees are provided.
