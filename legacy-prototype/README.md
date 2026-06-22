# Legacy prototype

The original flat-layout prototype of this project (formerly the standalone
`quant-strategies-temp/` folder), preserved for reference. It predates the
current structured `src/` codebase.

Kept here because two of its strategies were **never ported** to the main
codebase:

- `strategies/ema_crosses_st.py` (+ `backtests/ema_crosses_bt.py`)
- `strategies/bb_crosses_ema_confirmation_st.py` (+ its backtest)

`cumulative_volume_delta_bb_pulback` was reimplemented as
`src/strategies/implementations/cvd_bb_pullback.py`.

Data files and backtest result CSVs were dropped (the data is identical to the
repo's `data/`; results are reproducible by re-running the backtests).
