## Quantitative Trading Strategy Framework (WIP)

Status: Work in progress. This framework is under active development and not finished yet. Interfaces, APIs, and behavior may change at any time.

### What this is

Tools to research, backtest, and analyze quantitative trading strategies (built around Python and vectorized workflows).

### Current highlights

- Modular config-first structure (config/, src/)
- Data prep and notebooks for exploration (notebooks/)
- Early strategy prototypes and backtesting utilities

### Minimal setup

- Python 3.12 recommended
- Install dependencies: `pip install -r requirements.txt`
- Optional: editable install: `pip install -e .`
- Copy env template: `Copy-Item .env.example .env` (Windows PowerShell), then fill your local secrets

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
