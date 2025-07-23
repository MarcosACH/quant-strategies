# Enhanced Strategy Backtesting Workflow (Refactored)

This document outlines the enhanced backtesting workflow using the new `ParameterSelection` and `BacktestRunner` classes that integrate directly with QuestDB for data retrieval and provide multiple optimization techniques.

## Overview

The refactored workflow provides two main classes:

1. **ParameterSelection**: Generates parameter ranges for different optimization methods
2. **BacktestRunner**: Fetches data from QuestDB and runs backtests with optimization

### Optimization Methods Available

1. **Grid Search**: Exhaustive parameter search
2. **Random Search**: Random parameter sampling
3. **Bayesian Optimization**: Intelligent parameter search using Gaussian processes

## Architecture

### ParameterSelection Class

- Generates parameter ranges without performing backtesting
- Supports custom parameter ranges
- Methods:
  - `get_grid_search_params()`: Returns grid search parameter ranges
  - `get_random_search_params()`: Generates random parameter combinations
  - `get_bayesian_optimization_params()`: Sets up Bayesian optimization space

### BacktestRunner Class

- Fetches data directly from QuestDB
- Integrates data validation and cleaning
- Runs backtests with selected optimization method
- Saves results locally in `data/backtest_results/`
- Supports different optimization metrics

## Quick Start Usage

### Basic Example

```python
from scripts.backtesting.run_enhanced_cvd_bb_backtest import ParameterSelection, BacktestRunner

# Create parameter selector
param_selector = ParameterSelection()

# Create backtest runner
runner = BacktestRunner(
    symbol="BTC-USDT-SWAP",
    start_date="2022-01-01",
    end_date="2022-12-31",
    timeframe="1h",
    initial_cash=1000,
    fee_pct=0.05,
    risk_pct=1.0
)

# Run optimization
results = runner.run_backtest(
    param_selector=param_selector,
    method="grid",  # or "random", "bayesian"
    optimization_metric="sharpe_ratio",
    n_iter=100,  # for random/bayesian methods
    save_results=True
)
```

### Available Optimization Metrics

- `sharpe_ratio` (default)
- `total_return_pct`
- `win_rate_pct`
- `max_drawdown_pct` (minimize)
- `calmar_ratio`

## Detailed Usage Examples

### 1. Grid Search Optimization

```python
# Quick grid search with reduced parameter ranges
results = runner.run_backtest(
    param_selector=param_selector,
    method="grid",
    optimization_metric="sharpe_ratio"
)
```

### 2. Random Search Optimization

```python
# Random search with 100 iterations
results = runner.run_backtest(
    param_selector=param_selector,
    method="random",
    optimization_metric="total_return_pct",
    n_iter=100
)
```

### 3. Bayesian Optimization

```python
# Intelligent parameter search
results = runner.run_backtest(
    param_selector=param_selector,
    method="bayesian",
    optimization_metric="sharpe_ratio",
    n_iter=50
)
```

### 4. Custom Configuration

```python
# Different symbol, timeframe, and risk settings
runner = BacktestRunner(
    symbol="ETH-USDT-SWAP",
    start_date="2023-01-01",
    end_date="2023-06-30",
    timeframe="4h",
    initial_cash=5000,
    fee_pct=0.04,
    risk_pct=2.0
)
```

## Data Integration

### Direct QuestDB Integration

- No need for local data files
- Automatic data fetching from QuestDB
- Built-in data validation and cleaning
- Real-time data processing

### Data Requirements

- QuestDB must be running and accessible
- Market data should be available for the specified symbol and date range
- Data format should match the expected schema

## Output Files

Each optimization run produces:

1. **Results DataFrame**: `data/backtest_results/{config_name}_{method}_results.parquet`
   - Complete backtest results for all tested parameters
   - Performance metrics for each combination

2. **Best Parameters**: `data/backtest_results/{config_name}_{method}_best_params.json`
   - Optimal parameters found by the method
   - Associated performance metrics
   - Metadata about the optimization run

## Parameter Ranges

### Default Grid Search (Reduced for efficiency)

```python
{
    "bbands_length": [25, 50, 75, 100, 125, 150],
    "bbands_stddev": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    "cvd_length": [40, 50, 60],
    "atr_length": [10, 14, 20],
    "sl_coef": [2.0, 2.5, 3.0],
    "tpsl_ratio": [2.0, 2.5, 3.0, 3.5]
}
```

### Default Random/Bayesian Search (Broader ranges)

```python
{
    "bbands_length": list(range(25, 160, 5)),
    "bbands_stddev": [round(x, 1) for x in np.arange(2.0, 6.0, 0.1)],
    "cvd_length": list(range(35, 65, 5)),
    "atr_length": list(range(5, 25, 2)),
    "sl_coef": [round(x, 1) for x in np.arange(2.0, 3.5, 0.1)],
    "tpsl_ratio": [round(x, 1) for x in np.arange(2.0, 5.0, 0.1)]
}
```

## Running Examples

### Command Line Examples

```bash
# Run usage examples
python scripts/backtesting/usage_examples.py

# Direct execution with examples
python scripts/backtesting/run_enhanced_cvd_bb_backtest.py
```

### Jupyter Notebook Integration

```python
# In a Jupyter notebook
import sys
sys.path.append('path/to/quant-strategies')

from scripts.backtesting.run_enhanced_cvd_bb_backtest import ParameterSelection, BacktestRunner

# Your optimization code here...
```

## Best Practices

### 1. Start with Quick Testing

Use shorter date ranges and reduced parameters for initial testing:

```python
runner = BacktestRunner(
    symbol="BTC-USDT-SWAP",
    start_date="2022-06-01",  # Shorter period
    end_date="2022-08-31",
    timeframe="1h"
)
```

### 2. Progressive Optimization

1. **Random Search**: Initial exploration with 50-100 iterations
2. **Bayesian Optimization**: Refinement with 25-50 iterations
3. **Grid Search**: Final validation around promising regions

### 3. Metric Selection

- Use `sharpe_ratio` for risk-adjusted returns
- Use `total_return_pct` for absolute performance
- Use `max_drawdown_pct` to minimize risk

### 4. Memory Management

- Use appropriate `n_iter` values based on your system
- Grid search can be memory-intensive with large parameter ranges
- Random/Bayesian search are more memory-efficient

## Performance Monitoring

The framework provides comprehensive analysis:

- **Best parameter combinations** by chosen metric
- **Performance statistics** across all tested combinations
- **Execution time tracking**
- **Automatic result persistence**

## Error Handling

The framework includes robust error handling:

- QuestDB connection failures
- Data validation errors
- Backtest execution errors
- Invalid parameter combinations

## Next Steps

After optimization:

1. **Validation Testing**: Test best parameters on validation data
2. **Walk-Forward Analysis**: Rolling window testing
3. **Out-of-Sample Testing**: Final validation
4. **Live Trading Preparation**: Deploy optimized parameters

## Dependencies

Required packages:

```bash
pip install scikit-learn scipy scikit-optimize polars numpy vectorbt
```

Or install all requirements:

```bash
pip install -r requirements.txt
```
