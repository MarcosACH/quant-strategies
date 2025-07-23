"""
Enhanced CVD BB Pullback Strategy Backtest Script

This script runs comprehensive backtests of the CVD Bollinger Band Pullback strategy
using QuestDB data and multiple parameter optimization techniques:
1. Grid Search (exhaustive parameter search)
2. Random Search (random parameter sampling)
3. Bayesian Optimization (intelligent parameter search)

Usage:
    from scripts.backtesting.run_enhanced_cvd_bb_backtest import ParameterSelection, BacktestRunner
    
    # Create parameter selector
    param_selector = ParameterSelection()
    
    # Create backtest runner
    runner = BacktestRunner(
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h"
    )
    
    # Run optimization
    runner.run_backtest(param_selector, method="grid", optimization_metric="sharpe_ratio")
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.bt_engine.backtest_runner import BacktestRunner
    from src.optimization.parameters_selector import ParametersSelection
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def example_grid_search():
    """Example: Run grid search optimization."""
    param_ranges_small = {
        "bbands_length": np.arange(25, 150, 10),
        "bbands_stddev": np.arange(2.0, 6.0, 0.5),
        "cvd_length": [40],  # np.arange(35, 60, 5),
        "atr_length": [10],  # np.arange(5, 25, 5),
        "sl_coef": [2.0],  # np.arange(2.0, 3.5, 0.5),
        "tpsl_ratio": [2.5],  # np.arange(3.0, 5.5, 0.5)
    }

    runner = BacktestRunner(
        CVDBBPullbackStrategy(),
        QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    results = runner.run_backtest(
        param_selector=ParametersSelection(param_ranges_small),
        method="grid",
        optimization_metric="sharpe_ratio",
        save_results=True
    )

    return results


def example_random_search():
    """Example: Run random search optimization."""
    param_selector = ParametersSelection()

    runner = BacktestRunner(
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h"
    )

    results = runner.run_backtest(
        param_selector=param_selector,
        method="random",
        optimization_metric="sharpe_ratio",
        n_iter=100,
        save_results=True
    )

    return results


def example_bayesian_optimization():
    """Example: Run Bayesian optimization."""
    param_selector = ParametersSelection()

    runner = BacktestRunner(
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h"
    )

    results = runner.run_backtest(
        param_selector=param_selector,
        method="bayesian",
        optimization_metric="sharpe_ratio",
        n_iter=50,
        save_results=True
    )

    return results


def example_custom_optimization():
    """Example: Run optimization with custom parameters and metric."""
    param_selector = ParametersSelection()

    runner = BacktestRunner(
        symbol="ETH-USDT-SWAP",
        start_date="2023-01-01",
        end_date="2023-06-30",
        timeframe="4h",
        initial_cash=5000,
        fee_pct=0.04,
        risk_pct=2.0
    )

    results = runner.run_backtest(
        param_selector=param_selector,
        method="random",
        optimization_metric="total_return_pct",
        n_iter=150,
        save_results=True
    )

    return results


if __name__ == "__main__":
    """
    Direct execution examples. 
    Uncomment the desired example to run.
    """

    print("Enhanced CVD BB Pullback Strategy Backtesting")
    print("=" * 60)

    # Example 1: Grid Search
    print("Running Grid Search Example...")
    results = example_grid_search()

    # Example 2: Random Search (uncomment to run)
    # print("Running Random Search Example...")
    # results = example_random_search()

    # Example 3: Bayesian Optimization (uncomment to run)
    # print("Running Bayesian Optimization Example...")
    # results = example_bayesian_optimization()

    # Example 4: Custom optimization (uncomment to run)
    # print("Running Custom Optimization Example...")
    # results = example_custom_optimization()
