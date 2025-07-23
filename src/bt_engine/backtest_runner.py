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
import polars as pl
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List
from datetime import datetime, timezone
from skopt import gp_minimize
from skopt.utils import use_named_args


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.data.pipeline.data_preparation import DataPreparationPipeline
    from src.optimization.parameters_selector import ParametersSelection
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.bt_engine.vectorbt_engine import VectorBTEngine
    from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
    from src.data.config.data_config import DataConfig, DataSplitConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


class BacktestRunner:
    """
    Enhanced backtest runner with QuestDB integration and multiple optimization methods.
    """

    def __init__(self,
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 timeframe: str = "1h",
                 exchange: str = "OKX",
                 initial_cash: float = 1000,
                 fee_pct: float = 0.05,
                 risk_pct: float = 1.0,
                 config_name: str = None,
                 description: str = None
                 ):
        """
        Initialize the backtest runner.

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT-SWAP")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (e.g., "1h", "5m")
            exchange: Exchange name
            initial_cash: Initial cash amount
            fee_pct: Trading fee percentage
            risk_pct: Risk percentage for position sizing
        """
        self.symbol = symbol
        try:
            self.start_date = datetime.strptime(
                start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            self.end_date = datetime.strptime(
                end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            sys.exit(1)

        if start_date >= end_date:
            print("Start date must be before end date")
            sys.exit(1)
        self.timeframe = timeframe
        self.exchange = exchange
        self.initial_cash = initial_cash
        self.fee_pct = fee_pct
        self.risk_pct = risk_pct

        self.strategy = CVDBBPullbackStrategy()
        self.query_service = QuestDBMarketDataQuery()
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config_name = config_name or f"{symbol.lower().replace('-', '_')}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        self.description = description or f"{symbol} {timeframe} data from {start_date} to {end_date}"

    def load_training_data(self) -> pl.DataFrame:
        """
        Load training data from QuestDB.

        Returns:
            Training dataset
        """
        print(f"Loading data from QuestDB...")
        print(f"Symbol: {self.symbol}")
        print(
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Timeframe: {self.timeframe}")

        split_config = DataSplitConfig(
            train_pct=self.train_pct,
            validation_pct=self.validation_pct,
            test_pct=self.test_pct,
            purge_days=1
        )

        data_config = DataConfig(
            symbol=self.symbol,
            exchange=self.exchange,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=self.timeframe,
            split_config=split_config,
            config_name=self.config_name,
            description=self.description
        )

        print("DATA CONFIGURATION")
        print("=" * 50)
        print(str(data_config))

        split_dates = data_config.get_split_dates()
        print(f"\nSplit Dates:")
        for split_name, (start, end) in split_dates.items():
            duration = end - start
            print(
                f"   {split_name.upper():12} {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({duration.days} days)")

        print(
            f"\nExpected data points: {data_config.get_expected_data_points():,}")

        print("\nREADY TO PREPARE DATA")
        print("This will download and process the full dataset.")
        response = input("Continue? (y/N): ").strip().lower()

        if response != 'y':
            print("Operation cancelled.")
            return

        pipeline = DataPreparationPipeline(data_config)

        try:
            print("\nStarting data preparation...")
            prepared_datasets = pipeline.prepare_data(save_to_disk=False)

            print("\nData preparation completed successfully!")

            for split_name, dataset in prepared_datasets.items():
                if len(dataset) > 0:
                    print(f"\n{split_name.upper()} SET:")
                    print(f"   Records: {len(dataset):,}")
                    print(
                        f"   Period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")

            print(f"\nReady to proceed to Phase 2: Strategy Development!")
            print(
                f"\nUse this configuration name in your notebooks: '{self.config_name}'")

        except Exception as e:
            print(f"\nError during preparation: {e}")
            sys.exit(1)

    def run_backtest(self,
                     param_selector: ParametersSelection,
                     method: str = "grid",
                     optimization_metric: str = "sharpe_ratio",
                     n_iter: int = 100,
                     save_results: bool = True) -> pl.DataFrame:
        """
        Run backtest with specified optimization method.

        Args:
            param_selector: ParameterSelection instance
            method: Optimization method ("grid", "random", "bayesian")
            optimization_metric: Metric to optimize (e.g., "sharpe_ratio", "total_return_pct")
            n_iter: Number of iterations for random/bayesian search
            save_results: Whether to save results to disk

        Returns:
            Backtest results DataFrame
        """
        start_time = time.time()

        data = self.load_training_data()

        engine = VectorBTEngine(
            initial_cash=self.initial_cash,
            fee_pct=self.fee_pct,
            frequency=self.timeframe
        )

        print(f"\n{'='*60}")
        print(f"RUNNING {method.upper()} OPTIMIZATION")
        print(f"Optimization metric: {optimization_metric}")
        print(f"{'='*60}")

        if method == "grid":
            param_ranges = param_selector.get_grid_search_params(reduced=True)
            results = self._run_grid_search(engine, data, param_ranges)

        elif method == "random":
            param_ranges = param_selector.get_random_search_params(
                n_iter=n_iter)
            results = self._run_random_search(engine, data, param_ranges)

        elif method == "bayesian":
            dimensions, param_names = param_selector.get_bayesian_optimization_params(
                n_iter=n_iter)
            results = self._run_bayesian_optimization(engine, data, dimensions, param_names,
                                                      optimization_metric, n_iter)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        if len(results) > 0:
            self.analyze_results(results, method, optimization_metric)
            if save_results:
                self.save_results(results, method)
                self.save_best_parameters(results, method, optimization_metric)
        else:
            print("No results generated from optimization")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nBacktest completed successfully!")
        print(
            f"Total calculation time: {duration:.2f} seconds ({time.strftime('%H:%M:%S', time.gmtime(duration))})")

        return results

    def _run_grid_search(self, engine: VectorBTEngine, data: pl.DataFrame, param_ranges: Dict[str, List]) -> pl.DataFrame:
        """Run grid search optimization."""
        results = engine.simulate_portfolios(
            strategy=self.strategy,
            data=data,
            param_dict=param_ranges,
            ticker=self.symbol,
            sizing_method="Risk percent",
            risk_pct=self.risk_pct,
            exchange_broker=self.exchange.lower(),
            date_range=f"grid_search_{self.config_name}",
            save_results=False,
            indicator_batch_size=50
        )
        return results

    def _run_random_search(self, engine: VectorBTEngine, data: pl.DataFrame, param_ranges: Dict[str, List]) -> pl.DataFrame:
        """Run random search optimization."""
        results = engine.simulate_portfolios(
            strategy=self.strategy,
            data=data,
            param_dict=param_ranges,
            ticker=self.symbol,
            sizing_method="Risk percent",
            risk_pct=self.risk_pct,
            exchange_broker=self.exchange.lower(),
            date_range=f"random_search_{self.config_name}",
            save_results=False,
            indicator_batch_size=50
        )
        return results

    def _run_bayesian_optimization(self,
                                   engine: VectorBTEngine,
                                   data: pl.DataFrame,
                                   dimensions: List,
                                   param_names: List[str],
                                   optimization_metric: str,
                                   n_iter: int) -> pl.DataFrame:
        """Run Bayesian optimization."""
        evaluated_params = []
        all_results = []

        @use_named_args(dimensions)
        def objective(**params):
            """Objective function to minimize."""
            try:
                param_dict = {name: [value] for name, value in params.items()}

                results = engine.simulate_portfolios(
                    strategy=self.strategy,
                    data=data,
                    param_dict=param_dict,
                    ticker=self.symbol,
                    sizing_method="Risk percent",
                    risk_pct=self.risk_pct,
                    exchange_broker=self.exchange.lower(),
                    date_range="bayesian_opt",
                    save_results=False,
                    indicator_batch_size=1
                )

                if len(results) > 0:
                    result_dict = results.to_dicts()[0]
                    metric_value = result_dict[optimization_metric]

                    evaluated_params.append(params.copy())
                    all_results.append(result_dict)

                    print(
                        f"Iteration {len(evaluated_params)}: {optimization_metric} = {metric_value:.3f}")

                    # Return negative value for maximization (gp_minimize minimizes)
                    return -metric_value if optimization_metric in ["sharpe_ratio", "total_return_pct", "win_rate_pct"] else metric_value
                else:
                    return 1000.0  # High penalty for failed backtests

            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1000.0

        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_iter,
            random_state=42,
            acq_func='EI',  # Expected Improvement
            n_initial_points=min(10, n_iter // 2)
        )

        print(f"Bayesian optimization completed!")
        if len(all_results) > 0:
            best_idx = np.argmax([r[optimization_metric] for r in all_results]) if optimization_metric in [
                "sharpe_ratio", "total_return_pct", "win_rate_pct"] else np.argmin([r[optimization_metric] for r in all_results])
            print(
                f"Best {optimization_metric}: {all_results[best_idx][optimization_metric]:.3f}")
            print(f"Best parameters: {evaluated_params[best_idx]}")

        if all_results:
            return pl.DataFrame(all_results)
        else:
            return pl.DataFrame([])

    def analyze_results(self, results: pl.DataFrame, method: str, optimization_metric: str) -> None:
        """
        Analyze and display backtest results.

        Args:
            results: Backtest results DataFrame
            method: Optimization method used
            optimization_metric: Primary optimization metric
        """
        if len(results) == 0:
            print(f"No results to analyze for {method}")
            return

        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS SUMMARY - {method.upper()}")
        print(f"{'='*60}")
        print(f"Total parameter combinations tested: {len(results):,}")
        print(f"Primary optimization metric: {optimization_metric}")

        ascending = optimization_metric not in [
            "sharpe_ratio", "total_return_pct", "win_rate_pct"]
        sorted_results = results.sort(
            optimization_metric, descending=not ascending)

        print(f"\nTOP 5 STRATEGIES BY {optimization_metric.upper()}:")
        top_results = sorted_results.head(5)
        for i, row in enumerate(top_results.iter_rows(named=True), 1):
            print(f"{i}. {optimization_metric}: {row[optimization_metric]:.3f} | "
                  f"Return: {row['total_return_pct']:.1f}% | "
                  f"Sharpe: {row['sharpe_ratio']:.3f} | "
                  f"DD: {row['max_drawdown_pct']:.1f}% | "
                  f"Trades: {row['total_trades']:.0f} | "
                  f"WR: {row['win_rate_pct']:.1f}%")

        print(f"\nPERFORMANCE STATISTICS:")
        print(f"Average Return: {results['total_return_pct'].mean():.1f}%")
        print(f"Average Sharpe: {results['sharpe_ratio'].mean():.3f}")
        print(
            f"Average Max Drawdown: {results['max_drawdown_pct'].mean():.1f}%")
        print(f"Average Win Rate: {results['win_rate_pct'].mean():.1f}%")
        print(
            f"Average {optimization_metric}: {results[optimization_metric].mean():.3f}")

        best_row = sorted_results.head(1).to_dicts()[0]
        print(f"\nBEST PARAMETER COMBINATION:")
        param_columns = [col for col in results.columns if col.startswith(
            ('bbands_', 'cvd_', 'atr_', 'sl_', 'tpsl_'))]
        for param in param_columns:
            print(f"   {param}: {best_row[param]}")

    def save_results(self, results: pl.DataFrame, method: str) -> None:
        """Save results to parquet file."""
        results_file = self.results_dir / \
            f"{self.config_name}_{method}_results.parquet"
        results.write_parquet(results_file)
        print(f"\nResults saved to: {results_file}")

    def save_best_parameters(self, results: pl.DataFrame, method: str, optimization_metric: str) -> None:
        """
        Save the best parameters to a JSON file.

        Args:
            results: Backtest results DataFrame
            method: Optimization method used
            optimization_metric: Primary optimization metric
        """
        if len(results) == 0:
            return

        ascending = optimization_metric not in [
            "sharpe_ratio", "total_return_pct", "win_rate_pct"]
        sorted_results = results.sort(
            optimization_metric, descending=not ascending)
        best_row = sorted_results.head(1).to_dicts()[0]

        param_columns = [col for col in results.columns if col.startswith(
            ('bbands_', 'cvd_', 'atr_', 'sl_', 'tpsl_'))]
        best_params = {col: best_row[col] for col in param_columns}

        best_params['performance'] = {
            'sharpe_ratio': best_row['sharpe_ratio'],
            'total_return_pct': best_row['total_return_pct'],
            'max_drawdown_pct': best_row['max_drawdown_pct'],
            'win_rate_pct': best_row['win_rate_pct'],
            'total_trades': best_row['total_trades'],
            'optimization_metric': optimization_metric,
            'optimization_value': best_row[optimization_metric]
        }

        best_params['metadata'] = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'method': method,
            'config_name': self.config_name
        }

        params_file = self.results_dir / \
            f"{self.config_name}_{method}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)

        print(f"Best parameters saved to: {params_file}")


def example_grid_search():
    """Example: Run grid search optimization."""
    param_selector = ParametersSelection()

    runner = BacktestRunner(
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    results = runner.run_backtest(
        param_selector=param_selector,
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
