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
from typing import Dict, List, Any, Literal
from datetime import datetime, timezone
from skopt import gp_minimize
from skopt.utils import use_named_args


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.data.pipeline.data_preparation import DataPreparationPipeline
    from src.bt_engine.vectorbt_engine import VectorBTEngine
    from src.data.config.data_config import DataConfig, DataSplitConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)

# Lazy import to avoid circular dependencies
_ParametersSelection = None


def _get_parameters_selection():
    """Lazy import of ParametersSelection to avoid circular dependencies."""
    global _ParametersSelection
    if _ParametersSelection is None:
        try:
            from src.optimization.parameters_selector import ParametersSelection
            _ParametersSelection = ParametersSelection
        except ImportError as e:
            print(f"Error importing ParametersSelection: {e}")
            raise
    return _ParametersSelection


class BacktestRunner:
    """
    Enhanced backtest runner with QuestDB integration and multiple optimization methods.
    """

    def __init__(self,
                 strategy,
                 query_service,
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 timeframe: str = "1h",
                 exchange: str = "OKX",
                 initial_cash: float = 1000,
                 fee_pct: float = 0.05,
                 risk_pct: float = 1.0,
                 config_name: str = None,
                 description: str = None,
                 train_pct: float = 0.6,
                 validation_pct: float = 0.2,
                 test_pct: float = 0.2
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

        self.strategy = strategy
        self.query_service = query_service
        self.results_dir = Path("data/backtest_results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.train_pct = train_pct
        self.validation_pct = validation_pct
        self.test_pct = test_pct

        self.config_name = config_name or f"{symbol.lower().replace('-', '_')}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        self.description = description or f"{symbol} {timeframe} data from {start_date} to {end_date}"

    def _format_duration_to_days_hms(self, duration) -> str:
        """
        Format a Polars Duration to days HH:MM:SS format.

        Args:
            duration: Polars Duration value

        Returns:
            String in format "X days HH:MM:SS" or "HH:MM:SS" if less than 1 day
        """
        if duration is None:
            return None

        total_seconds = duration.total_seconds()

        days = int(total_seconds // 86400)
        remaining_seconds = total_seconds % 86400
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)

        return f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"

    def load_data(self, data_type: Literal['train', 'validation', 'test']) -> pl.DataFrame:
        """
        Load data from QuestDB.

        Args:
            data_type: The type of data to load ('train', 'validation', 'test').

        Returns:
            DataFrame containing the requested dataset
        """
        print(f"Loading {data_type} data from QuestDB...")
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

        print("\nDATA CONFIGURATION")
        print("=" * 50)
        print(str(data_config))

        print("\nREADY TO PREPARE DATA")
        print("This will collect and process the data from the data base.")
        response = input("Continue? (y/N): ").strip().lower()

        if response != 'y':
            print("Operation cancelled.")
            return

        pipeline = DataPreparationPipeline(data_config)

        try:
            print("\nStarting data preparation...")
            prepared_datasets = pipeline.prepare_data(save_to_disk=False)

            for split_name, dataset in prepared_datasets.items():
                if len(dataset) > 0:
                    print(f"\n{split_name.upper()} SET:")
                    print(f"   Records: {len(dataset):,}")
                    print(
                        f"   Period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")

            return prepared_datasets[data_type]

        except Exception as e:
            print(f"\nError during preparation: {e}")
            sys.exit(1)

    def run_backtest(self,
                     data_type: Literal['train', 'validation', 'test'],
                     param_ranges: Dict[str, List],
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

        if method == "random" and n_iter > np.prod([len(v) for v in param_ranges.values()]):
            raise ValueError(
                f"n_iter ({n_iter}) cannot be greater than the number of parameter combinations ({np.prod([len(v) for v in param_ranges.values()])})")

        if data_type not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown data type: {data_type}")
        else:
            data = self.load_data(data_type)

        engine = VectorBTEngine(
            initial_cash=self.initial_cash,
            fee_pct=self.fee_pct,
            frequency=self.timeframe
        )

        print(f"\n{'='*60}")
        print(f"RUNNING {method.upper()} OPTIMIZATION")
        print(f"Optimization metric: {optimization_metric}")
        print(f"{'='*60}")

        ParametersSelection = _get_parameters_selection()
        param_selector = ParametersSelection(param_ranges)

        if method == "grid":
            param_ranges = param_selector.get_grid_search_params()
            results = self._run_grid_search(engine, data, param_ranges)

        elif method == "random":
            param_combinations = param_selector.get_random_search_params(
                n_iter=n_iter)
            results = self._run_random_search(engine, data, param_combinations)

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
            ticker=self.symbol,
            param_dict=param_ranges,
            sizing_method="Risk percent",
            risk_pct=self.risk_pct,
            exchange_broker=self.exchange.lower(),
            date_range=f"grid_search_{self.config_name}",
            save_results=False,
            indicator_batch_size=50
        )
        return results

    def _run_random_search(self, engine: VectorBTEngine, data: pl.DataFrame, param_combinations: List[Dict[str, Any]]) -> pl.DataFrame:
        """Run random search optimization."""
        results = engine.simulate_portfolios(
            strategy=self.strategy,
            data=data,
            ticker=self.symbol,
            param_combinations=param_combinations,
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
            param_dict = {name: [value] for name, value in params.items()}

            results = engine.simulate_portfolios(
                strategy=self.strategy,
                data=data,
                ticker=self.symbol,
                param_dict=param_dict,
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

        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_iter,
            random_state=42,
            acq_func='EI',  # Expected Improvement
            n_initial_points=min(10, n_iter // 2),
            n_jobs=-1
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
        param_columns = [col for col in results.columns if any(param in col for param in [
                                                               'bbands_length', 'bbands_stddev', 'cvd_length', 'atr_length', 'sl_coef', 'tpsl_ratio'])]
        for param in param_columns:
            print(f"   {param}: {best_row[param]}")

    def save_results(self, results: pl.DataFrame, method: str) -> None:
        """Save results to csv file."""
        results_file = self.results_dir / \
            f"{self.config_name}_{method}_results.csv"

        for col in results.columns:
            dtype = results[col].dtype

            if dtype == pl.Duration:
                results = results.with_columns(
                    pl.col(col).map_elements(
                        lambda x: self._format_duration_to_days_hms(
                            x) if x is not None else None,
                        return_dtype=pl.Utf8
                    )
                )

        results.write_csv(results_file)
        print(f"\nResults saved to: {results_file}")

        duration_cols = [col for col in results.columns
                         if results[col].dtype == pl.Utf8 and len(results) > 0
                         and any(str(val).count(':') == 2 for val in results[col].head(5) if val is not None)]
        if duration_cols:
            print(
                f"Note: Duration columns {duration_cols} were converted to 'days HH:MM:SS' format for CSV compatibility")

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

        param_columns = [col for col in results.columns if any(param in col for param in [
                                                               'bbands_length', 'bbands_stddev', 'cvd_length', 'atr_length', 'sl_coef', 'tpsl_ratio'])]
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
