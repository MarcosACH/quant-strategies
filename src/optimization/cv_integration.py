"""
Cross-Validation Integration Module

This module provides integration between the cross-validation system and the existing
optimization workflow. It includes enhanced parameter selection with CV validation
and integrated reporting.

Key Features:
- Enhanced parameter selection with CV validation
- Integration with existing BacktestRunner
- Automated CV workflow execution
- Performance comparison across validation methods
- Robust parameter recommendation system

Usage:
    from src.optimization.cv_integration import CVIntegratedOptimizer
    
    optimizer = CVIntegratedOptimizer(
        strategy=strategy,
        query_service=query_service,
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31"
    )
    
    results = optimizer.run_cv_optimization(
        param_ranges=param_ranges,
        cv_method="rolling_window"
    )
"""

import sys
import polars as pl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import time
from dataclasses import dataclass

from .cross_validator import (
    TimeSeriesCrossValidator,
    ValidationMethod,
    CVSummary
)
from .parameters_selector import ParametersSelection

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Use lazy imports to avoid circular dependencies
_BacktestRunner = None
_VectorBTEngine = None


def _get_backtest_runner():
    """Lazy import of BacktestRunner to avoid circular dependencies."""
    global _BacktestRunner
    if _BacktestRunner is None:
        try:
            from bt_engine.backtest_runner import BacktestRunner
            _BacktestRunner = BacktestRunner
        except ImportError as e:
            print(f"Error importing BacktestRunner: {e}")
            raise
    return _BacktestRunner


def _get_vectorbt_engine():
    """Lazy import of VectorBTEngine to avoid circular dependencies."""
    global _VectorBTEngine
    if _VectorBTEngine is None:
        try:
            from bt_engine.vectorbt_engine import VectorBTEngine
            _VectorBTEngine = VectorBTEngine
        except ImportError as e:
            print(f"Error importing VectorBTEngine: {e}")
            raise
    return _VectorBTEngine


@dataclass
class CVOptimizationResult:
    """Result from cross-validation based optimization."""
    cv_summary: CVSummary
    best_parameters: Dict[str, Any]
    cv_score: float
    cv_std: float
    validation_method: str
    optimization_details: Dict[str, Any]
    final_validation_results: Optional[pl.DataFrame] = None


class CVIntegratedOptimizer:
    """
    Integrated optimizer that combines parameter optimization with cross-validation.

    This class provides a complete workflow for robust parameter selection using
    time-series cross-validation methods.
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
                 train_pct: float = 0.6,
                 validation_pct: float = 0.2,
                 test_pct: float = 0.2):
        """
        Initialize the CV-integrated optimizer.

        Args:
            strategy: Trading strategy instance
            query_service: Data query service
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe
            exchange: Exchange name
            initial_cash: Initial cash amount
            fee_pct: Trading fee percentage
            risk_pct: Risk percentage for position sizing
            train_pct: Training data percentage
            validation_pct: Validation data percentage  
            test_pct: Test data percentage
        """
        self.strategy = strategy
        self.query_service = query_service
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.exchange = exchange
        self.initial_cash = initial_cash
        self.fee_pct = fee_pct
        self.risk_pct = risk_pct

        self.train_pct = train_pct
        self.validation_pct = validation_pct
        self.test_pct = test_pct

        BacktestRunner = _get_backtest_runner()
        self.runner = BacktestRunner(
            strategy=strategy,
            query_service=query_service,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            exchange=exchange,
            initial_cash=initial_cash,
            fee_pct=fee_pct,
            risk_pct=risk_pct,
            train_pct=train_pct,
            validation_pct=validation_pct,
            test_pct=test_pct
        )

        VectorBTEngine = _get_vectorbt_engine()
        self.engine = VectorBTEngine(
            initial_cash=initial_cash,
            fee_pct=fee_pct,
            frequency=timeframe
        )

        self.results_dir = Path("data/backtest_results/cv_optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_cv_optimization(self,
                            param_ranges: Dict[str, List],
                            cv_method: Union[str,
                                             ValidationMethod] = "rolling_window",
                            n_splits: int = 5,
                            train_size_months: int = 6,
                            test_size_months: int = 2,
                            initial_train_months: int = 6,
                            purge_days: int = 1,
                            optimization_metric: str = "sharpe_ratio",
                            param_selection_method: str = "random",
                            n_param_combinations: int = 20,
                            save_results: bool = True,
                            run_final_validation: bool = True) -> CVOptimizationResult:
        """
        Run complete cross-validation based optimization.

        Args:
            param_ranges: Parameter ranges to optimize
            cv_method: Cross-validation method
            n_splits: Number of CV splits
            train_size_months: Training window size (for rolling window)
            test_size_months: Test window size
            initial_train_months: Initial training size (for expanding window)
            purge_days: Days to purge between train/test
            optimization_metric: Metric to optimize
            param_selection_method: Parameter selection method ("grid", "random", "bayesian")
            n_param_combinations: Number of parameter combinations to test
            save_results: Whether to save results
            run_final_validation: Whether to run final validation on test set

        Returns:
            CVOptimizationResult with optimization results
        """
        print(f"\n{'='*80}")
        print(f"CV-INTEGRATED OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"CV Method: {cv_method}")
        print(f"Optimization Metric: {optimization_metric}")
        print(f"Parameter Selection: {param_selection_method}")

        start_time = time.time()

        print(f"\nLoading data...")
        data = self.runner.load_training_data()
        print(f"Data loaded: {len(data):,} records")

        print(f"\nGenerating parameter combinations...")
        param_selector = ParametersSelection(param_ranges)

        if param_selection_method == "grid":
            param_combinations = param_selector.get_grid_search_params(
                reduced=True)
            param_combinations = self._grid_to_combinations(param_combinations)
        elif param_selection_method == "random":
            param_combinations = param_selector.get_random_search_params(
                n_iter=n_param_combinations
            )
        elif param_selection_method == "bayesian":
            # For Bayesian, use random for CV and then refine
            param_combinations = param_selector.get_random_search_params(
                n_iter=n_param_combinations
            )
        else:
            raise ValueError(
                f"Unknown parameter selection method: {param_selection_method}")

        print(f"Generated {len(param_combinations)} parameter combinations")

        print(f"\nSetting up cross-validator...")
        if isinstance(cv_method, str):
            cv_method = ValidationMethod(cv_method)

        cv = TimeSeriesCrossValidator(
            strategy=self.strategy,
            engine=self.engine,
            validation_method=cv_method,
            n_splits=n_splits,
            train_size_months=train_size_months,
            test_size_months=test_size_months,
            initial_train_months=initial_train_months,
            purge_days=purge_days
        )

        print(f"\nRunning cross-validation...")
        cv_summary = cv.cross_validate(
            data=data,
            param_combinations=param_combinations,
            optimization_metric=optimization_metric,
            save_results=save_results,
            config_name=f"cv_opt_{self.symbol}_{cv_method.value}"
        )

        final_validation_results = None
        if run_final_validation and cv_summary.best_parameters:
            print(f"\nRunning final validation on hold-out test set...")
            final_validation_results = self._run_final_validation(
                cv_summary.best_parameters,
                optimization_metric
            )

        result = CVOptimizationResult(
            cv_summary=cv_summary,
            best_parameters=cv_summary.best_parameters,
            cv_score=cv_summary.cv_mean,
            cv_std=cv_summary.cv_std,
            validation_method=cv_method.value,
            optimization_details={
                'param_selection_method': param_selection_method,
                'n_param_combinations': len(param_combinations),
                'n_splits': n_splits,
                'train_size_months': train_size_months,
                'test_size_months': test_size_months,
                'purge_days': purge_days,
                'optimization_metric': optimization_metric
            },
            final_validation_results=final_validation_results
        )

        self._print_optimization_summary(result)

        if save_results:
            self._save_optimization_results(result)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nOptimization completed in {duration:.2f} seconds")

        return result

    def compare_cv_methods(self,
                           param_ranges: Dict[str, List],
                           optimization_metric: str = "sharpe_ratio",
                           n_param_combinations: int = 10,
                           save_results: bool = True) -> Dict[str, CVOptimizationResult]:
        """
        Compare different cross-validation methods on the same parameter space.

        Args:
            param_ranges: Parameter ranges to test
            optimization_metric: Metric to optimize
            n_param_combinations: Number of parameter combinations to test
            save_results: Whether to save results

        Returns:
            Dictionary mapping CV method names to optimization results
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION METHODS COMPARISON")
        print(f"{'='*80}")

        methods_to_compare = [
            ("rolling_window", ValidationMethod.ROLLING_WINDOW),
            ("expanding_window", ValidationMethod.EXPANDING_WINDOW),
            ("blocked_timeseries", ValidationMethod.BLOCKED_TIMESERIES)
        ]

        comparison_results = {}

        for method_name, method in methods_to_compare:
            print(f"\nTesting {method_name.upper()} method...")

            result = self.run_cv_optimization(
                param_ranges=param_ranges,
                cv_method=method,
                n_splits=4,
                optimization_metric=optimization_metric,
                param_selection_method="random",
                n_param_combinations=n_param_combinations,
                save_results=save_results,
                run_final_validation=False
            )

            comparison_results[method_name] = result

        self._print_methods_comparison(comparison_results, optimization_metric)

        if save_results:
            self._save_methods_comparison(comparison_results)

        return comparison_results

    def run_complete_optimization_workflow(self,
                                           param_ranges: Dict[str, List],
                                           optimization_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Run a complete robust optimization workflow with multiple validation stages.

        This implements a multi-stage validation process:
        1. Initial parameter screening with blocked CV
        2. Refined testing with rolling window CV  
        3. Final validation with expanding window CV
        4. Out-of-sample testing on hold-out set

        Args:
            param_ranges: Parameter ranges to optimize
            optimization_metric: Metric to optimize

        Returns:
            Dictionary with results from all stages
        """
        print(f"\n{'='*80}")
        print(f"ROBUST OPTIMIZATION WORKFLOW")
        print(f"{'='*80}")

        workflow_results = {}

        print(f"\nSTAGE 1: Initial Parameter Screening (Blocked CV)")
        print("-" * 60)

        stage1_result = self.run_cv_optimization(
            param_ranges=param_ranges,
            cv_method=ValidationMethod.BLOCKED_TIMESERIES,
            n_splits=4,
            optimization_metric=optimization_metric,
            param_selection_method="random",
            n_param_combinations=30,
            save_results=True,
            run_final_validation=False
        )

        workflow_results['stage1_screening'] = stage1_result

        print(f"\nSTAGE 2: Refined Testing (Rolling Window CV)")
        print("-" * 60)

        if stage1_result.best_parameters:
            refined_ranges = self._create_refined_param_ranges(
                stage1_result.best_parameters,
                original_ranges=param_ranges
            )

            stage2_result = self.run_cv_optimization(
                param_ranges=refined_ranges,
                cv_method=ValidationMethod.ROLLING_WINDOW,
                n_splits=5,
                train_size_months=4,
                test_size_months=1,
                optimization_metric=optimization_metric,
                param_selection_method="random",
                n_param_combinations=20,
                save_results=True,
                run_final_validation=False
            )

            workflow_results['stage2_refined'] = stage2_result
        else:
            print("No valid parameters found in stage 1, skipping stage 2")
            workflow_results['stage2_refined'] = None
            stage2_result = None

        print(f"\nSTAGE 3: Final Validation (Expanding Window CV)")
        print("-" * 60)

        if stage2_result and stage2_result.best_parameters:
            best_param_ranges = {
                param: [value] for param, value in stage2_result.best_parameters.items()
            }

            stage3_result = self.run_cv_optimization(
                param_ranges=best_param_ranges,
                cv_method=ValidationMethod.EXPANDING_WINDOW,
                n_splits=4,
                initial_train_months=3,
                test_size_months=1,
                optimization_metric=optimization_metric,
                param_selection_method="grid",
                save_results=True,
                run_final_validation=True
            )

            workflow_results['stage3_final'] = stage3_result
        else:
            print("No valid parameters from stage 2, skipping stage 3")
            workflow_results['stage3_final'] = None
            stage3_result = None

        self._print_workflow_summary(workflow_results, optimization_metric)

        self._save_workflow_results(workflow_results)

        return workflow_results

    def _grid_to_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Convert parameter grid to list of combinations."""
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations

    def _run_final_validation(self,
                              best_parameters: Dict[str, Any],
                              optimization_metric: str) -> pl.DataFrame:
        """Run final validation on hold-out test set."""

        # This would use the test split from your data configuration
        # For now, we'll simulate this by using the existing runner
        results = self.runner.run_backtest(
            param_ranges={param: [value]
                          for param, value in best_parameters.items()},
            method="grid",
            optimization_metric=optimization_metric,
            save_results=False
        )

        if len(results) > 0:
            final_result = results.head(1)
            print(
                f"Final validation {optimization_metric}: {final_result[optimization_metric].item():.4f}")
            return final_result
        else:
            print("Final validation failed - no results generated")
            return pl.DataFrame()

    def _create_refined_param_ranges(self,
                                     best_params: Dict[str, Any],
                                     original_ranges: Dict[str, List],
                                     refinement_factor: float = 0.2) -> Dict[str, List]:
        """Create refined parameter ranges around best parameters."""
        refined_ranges = {}

        for param, best_value in best_params.items():
            if param in original_ranges:
                original_range = original_ranges[param]

                if isinstance(best_value, (int, float)):
                    range_size = (max(original_range) -
                                  min(original_range)) * refinement_factor

                    if isinstance(best_value, int):
                        refined_min = max(min(original_range),
                                          int(best_value - range_size))
                        refined_max = min(max(original_range),
                                          int(best_value + range_size))
                        refined_ranges[param] = list(range(refined_min, refined_max + 1,
                                                           max(1, (refined_max - refined_min) // 5)))
                    else:
                        refined_min = max(min(original_range),
                                          best_value - range_size)
                        refined_max = min(max(original_range),
                                          best_value + range_size)
                        step = (refined_max - refined_min) / 5
                        refined_ranges[param] = [round(refined_min + i * step, 2)
                                                 for i in range(6)]
                else:
                    refined_ranges[param] = original_range
            else:
                refined_ranges[param] = [best_value]

        return refined_ranges

    def _print_optimization_summary(self, result: CVOptimizationResult):
        """Print optimization summary."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*60}")

        print(f"Validation Method: {result.validation_method}")
        print(f"CV Score: {result.cv_score:.4f} ± {result.cv_std:.4f}")

        if result.cv_summary.statistical_significance:
            sig = result.cv_summary.statistical_significance
            print(
                f"Statistical Significance: {'Yes' if sig.get('is_significant', False) else 'No'}")
            print(f"P-Value: {sig.get('p_value', 'N/A')}")

        print(f"\nBest Parameters:")
        for param, value in result.best_parameters.items():
            print(f"  {param}: {value}")

        if result.final_validation_results is not None and len(result.final_validation_results) > 0:
            final_score = result.final_validation_results[result.optimization_details['optimization_metric']].item(
            )
            print(f"\nFinal Validation Score: {final_score:.4f}")

    def _print_methods_comparison(self,
                                  comparison_results: Dict[str, CVOptimizationResult],
                                  optimization_metric: str):
        """Print comparison of CV methods."""
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION METHODS COMPARISON")
        print(f"{'='*80}")

        print(
            f"{'Method':<20} {'CV Score':<15} {'Std Dev':<10} {'Significant':<12} {'Best Score':<12}")
        print("-" * 80)

        for method_name, result in comparison_results.items():
            sig_text = "Yes" if result.cv_summary.statistical_significance.get(
                'is_significant', False) else "No"
            print(f"{method_name:<20} "
                  f"{result.cv_score:.3f} ± {result.cv_std:.3f}   "
                  f"{result.cv_std:<10.3f} "
                  f"{sig_text:<12} "
                  f"{result.cv_summary.best_cv_score:<12.3f}")

    def _print_workflow_summary(self,
                                workflow_results: Dict[str, Any],
                                optimization_metric: str):
        """Print workflow summary."""
        print(f"\n{'='*80}")
        print(f"ROBUST OPTIMIZATION WORKFLOW SUMMARY")
        print(f"{'='*80}")

        for stage_name, result in workflow_results.items():
            if result is not None:
                print(f"\n{stage_name.upper()}:")
                print(
                    f"  CV Score: {result.cv_score:.4f} ± {result.cv_std:.4f}")
                if result.cv_summary.statistical_significance:
                    sig = result.cv_summary.statistical_significance.get(
                        'is_significant', False)
                    print(f"  Significant: {'Yes' if sig else 'No'}")

                if result.final_validation_results is not None and len(result.final_validation_results) > 0:
                    final_score = result.final_validation_results[optimization_metric].item(
                    )
                    print(f"  Final Validation: {final_score:.4f}")

        if workflow_results.get('stage3_final'):
            final_result = workflow_results['stage3_final']
            print(f"\nOVERALL ASSESSMENT:")
            print(f"Recommended Parameters: {final_result.best_parameters}")
            print(
                f"Expected Performance: {final_result.cv_score:.4f} ± {final_result.cv_std:.4f}")

            if final_result.cv_summary.statistical_significance.get('is_significant', False):
                print(f"Status: ✓ READY FOR DEPLOYMENT")
            else:
                print(f"Status: ⚠ NEEDS FURTHER VALIDATION")
        else:
            print(f"\nOVERALL ASSESSMENT: ✗ OPTIMIZATION FAILED")

    def _save_optimization_results(self, result: CVOptimizationResult):
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cv_optimization_{self.symbol}_{result.validation_method}_{timestamp}.json"
        filepath = self.results_dir / filename

        data = {
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'validation_method': result.validation_method,
            'best_parameters': result.best_parameters,
            'cv_score': result.cv_score,
            'cv_std': result.cv_std,
            'optimization_details': result.optimization_details,
            'statistical_significance': result.cv_summary.statistical_significance,
            'confidence_interval': result.cv_summary.confidence_interval,
            'timestamp': timestamp
        }

        if result.final_validation_results is not None and len(result.final_validation_results) > 0:
            data['final_validation'] = result.final_validation_results.to_dicts()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Optimization results saved: {filepath}")

    def _save_methods_comparison(self, comparison_results: Dict[str, CVOptimizationResult]):
        """Save methods comparison results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cv_methods_comparison_{self.symbol}_{timestamp}.json"
        filepath = self.results_dir / filename

        data = {
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'timestamp': timestamp,
            'methods': {}
        }

        for method_name, result in comparison_results.items():
            data['methods'][method_name] = {
                'cv_score': result.cv_score,
                'cv_std': result.cv_std,
                'best_parameters': result.best_parameters,
                'statistical_significance': result.cv_summary.statistical_significance
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Methods comparison saved: {filepath}")

    def _save_workflow_results(self, workflow_results: Dict[str, Any]):
        """Save workflow results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robust_workflow_{self.symbol}_{timestamp}.json"
        filepath = self.results_dir / filename

        data = {
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'timestamp': timestamp,
            'stages': {}
        }

        for stage_name, result in workflow_results.items():
            if result is not None:
                data['stages'][stage_name] = {
                    'cv_score': result.cv_score,
                    'cv_std': result.cv_std,
                    'best_parameters': result.best_parameters,
                    'validation_method': result.validation_method,
                    'statistical_significance': result.cv_summary.statistical_significance
                }

                if result.final_validation_results is not None and len(result.final_validation_results) > 0:
                    data['stages'][stage_name]['final_validation'] = result.final_validation_results.to_dicts()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Workflow results saved: {filepath}")
