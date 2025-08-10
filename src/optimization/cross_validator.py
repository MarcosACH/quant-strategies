"""
Time-Series Cross-Validation Module

This module implements robust time-series cross-validation methods specifically designed
for quantitative trading strategies. It includes multiple validation techniques with
proper temporal handling to prevent data leakage and ensure reliable out-of-sample results.

Key Features:
- Time-series aware cross-validation (no random shuffling)
- Rolling window validation
- Expanding window validation
- Blocked time-series validation
- Purging and embargo periods
- Comprehensive performance metrics aggregation
- Statistical significance testing

Usage:
    from src.optimization.cross_validator import TimeSeriesCrossValidator
    
    validator = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method="rolling_window",
        n_splits=5,
        train_size_months=6,
        test_size_months=2
    )
    
    results = validator.cross_validate(
        data=data,
        param_combinations=param_combinations,
        optimization_metric="sharpe_ratio"
    )
"""

from scipy.stats import ttest_1samp
from scipy import stats
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from config.settings import settings
from typing import Iterable

import warnings
warnings.filterwarnings('ignore')

# Statistical testing


class ValidationMethod(Enum):
    """Enumeration of available cross-validation methods."""
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    BLOCKED_TIMESERIES = "blocked_timeseries"
    WALK_FORWARD = "walk_forward"


@dataclass
class CVSplitInfo:
    """Information about a single cross-validation split."""
    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    train_data: pl.DataFrame
    test_data: pl.DataFrame


@dataclass
class CVResult:
    """Result from a single cross-validation fold."""
    split_id: int
    parameters: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_duration: float
    test_duration: float
    split_info: CVSplitInfo


@dataclass
class CVSummary:
    """Summary statistics across all cross-validation folds."""
    n_splits: int
    total_parameters_tested: int
    best_parameters: Dict[str, Any]
    best_cv_score: float
    cv_scores: List[float]
    cv_std: float
    cv_mean: float
    confidence_interval: Tuple[float, float]
    statistical_significance: Dict[str, Any]
    fold_results: List[CVResult]
    # metric -> {'mean', 'std', 'min', 'max'}
    overall_metrics: Dict[str, Dict[str, float]]


class TimeSeriesSplitter(ABC):
    """Abstract base class for time-series splitting strategies."""

    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 1,
                 embargo_days: int = 0):
        """
        Initialize the splitter.

        Args:
            n_splits: Number of cross-validation splits
            purge_days: Days to purge between train and test to prevent leakage
            embargo_days: Additional embargo period after test set
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    @abstractmethod
    def split(self, data: pl.DataFrame) -> List[CVSplitInfo]:
        """Generate cross-validation splits."""
        pass

    def _create_split_info(self,
                           split_id: int,
                           data: pl.DataFrame,
                           train_start: datetime,
                           train_end: datetime,
                           test_start: datetime,
                           test_end: datetime) -> CVSplitInfo:
        """Create a CVSplitInfo object with data filtering."""

        train_data = data.filter(
            (pl.col("timestamp") >= train_start) &
            (pl.col("timestamp") <= train_end)
        )

        test_data = data.filter(
            (pl.col("timestamp") >= test_start) &
            (pl.col("timestamp") <= test_end)
        )

        return CVSplitInfo(
            split_id=split_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_samples=len(train_data),
            test_samples=len(test_data),
            train_data=train_data,
            test_data=test_data
        )


class RollingWindowSplitter(TimeSeriesSplitter):
    """Rolling window time-series cross-validation splitter."""

    def __init__(self,
                 n_splits: int = 5,
                 train_size_months: int = 6,
                 test_size_months: int = 2,
                 purge_days: int = 1,
                 embargo_days: int = 0):
        """
        Initialize rolling window splitter.

        Args:
            train_size_months: Size of training window in months
            test_size_months: Size of testing window in months
        """
        super().__init__(n_splits, purge_days, embargo_days)
        self.train_size_months = train_size_months
        self.test_size_months = test_size_months

    def split(self, data: pl.DataFrame) -> List[CVSplitInfo]:
        """Generate rolling window splits."""
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain 'timestamp' column")

        data_start = data["timestamp"].min()
        data_end = data["timestamp"].max()

        train_delta = timedelta(days=self.train_size_months * 30)
        test_delta = timedelta(days=self.test_size_months * 30)
        purge_delta = timedelta(days=self.purge_days)
        embargo_delta = timedelta(days=self.embargo_days)

        total_period = data_end - data_start
        available_period = total_period - train_delta - \
            test_delta - purge_delta - embargo_delta
        step_size = available_period / \
            (self.n_splits - 1) if self.n_splits > 1 else timedelta(0)

        splits = []
        for i in range(self.n_splits):
            window_start = data_start + (step_size * i)

            train_start = window_start
            train_end = train_start + train_delta

            test_start = train_end + purge_delta
            test_end = test_start + test_delta

            if test_end > data_end:
                break

            split_info = self._create_split_info(
                split_id=i,
                data=data,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            if split_info.train_samples > 100 and split_info.test_samples > 50:
                splits.append(split_info)

        return splits


class ExpandingWindowSplitter(TimeSeriesSplitter):
    """Expanding window time-series cross-validation splitter."""

    def __init__(self,
                 n_splits: int = 5,
                 initial_train_months: int = 6,
                 test_size_months: int = 2,
                 purge_days: int = 1,
                 embargo_days: int = 0):
        """
        Initialize expanding window splitter.

        Args:
            initial_train_months: Initial training window size in months
            test_size_months: Size of testing window in months
        """
        super().__init__(n_splits, purge_days, embargo_days)
        self.initial_train_months = initial_train_months
        self.test_size_months = test_size_months

    def split(self, data: pl.DataFrame) -> List[CVSplitInfo]:
        """Generate expanding window splits."""
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain 'timestamp' column")

        data_start = data["timestamp"].min()
        data_end = data["timestamp"].max()

        initial_train_delta = timedelta(days=self.initial_train_months * 30)
        test_delta = timedelta(days=self.test_size_months * 30)
        purge_delta = timedelta(days=self.purge_days)

        remaining_period = data_end - data_start - \
            initial_train_delta - test_delta - purge_delta
        step_size = remaining_period / \
            (self.n_splits - 1) if self.n_splits > 1 else timedelta(0)

        splits = []
        for i in range(self.n_splits):
            train_start = data_start
            train_end = data_start + initial_train_delta + (step_size * i)

            test_start = train_end + purge_delta
            test_end = test_start + test_delta

            if test_end > data_end:
                break

            split_info = self._create_split_info(
                split_id=i,
                data=data,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            if split_info.train_samples > 100 and split_info.test_samples > 50:
                splits.append(split_info)

        return splits


class BlockedTimeSeriesSplitter(TimeSeriesSplitter):
    """Blocked time-series cross-validation splitter."""

    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 1,
                 embargo_days: int = 0):
        """
        Initialize blocked time-series splitter.

        This method divides the entire dataset into n_splits blocks,
        using each block as test set and preceding blocks as training.
        """
        super().__init__(n_splits, purge_days, embargo_days)

    def split(self, data: pl.DataFrame) -> List[CVSplitInfo]:
        """Generate blocked time-series splits."""
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain 'timestamp' column")

        data_start = data["timestamp"].min()
        data_end = data["timestamp"].max()

        total_duration = data_end - data_start
        block_duration = total_duration / self.n_splits
        purge_delta = timedelta(days=self.purge_days)

        splits = []
        for i in range(1, self.n_splits):  # Start from 1 to have training data
            train_start = data_start
            train_end = data_start + (block_duration * i) - purge_delta

            test_start = data_start + (block_duration * i)
            test_end = data_start + (block_duration * (i + 1))

            split_info = self._create_split_info(
                split_id=i - 1,
                data=data,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            if split_info.train_samples > 100 and split_info.test_samples > 50:
                splits.append(split_info)

        return splits


class TimeSeriesCrossValidator:
    """
    Time-Series Cross-Validation for trading strategies.

    This class implements various time-series cross-validation methods
    specifically designed for financial data and trading strategies.
    """

    def __init__(self,
                 strategy,
                 engine,
                 validation_method: Union[str,
                                          ValidationMethod] = ValidationMethod.ROLLING_WINDOW,
                 n_splits: int = 5,
                 train_size_months: int = 6,
                 test_size_months: int = 2,
                 initial_train_months: int = 6,
                 purge_days: int = 1,
                 embargo_days: int = 0,
                 min_train_samples: int = 1000,
                 min_test_samples: int = 500,
                 significance_level: float = 0.05,
                 random_state: int = 42):
        """
        Initialize the time-series cross-validator.

        Args:
            strategy: Trading strategy instance
            engine: Backtesting engine instance
            validation_method: Cross-validation method to use
            n_splits: Number of cross-validation splits
            train_size_months: Training window size in months (for rolling window)
            test_size_months: Test window size in months
            initial_train_months: Initial training size for expanding window
            purge_days: Days to purge between train and test
            embargo_days: Additional embargo period
            min_train_samples: Minimum samples required in training set
            min_test_samples: Minimum samples required in test set
            significance_level: Statistical significance level
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.engine = engine

        if isinstance(validation_method, str):
            validation_method = ValidationMethod(validation_method)
        self.validation_method = validation_method

        self.n_splits = n_splits
        self.train_size_months = train_size_months
        self.test_size_months = test_size_months
        self.initial_train_months = initial_train_months
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.significance_level = significance_level
        self.random_state = random_state

        self.splitter = self._create_splitter()

    @staticmethod
    def _sanitize_for_json(obj: Any):
        """Recursively convert values to JSON-safe types and replace NaN/Inf with null.

        Rules:
        - float/numpy floating: NaN/Inf -> None; else cast to float
        - int/numpy integer: cast to int
        - list/tuple/set: sanitize each element (sets become lists)
        - dict: ensure string keys and sanitize values
        - numpy arrays: convert to list then sanitize
        - datetime/Path: convert to ISO string / path string
        - other basic types (str, bool, None): return as-is
        """
        if obj is None or isinstance(obj, (str, bool)):
            return obj

        if isinstance(obj, (float, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        if isinstance(obj, (int, np.integer)):
            return int(obj)

        if isinstance(obj, np.ndarray):
            return TimeSeriesCrossValidator._sanitize_for_json(obj.tolist())

        if isinstance(obj, (list, tuple)):
            return [TimeSeriesCrossValidator._sanitize_for_json(x) for x in obj]
        if isinstance(obj, set):
            return [TimeSeriesCrossValidator._sanitize_for_json(x) for x in obj]

        if isinstance(obj, dict):
            return {str(k): TimeSeriesCrossValidator._sanitize_for_json(v) for k, v in obj.items()}

        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)

        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return TimeSeriesCrossValidator._sanitize_for_json(obj.item())
            except Exception:
                pass

        return str(obj)

    def _create_splitter(self) -> TimeSeriesSplitter:
        """Create the appropriate splitter based on validation method."""
        if self.validation_method == ValidationMethod.ROLLING_WINDOW:
            return RollingWindowSplitter(
                n_splits=self.n_splits,
                train_size_months=self.train_size_months,
                test_size_months=self.test_size_months,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )
        elif self.validation_method == ValidationMethod.EXPANDING_WINDOW:
            return ExpandingWindowSplitter(
                n_splits=self.n_splits,
                initial_train_months=self.initial_train_months,
                test_size_months=self.test_size_months,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )
        elif self.validation_method == ValidationMethod.BLOCKED_TIMESERIES:
            return BlockedTimeSeriesSplitter(
                n_splits=self.n_splits,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )
        else:
            raise ValueError(
                f"Unsupported validation method: {self.validation_method}")

    def cross_validate(self,
                       data: pl.DataFrame,
                       param_combinations: List[Dict[str, Any]],
                       optimization_metric: str = "sharpe_ratio",
                       parallel: bool = True,
                       save_results: bool = True,
                       config_name: str = "cv_results") -> CVSummary:
        """
        Perform cross-validation on parameter combinations.

        Args:
            data: Market data DataFrame
            param_combinations: List of parameter combinations to test
            optimization_metric: Metric to optimize
            parallel: Whether to use parallel processing (future enhancement)
            save_results: Whether to save results to disk
            config_name: Configuration name for saved results

        Returns:
            CVSummary object with cross-validation results
        """
        print(f"\n{'='*70}")
        print(f"STARTING TIME-SERIES CROSS-VALIDATION")
        print(f"{'='*70}")
        print(f"Method: {self.validation_method.value}")
        print(f"Splits: {self.n_splits}")
        print(f"Parameters to test: {len(param_combinations):,}")
        print(f"Optimization metric: {optimization_metric}")
        print(
            f"Data period: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Data points: {len(data):,}")

        start_time = time.time()

        splits = self.splitter.split(data)
        print(f"\nGenerated {len(splits)} valid cross-validation splits")

        self._print_split_info(splits)

        if len(splits) == 0:
            raise ValueError(
                "No valid splits generated. Check data size and parameters.")

        all_fold_results = []

        for param_idx, params in enumerate(param_combinations):
            print(f"\n{'-'*50}")
            print(
                f"Testing parameter combination {param_idx + 1}/{len(param_combinations)}")
            print(f"Parameters: {params}")

            fold_results = []

            for split in splits:
                print(f"\n  Fold {split.split_id + 1}/{len(splits)} "
                      f"(Train: {split.train_samples:,}, Test: {split.test_samples:,})")

                fold_result = self._run_single_fold(
                    split=split,
                    parameters=params,
                    optimization_metric=optimization_metric
                )

                fold_results.append(fold_result)

                print(
                    f"    Train {optimization_metric}: {fold_result.train_metrics[optimization_metric]:.4f}")
                print(
                    f"    Test {optimization_metric}: {fold_result.test_metrics[optimization_metric]:.4f}")

            all_fold_results.extend(fold_results)

            cv_scores = [result.test_metrics[optimization_metric]
                         for result in fold_results]
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

        summary = self._calculate_cv_summary(
            all_fold_results=all_fold_results,
            optimization_metric=optimization_metric,
            param_combinations=param_combinations,
            n_splits=len(splits)
        )

        self._print_cv_summary(summary, optimization_metric)

        if save_results:
            self._save_cv_results(summary, config_name)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nCross-validation completed in {duration:.2f} seconds")
        print(f"Total folds executed: {len(all_fold_results)}")

        return summary

    def _run_single_fold(self,
                         split: CVSplitInfo,
                         parameters: Dict[str, Any],
                         optimization_metric: str) -> CVResult:
        """Run a single cross-validation fold."""

        fold_start_time = time.time()

        train_results = self.engine.simulate_portfolios(
            strategy=self.strategy,
            data=split.train_data,
            ticker="CV_TRAIN",
            param_combinations=[parameters],
            sizing_method="Risk percent",
            risk_pct=1.0,
            exchange_broker="cv",
            date_range=f"cv_train_fold_{split.split_id}",
            save_results=False,
            indicator_batch_size=1
        )

        train_duration = time.time() - fold_start_time

        test_start_time = time.time()

        test_results = self.engine.simulate_portfolios(
            strategy=self.strategy,
            data=split.test_data,
            ticker="CV_TEST",
            param_combinations=[parameters],
            sizing_method="Risk percent",
            risk_pct=1.0,
            exchange_broker="cv",
            date_range=f"cv_test_fold_{split.split_id}",
            save_results=False,
            indicator_batch_size=1
        )

        test_duration = time.time() - test_start_time

        train_metrics = train_results.to_dicts(
        )[0] if len(train_results) > 0 else {}
        test_metrics = test_results.to_dicts(
        )[0] if len(test_results) > 0 else {}

        return CVResult(
            split_id=split.split_id,
            parameters=parameters,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_duration=train_duration,
            test_duration=test_duration,
            split_info=split
        )

    def _calculate_cv_summary(self,
                              all_fold_results: List[CVResult],
                              optimization_metric: str,
                              param_combinations: List[Dict[str, Any]],
                              n_splits: int) -> CVSummary:
        """Calculate cross-validation summary statistics."""

        param_results = {}
        for result in all_fold_results:
            param_key = str(sorted(result.parameters.items()))
            if param_key not in param_results:
                param_results[param_key] = []
            param_results[param_key].append(result)

        best_cv_score = -np.inf if optimization_metric in [
            "sharpe_ratio", "total_return_pct", "win_rate_pct"] else np.inf
        best_parameters = None
        best_results = None

        param_cv_scores = {}

        for param_key, results in param_results.items():
            cv_scores = [r.test_metrics.get(
                optimization_metric, np.nan) for r in results]
            cv_scores = [score for score in cv_scores if not np.isnan(score)]

            if len(cv_scores) == 0:
                continue

            cv_mean = np.mean(cv_scores)
            param_cv_scores[param_key] = {
                'mean': cv_mean,
                'scores': cv_scores,
                'results': results
            }

            is_better = (optimization_metric in ["sharpe_ratio", "total_return_pct", "win_rate_pct"] and cv_mean > best_cv_score) or \
                (optimization_metric not in [
                 "sharpe_ratio", "total_return_pct", "win_rate_pct"] and cv_mean < best_cv_score)

            if is_better:
                best_cv_score = cv_mean
                best_parameters = results[0].parameters
                best_results = results

        if best_results:
            best_cv_scores = [r.test_metrics.get(
                optimization_metric, np.nan) for r in best_results]
            best_cv_scores = [
                score for score in best_cv_scores if not np.isnan(score)]

            cv_mean = np.mean(best_cv_scores)
            cv_std = np.std(best_cv_scores)

            if len(best_cv_scores) > 1:
                confidence_interval = stats.t.interval(
                    1 - self.significance_level,
                    len(best_cv_scores) - 1,
                    loc=cv_mean,
                    scale=stats.sem(best_cv_scores)
                )
            else:
                confidence_interval = (cv_mean, cv_mean)

            if len(best_cv_scores) > 1:
                t_stat, p_value = ttest_1samp(best_cv_scores, 0)
                is_significant = p_value < self.significance_level
            else:
                t_stat, p_value = np.nan, np.nan
                is_significant = False

            statistical_significance = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'significance_level': self.significance_level
            }
        else:
            cv_mean = np.nan
            cv_std = np.nan
            confidence_interval = (np.nan, np.nan)
            statistical_significance = {}
            best_cv_scores = []

        overall_metrics = self._calculate_overall_metrics(all_fold_results)

        return CVSummary(
            n_splits=n_splits,
            total_parameters_tested=len(param_combinations),
            best_parameters=best_parameters or {},
            best_cv_score=best_cv_score,
            cv_scores=best_cv_scores,
            cv_std=cv_std,
            cv_mean=cv_mean,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            fold_results=all_fold_results,
            overall_metrics=overall_metrics
        )

    def _calculate_overall_metrics(self, all_fold_results: List[CVResult]) -> Dict[str, Dict[str, float]]:
        """Calculate overall metrics across all folds."""
        metrics = {}

        if not all_fold_results:
            return metrics

        sample_metrics = all_fold_results[0].test_metrics.keys()

        for metric in sample_metrics:
            values = []
            for result in all_fold_results:
                if metric in result.test_metrics:
                    metric_value = result.test_metrics[metric]
                    if isinstance(metric_value, (int, float, np.number)) and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                        values.append(metric_value)

            if values:
                metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }

        return metrics

    def _print_split_info(self, splits: List[CVSplitInfo]) -> None:
        """Print information about cross-validation splits."""
        print(f"\n{'Split':<5} {'Train Start':<12} {'Train End':<12} {'Test Start':<12} {'Test End':<12} {'Train Size':<10} {'Test Size':<10}")
        print("-" * 80)

        for split in splits:
            print(f"{split.split_id + 1:<5} "
                  f"{split.train_start.strftime('%Y-%m-%d'):<12} "
                  f"{split.train_end.strftime('%Y-%m-%d'):<12} "
                  f"{split.test_start.strftime('%Y-%m-%d'):<12} "
                  f"{split.test_end.strftime('%Y-%m-%d'):<12} "
                  f"{split.train_samples:<10,} "
                  f"{split.test_samples:<10,}")

    def _print_cv_summary(self, summary: CVSummary, optimization_metric: str) -> None:
        """Print cross-validation summary."""
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")

        print(f"Total splits: {summary.n_splits}")
        print(
            f"Total parameter combinations tested: {summary.total_parameters_tested}")
        print(f"Total folds executed: {len(summary.fold_results)}")

        if summary.best_parameters:
            print(f"\nBEST PARAMETERS (by {optimization_metric}):")
            for param, value in summary.best_parameters.items():
                print(f"  {param}: {value}")

            print(f"\nCROSS-VALIDATION PERFORMANCE:")
            print(f"  CV Score: {summary.cv_mean:.4f} ± {summary.cv_std:.4f}")
            print(
                f"  Confidence Interval (95%): [{summary.confidence_interval[0]:.4f}, {summary.confidence_interval[1]:.4f}]")

            if summary.statistical_significance:
                print(f"\nSTATISTICAL SIGNIFICANCE:")
                print(
                    f"  t-statistic: {summary.statistical_significance['t_statistic']:.4f}")
                print(
                    f"  p-value: {summary.statistical_significance['p_value']:.4f}")
                print(
                    f"  Significant: {summary.statistical_significance['is_significant']}")

            print(f"\nOVERALL METRICS ACROSS ALL FOLDS:")
            for metric, stats in summary.overall_metrics.items():
                print(f"  {metric}:")
                print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    Median: {stats['median']:.4f}")

    def _save_cv_results(self, summary: CVSummary, config_name: str) -> None:
        """Save cross-validation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = settings.RESULTS_ROOT_PATH / "optimization" / "cross_validation" / \
            f"{config_name}_{self.validation_method.value}_summary_{timestamp}.json"

        summary_data = {
            'validation_method': self.validation_method.value,
            'n_splits': summary.n_splits,
            'total_parameters_tested': summary.total_parameters_tested,
            'best_parameters': summary.best_parameters,
            'best_cv_score': summary.best_cv_score,
            'cv_mean': summary.cv_mean,
            'cv_std': summary.cv_std,
            'confidence_interval': summary.confidence_interval,
            'statistical_significance': summary.statistical_significance,
            'overall_metrics': summary.overall_metrics,
            'configuration': {
                'train_size_months': self.train_size_months,
                'test_size_months': self.test_size_months,
                'purge_days': self.purge_days,
                'embargo_days': self.embargo_days,
                'significance_level': self.significance_level
            }
        }

        # Sanitize and save summary with strict JSON (no NaN/Inf)
        with open(summary_file, 'w') as f:
            json.dump(self._sanitize_for_json(summary_data),
                      f, indent=2, allow_nan=False)

        detailed_file = settings.RESULTS_ROOT_PATH / "optimization" / "cross_validation" / \
            f"{config_name}_{self.validation_method.value}_detailed_{timestamp}.json"

        detailed_data = []
        for result in summary.fold_results:
            detailed_data.append({
                'split_id': result.split_id,
                'parameters': result.parameters,
                'train_metrics': result.train_metrics,
                'test_metrics': result.test_metrics,
                'train_duration': result.train_duration,
                'test_duration': result.test_duration,
                'split_info': {
                    'train_start': result.split_info.train_start.isoformat(),
                    'train_end': result.split_info.train_end.isoformat(),
                    'test_start': result.split_info.test_start.isoformat(),
                    'test_end': result.split_info.test_end.isoformat(),
                    'train_samples': result.split_info.train_samples,
                    'test_samples': result.split_info.test_samples
                }
            })

        # Sanitize and save detailed results with strict JSON (no NaN/Inf)
        with open(detailed_file, 'w') as f:
            json.dump(self._sanitize_for_json(detailed_data),
                      f, indent=2, allow_nan=False)

        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")

    def plot_cv_results(self, summary: CVSummary, optimization_metric: str = "sharpe_ratio"):
        """
        Plot cross-validation results (placeholder for future visualization).

        This method can be extended to create visualizations of:
        - CV scores across folds
        - Parameter sensitivity
        - Performance stability over time
        - Metric distributions
        """
        print("Visualization methods can be implemented here using matplotlib/plotly")
        print("Suggested plots:")
        print("- CV scores across folds")
        print("- Parameter sensitivity analysis")
        print("- Performance stability over time")
        print("- Metric distributions")
        pass


def rolling_window_cv(strategy, engine, data, param_combinations, **kwargs):
    """Convenience function for rolling window cross-validation."""
    validator = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.ROLLING_WINDOW,
        **kwargs
    )
    return validator.cross_validate(data, param_combinations)


def expanding_window_cv(strategy, engine, data, param_combinations, **kwargs):
    """Convenience function for expanding window cross-validation."""
    validator = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.EXPANDING_WINDOW,
        **kwargs
    )
    return validator.cross_validate(data, param_combinations)


def blocked_timeseries_cv(strategy, engine, data, param_combinations, **kwargs):
    """Convenience function for blocked time-series cross-validation."""
    validator = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.BLOCKED_TIMESERIES,
        **kwargs
    )
    return validator.cross_validate(data, param_combinations)
