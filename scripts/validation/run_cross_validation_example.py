"""
Cross-Validation Example Script

This script demonstrates how to use the TimeSeriesCrossValidator for robust
strategy validation using time-series aware cross-validation methods.

The script includes examples of:
1. Rolling Window Cross-Validation
2. Expanding Window Cross-Validation
3. Blocked Time-Series Cross-Validation
4. Parameter comparison across validation methods
5. Statistical significance testing

Usage:
    python scripts/validation/run_cross_validation_example.py
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.optimization.cross_validator import (
        TimeSeriesCrossValidator,
        ValidationMethod,
    )
    from src.bt_engine.backtest_runner import BacktestRunner
    from src.bt_engine.vectorbt_engine import VectorBTEngine
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
    from src.optimization.parameters_selector import ParametersSelection
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def example_rolling_window_cv():
    """Example: Rolling Window Cross-Validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: ROLLING WINDOW CROSS-VALIDATION")
    print("=" * 80)

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    param_ranges = {
        "bbands_length": [30, 50, 70],
        "bbands_stddev": [2.0, 2.5, 3.0],
        "cvd_length": [40, 50],
        "atr_length": [10, 14],
        "sl_coef": [2.0, 2.5],
        "tpsl_ratio": [2.0, 2.5]
    }

    param_selector = ParametersSelection(param_ranges)
    param_combinations = param_selector.get_random_search_params(n_iter=8)

    cv = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.ROLLING_WINDOW,
        n_splits=4,
        train_size_months=4,
        test_size_months=1,
        purge_days=1,
        embargo_days=0
    )

    results = cv.cross_validate(
        data=data,
        param_combinations=param_combinations,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="rolling_window_cv_example"
    )

    return results


def example_expanding_window_cv():
    """Example: Expanding Window Cross-Validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: EXPANDING WINDOW CROSS-VALIDATION")
    print("=" * 80)

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    param_combinations = [
        {"bbands_length": 30, "bbands_stddev": 2.0, "cvd_length": 40,
            "atr_length": 10, "sl_coef": 2.0, "tpsl_ratio": 2.0},
        {"bbands_length": 50, "bbands_stddev": 2.5, "cvd_length": 50,
            "atr_length": 14, "sl_coef": 2.5, "tpsl_ratio": 2.5},
        {"bbands_length": 70, "bbands_stddev": 3.0, "cvd_length": 40,
            "atr_length": 10, "sl_coef": 2.0, "tpsl_ratio": 3.0},
    ]

    cv = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.EXPANDING_WINDOW,
        n_splits=4,
        initial_train_months=3,
        test_size_months=1,
        purge_days=1
    )

    results = cv.cross_validate(
        data=data,
        param_combinations=param_combinations,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="expanding_window_cv_example"
    )

    return results


def example_blocked_timeseries_cv():
    """Example: Blocked Time-Series Cross-Validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: BLOCKED TIME-SERIES CROSS-VALIDATION")
    print("=" * 80)

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    param_combinations = [
        {"bbands_length": 40, "bbands_stddev": 2.0, "cvd_length": 50,
            "atr_length": 14, "sl_coef": 2.5, "tpsl_ratio": 2.0},
        {"bbands_length": 60, "bbands_stddev": 2.5, "cvd_length": 40,
            "atr_length": 10, "sl_coef": 2.0, "tpsl_ratio": 2.5},
    ]

    cv = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.BLOCKED_TIMESERIES,
        n_splits=5,
        purge_days=2
    )

    results = cv.cross_validate(
        data=data,
        param_combinations=param_combinations,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="blocked_timeseries_cv_example"
    )

    return results


def compare_cv_methods():
    """Compare different cross-validation methods on the same data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: COMPARING CROSS-VALIDATION METHODS")
    print("=" * 80)

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    param_combinations = [
        {"bbands_length": 50, "bbands_stddev": 2.5, "cvd_length": 50,
            "atr_length": 14, "sl_coef": 2.5, "tpsl_ratio": 2.5},
    ]

    methods_to_test = [
        ("rolling_window", ValidationMethod.ROLLING_WINDOW),
        ("expanding_window", ValidationMethod.EXPANDING_WINDOW),
        ("blocked_timeseries", ValidationMethod.BLOCKED_TIMESERIES)
    ]

    comparison_results = {}

    for method_name, method in methods_to_test:
        print(f"\nTesting {method_name.upper()} method...")

        cv = TimeSeriesCrossValidator(
            strategy=strategy,
            engine=engine,
            validation_method=method,
            n_splits=4,
            train_size_months=4,
            test_size_months=1,
            initial_train_months=3,
            purge_days=1
        )

        results = cv.cross_validate(
            data=data,
            param_combinations=param_combinations,
            optimization_metric="sharpe_ratio",
            save_results=True,
            config_name=f"comparison_{method_name}"
        )

        comparison_results[method_name] = {
            'cv_mean': results.cv_mean,
            'cv_std': results.cv_std,
            'confidence_interval': results.confidence_interval,
            'is_significant': results.statistical_significance.get('is_significant', False),
            'p_value': results.statistical_significance.get('p_value', np.nan),
            'n_splits': results.n_splits
        }

    print(f"\n{'='*80}")
    print("CROSS-VALIDATION METHODS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'CV Score':<15} {'Std Dev':<10} {'95% CI':<25} {'Significant':<12} {'P-Value':<10}")
    print("-" * 100)

    for method_name, metrics in comparison_results.items():
        ci_str = f"[{metrics['confidence_interval'][0]:.3f}, {metrics['confidence_interval'][1]:.3f}]"
        print(f"{method_name:<20} "
              f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}   "
              f"{metrics['cv_std']:<10.3f} "
              f"{ci_str:<25} "
              f"{str(metrics['is_significant']):<12} "
              f"{metrics['p_value']:<10.3f}")

    print(f"\nInterpretation:")
    print(f"- Lower std dev indicates more stable performance across folds")
    print(f"- Narrow confidence intervals suggest reliable estimates")
    print(f"- Significant results (p < 0.05) indicate strategy performs better than random")
    print(f"- Rolling window: Tests adaptability to recent market conditions")
    print(f"- Expanding window: Tests performance with increasing data")
    print(f"- Blocked series: Tests performance across different time periods")

    return comparison_results


def example_parameter_sensitivity():
    """Example: Parameter sensitivity analysis using cross-validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    base_params = {
        "bbands_length": 50,
        "bbands_stddev": 2.5,
        "cvd_length": 50,
        "atr_length": 14,
        "sl_coef": 2.5,
        "tpsl_ratio": 2.5
    }

    # Create parameter variations (±20% around base values)
    sensitivity_params = []

    for param_name, base_value in base_params.items():
        if isinstance(base_value, (int, float)):
            variations = [base_value * 0.8, base_value, base_value * 1.2]

            for variation in variations:
                params = base_params.copy()
                if isinstance(base_value, int):
                    params[param_name] = int(round(variation))
                else:
                    params[param_name] = round(variation, 1)

                sensitivity_params.append(params)

    unique_params = []
    seen = set()
    for params in sensitivity_params:
        param_tuple = tuple(sorted(params.items()))
        if param_tuple not in seen:
            seen.add(param_tuple)
            unique_params.append(params)

    print(
        f"Testing {len(unique_params)} parameter combinations for sensitivity analysis...")

    cv = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.ROLLING_WINDOW,
        n_splits=3,
        train_size_months=4,
        test_size_months=1,
        purge_days=1
    )

    results = cv.cross_validate(
        data=data,
        param_combinations=unique_params,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="parameter_sensitivity"
    )

    print(f"\nParameter Sensitivity Analysis:")
    print(
        f"Base parameter set CV score: {results.cv_mean:.4f} ± {results.cv_std:.4f}")
    print(f"Best parameter set found: {results.best_parameters}")
    print(f"Best CV score: {results.best_cv_score:.4f}")

    param_impacts = {}
    for param_name in base_params.keys():
        param_scores = []
        for result in results.fold_results:
            if param_name in result.parameters:
                param_scores.append({
                    'value': result.parameters[param_name],
                    'score': result.test_metrics.get('sharpe_ratio', np.nan)
                })

        if param_scores:
            scores = [s['score']
                      for s in param_scores if not np.isnan(s['score'])]
            if scores:
                param_impacts[param_name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'range': np.max(scores) - np.min(scores)
                }

    print(f"\nParameter Impact Summary:")
    for param, impact in param_impacts.items():
        print(
            f"  {param}: Score range = {impact['range']:.4f}, Std = {impact['std_score']:.4f}")

    return results


def example_robust_validation_workflow():
    """Example: Complete robust validation workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: COMPLETE ROBUST VALIDATION WORKFLOW")
    print("=" * 80)

    # This example demonstrates a complete validation workflow that you might
    # use in production for validating a trading strategy

    print("This workflow includes:")
    print("1. Initial parameter screening with blocked CV")
    print("2. Refined testing with rolling window CV")
    print("3. Final validation with expanding window CV")
    print("4. Statistical significance testing")
    print("5. Robustness assessment")

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=1000,
        fee_pct=0.05,
        frequency="1h"
    )

    runner = BacktestRunner(
        strategy=strategy,
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    data = runner.load_data("train")

    print("\nStep 1: Initial Parameter Screening")
    print("-" * 40)

    broad_param_ranges = {
        "bbands_length": [30, 40, 50, 60, 70],
        "bbands_stddev": [2.0, 2.5, 3.0],
        "cvd_length": [40, 50, 60],
        "atr_length": [10, 14],
        "sl_coef": [2.0, 2.5],
        "tpsl_ratio": [2.0, 2.5, 3.0]
    }

    param_selector = ParametersSelection(broad_param_ranges)
    initial_params = param_selector.get_random_search_params(n_iter=12)

    cv1 = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.BLOCKED_TIMESERIES,
        n_splits=4,
        purge_days=1
    )

    initial_results = cv1.cross_validate(
        data=data,
        param_combinations=initial_params,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="workflow_step1_screening"
    )

    print("\nStep 2: Refined Testing of Top Candidates")
    print("-" * 40)

    # For demonstration, use the best parameter set found
    # In practice, you might select top 3-5 candidates
    top_candidates = [initial_results.best_parameters]

    cv2 = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.ROLLING_WINDOW,
        n_splits=5,
        train_size_months=4,
        test_size_months=1,
        purge_days=1
    )

    refined_results = cv2.cross_validate(
        data=data,
        param_combinations=top_candidates,
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="workflow_step2_refined"
    )

    print("\nStep 3: Final Validation")
    print("-" * 40)

    cv3 = TimeSeriesCrossValidator(
        strategy=strategy,
        engine=engine,
        validation_method=ValidationMethod.EXPANDING_WINDOW,
        n_splits=4,
        initial_train_months=3,
        test_size_months=1,
        purge_days=1
    )

    final_results = cv3.cross_validate(
        data=data,
        param_combinations=[refined_results.best_parameters],
        optimization_metric="sharpe_ratio",
        save_results=True,
        config_name="workflow_step3_final"
    )

    print("\nStep 4: Validation Summary and Recommendations")
    print("-" * 40)

    print(f"Initial screening (Blocked CV):")
    print(
        f"  Best CV score: {initial_results.cv_mean:.4f} ± {initial_results.cv_std:.4f}")
    print(
        f"  Significant: {initial_results.statistical_significance.get('is_significant', 'N/A')}")

    print(f"\nRefined testing (Rolling Window CV):")
    print(
        f"  Best CV score: {refined_results.cv_mean:.4f} ± {refined_results.cv_std:.4f}")
    print(
        f"  Significant: {refined_results.statistical_significance.get('is_significant', 'N/A')}")

    print(f"\nFinal validation (Expanding Window CV):")
    print(
        f"  Best CV score: {final_results.cv_mean:.4f} ± {final_results.cv_std:.4f}")
    print(
        f"  Significant: {final_results.statistical_significance.get('is_significant', 'N/A')}")

    print(f"\nFinal recommended parameters:")
    for param, value in final_results.best_parameters.items():
        print(f"  {param}: {value}")

    cv_scores = [initial_results.cv_mean,
                 refined_results.cv_mean, final_results.cv_mean]
    cv_consistency = np.std(
        cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else np.inf

    print(f"\nRobustness Assessment:")
    print(f"  CV score consistency (CV): {cv_consistency:.3f}")
    print(f"  {'ROBUST' if cv_consistency < 0.3 else 'NEEDS REVIEW' if cv_consistency < 0.5 else 'NOT ROBUST'}")

    if cv_consistency < 0.3:
        print("  ✓ Strategy shows consistent performance across validation methods")
    elif cv_consistency < 0.5:
        print("  ! Strategy shows moderate consistency - consider additional testing")
    else:
        print("  ✗ Strategy shows poor consistency - not recommended for deployment")

    return {
        'initial': initial_results,
        'refined': refined_results,
        'final': final_results,
        'consistency': cv_consistency
    }


if __name__ == "__main__":
    """
    Run cross-validation examples.
    Uncomment the examples you want to run.
    """

    print("Time-Series Cross-Validation Examples")
    print("=" * 80)
    print("These examples demonstrate robust cross-validation methods")
    print("specifically designed for time-series financial data.")
    print()

    try:
        # Example 1: Rolling Window CV
        # print("Running Rolling Window Cross-Validation Example...")
        # example_rolling_window_cv()

        # Example 2: Expanding Window CV
        # print("\nRunning Expanding Window Cross-Validation Example...")
        # example_expanding_window_cv()

        # Example 3: Blocked Time-Series CV
        # print("\nRunning Blocked Time-Series Cross-Validation Example...")
        # example_blocked_timeseries_cv()

        # Example 4: Method Comparison
        # print("\nRunning Cross-Validation Methods Comparison...")
        # compare_cv_methods()

        # Example 5: Parameter Sensitivity (uncomment to run)
        # print("\nRunning Parameter Sensitivity Analysis...")
        # example_parameter_sensitivity()

        # Example 6: Complete Workflow (uncomment to run)
        print("\nRunning Complete Robust Validation Workflow...")
        example_robust_validation_workflow()

        print(f"\n{'='*80}")
        print("ALL CROSS-VALIDATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("Check the results in: data/backtest_results/cross_validation/")
        print()
        print("Next steps:")
        print("1. Review the cross-validation results")
        print("2. Analyze parameter stability across folds")
        print("3. Check statistical significance")
        print("4. Consider the method that best fits your use case")
        print("5. Proceed with final out-of-sample testing")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure QuestDB is running and data is available")
        import traceback
        traceback.print_exc()
