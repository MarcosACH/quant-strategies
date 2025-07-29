"""
Integrated Cross-Validation Example

This script demonstrates how to integrate the new cross-validation system
with your existing optimization workflow. It shows how to use the 
CVIntegratedOptimizer for robust parameter selection.

Usage:
    python scripts/validation/run_integrated_cv_example.py
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.optimization.cv_integration import CVIntegratedOptimizer
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.data.query.questdb_market_data_query import QuestDBMarketDataQuery
    from src.optimization import ValidationMethod
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def example_integrated_cv_optimization():
    """Example: Complete CV-integrated optimization workflow."""
    print("\n" + "=" * 80)
    print("INTEGRATED CROSS-VALIDATION OPTIMIZATION EXAMPLE")
    print("=" * 80)

    optimizer = CVIntegratedOptimizer(
        strategy=CVDBBPullbackStrategy(),
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    param_ranges = {
        "bbands_length": [30, 40, 50, 60, 70, 80],
        "bbands_stddev": [2.0, 2.2, 2.5, 2.8, 3.0, 3.2],
        "cvd_length": [35, 40, 45, 50, 55, 60],
        "atr_length": [8, 10, 12, 14, 16, 18],
        "sl_coef": [1.8, 2.0, 2.2, 2.5, 2.8, 3.0],
        "tpsl_ratio": [1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    }

    result = optimizer.run_cv_optimization(
        param_ranges=param_ranges,
        cv_method="rolling_window",
        n_splits=5,
        train_size_months=4,
        test_size_months=1,
        optimization_metric="sharpe_ratio",
        param_selection_method="random",
        n_param_combinations=5,
        save_results=True,
        run_final_validation=True
    )

    print(f"\nOPTIMIZATION COMPLETED!")
    print(f"Best CV Score: {result.cv_score:.4f} ± {result.cv_std:.4f}")
    print(f"Recommended Parameters: {result.best_parameters}")

    return result


def example_cv_methods_comparison():
    """Example: Compare different CV methods."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION METHODS COMPARISON EXAMPLE")
    print("=" * 80)

    optimizer = CVIntegratedOptimizer(
        strategy=CVDBBPullbackStrategy(),
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    param_ranges = {
        "bbands_length": [40, 50, 60],
        "bbands_stddev": [2.0, 2.5, 3.0],
        "cvd_length": [40, 50],
        "atr_length": [10, 14],
        "sl_coef": [2.0, 2.5],
        "tpsl_ratio": [2.0, 2.5]
    }

    comparison_results = optimizer.compare_cv_methods(
        param_ranges=param_ranges,
        optimization_metric="sharpe_ratio",
        n_param_combinations=8,
        save_results=True
    )

    print(f"\nCOMPARISON COMPLETED!")
    print(f"Methods tested: {list(comparison_results.keys())}")

    best_method = None
    best_score = -np.inf

    for method, result in comparison_results.items():
        if result.cv_score > best_score:
            best_score = result.cv_score
            best_method = method

    print(
        f"Best performing method: {best_method} (CV Score: {best_score:.4f})")

    return comparison_results


def example_robust_workflow():
    """Example: Complete robust optimization workflow."""
    print("\n" + "=" * 80)
    print("ROBUST OPTIMIZATION WORKFLOW EXAMPLE")
    print("=" * 80)

    optimizer = CVIntegratedOptimizer(
        strategy=CVDBBPullbackStrategy(),
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    param_ranges = {
        "bbands_length": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "bbands_stddev": [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5],
        "cvd_length": [30, 35, 40, 45, 50, 55, 60, 65],
        "atr_length": [6, 8, 10, 12, 14, 16, 18, 20],
        "sl_coef": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5],
        "tpsl_ratio": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]
    }

    workflow_results = optimizer.run_complete_optimization_workflow(
        param_ranges=param_ranges,
        optimization_metric="sharpe_ratio"
    )

    print(f"\nROBUST WORKFLOW COMPLETED!")

    final_stage = workflow_results.get('stage3_final')
    if final_stage:
        print(f"Final Recommended Parameters: {final_stage.best_parameters}")
        print(
            f"Expected Performance: {final_stage.cv_score:.4f} ± {final_stage.cv_std:.4f}")

        if final_stage.cv_summary.statistical_significance.get('is_significant', False):
            print(f"Strategy is statistically significant and ready for deployment!")
        else:
            print(
                f"Strategy is not statistically significant - consider further optimization")
    else:
        print(f"Robust workflow did not find suitable parameters")

    return workflow_results


def example_custom_cv_configuration():
    """Example: Custom cross-validation configuration."""
    print("\n" + "=" * 80)
    print("CUSTOM CROSS-VALIDATION CONFIGURATION EXAMPLE")
    print("=" * 80)

    optimizer = CVIntegratedOptimizer(
        strategy=CVDBBPullbackStrategy(),
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    param_ranges = {
        "bbands_length": [45, 50, 55],
        "bbands_stddev": [2.2, 2.5, 2.8],
        "cvd_length": [45, 50, 55],
        "atr_length": [12, 14, 16],
        "sl_coef": [2.2, 2.5, 2.8],
        "tpsl_ratio": [2.2, 2.5, 2.8]
    }

    result = optimizer.run_cv_optimization(
        param_ranges=param_ranges,
        cv_method=ValidationMethod.ROLLING_WINDOW,
        n_splits=8,
        train_size_months=2,
        test_size_months=0.5,
        purge_days=3,
        optimization_metric="sharpe_ratio",
        param_selection_method="grid",
        save_results=True,
        run_final_validation=True
    )

    print(f"\nCUSTOM CONFIGURATION COMPLETED!")
    print(f"Configuration used:")
    print(f"  - Splits: 8")
    print(f"  - Training: 2 months")
    print(f"  - Testing: 2 weeks")
    print(f"  - Purge: 3 days")
    print(f"  - Method: Grid search")

    print(f"\nResults:")
    print(f"  CV Score: {result.cv_score:.4f} ± {result.cv_std:.4f}")
    print(f"  Best Parameters: {result.best_parameters}")

    return result


def example_parameter_stability_analysis():
    """Example: Analyze parameter stability across different CV methods."""
    print("\n" + "=" * 80)
    print("PARAMETER STABILITY ANALYSIS EXAMPLE")
    print("=" * 80)

    optimizer = CVIntegratedOptimizer(
        strategy=CVDBBPullbackStrategy(),
        query_service=QuestDBMarketDataQuery(),
        symbol="BTC-USDT-SWAP",
        start_date="2022-01-01",
        end_date="2022-12-31",
        timeframe="1h",
        initial_cash=1000,
        fee_pct=0.05,
        risk_pct=1.0
    )

    param_ranges = {
        "bbands_length": [40, 45, 50, 55, 60],
        "bbands_stddev": [2.0, 2.3, 2.5, 2.7, 3.0],
        "cvd_length": [40, 45, 50, 55],
        "atr_length": [10, 12, 14, 16],
        "sl_coef": [2.0, 2.3, 2.5, 2.7],
        "tpsl_ratio": [2.0, 2.3, 2.5, 2.7]
    }

    methods_to_test = [
        ("rolling_3m", {"cv_method": "rolling_window",
         "train_size_months": 3, "test_size_months": 1}),
        ("rolling_4m", {"cv_method": "rolling_window",
         "train_size_months": 4, "test_size_months": 1}),
        ("rolling_5m", {"cv_method": "rolling_window",
         "train_size_months": 5, "test_size_months": 1}),
        ("expanding", {"cv_method": "expanding_window",
         "initial_train_months": 3, "test_size_months": 1}),
    ]

    stability_results = {}

    for method_name, config in methods_to_test:
        print(f"\nTesting {method_name}...")

        result = optimizer.run_cv_optimization(
            param_ranges=param_ranges,
            optimization_metric="sharpe_ratio",
            param_selection_method="random",
            n_param_combinations=15,
            save_results=False,
            run_final_validation=False,
            **config
        )

        stability_results[method_name] = {
            'best_params': result.best_parameters,
            'cv_score': result.cv_score,
            'cv_std': result.cv_std
        }

    print(f"\nSTABILITY ANALYSIS:")
    print(f"{'Method':<12} {'CV Score':<12} {'bbands_len':<10} {'bbands_std':<10} {'cvd_len':<8} {'atr_len':<8}")
    print("-" * 70)

    param_variations = {}
    for param in param_ranges.keys():
        param_variations[param] = []

    for method_name, result in stability_results.items():
        best_params = result['best_params']
        print(f"{method_name:<12} "
              f"{result['cv_score']:.3f} ± {result['cv_std']:.3f}  "
              f"{best_params.get('bbands_length', 'N/A'):<10} "
              f"{best_params.get('bbands_stddev', 'N/A'):<10} "
              f"{best_params.get('cvd_length', 'N/A'):<8} "
              f"{best_params.get('atr_length', 'N/A'):<8}")

        for param, value in best_params.items():
            if param in param_variations:
                param_variations[param].append(value)

    print(f"\nPARAMETER STABILITY (lower is more stable):")
    for param, values in param_variations.items():
        if values and len(values) > 1:
            cv = np.std(values) / \
                np.mean(values) if np.mean(values) != 0 else np.inf
            print(f"  {param}: CV = {cv:.3f} (values: {values})")

    return stability_results


if __name__ == "__main__":
    """
    Run integrated cross-validation examples.
    Uncomment the examples you want to run.
    """

    print("Integrated Cross-Validation Optimization Examples")
    print("=" * 80)
    print("These examples show how to use the CV-integrated optimizer")
    print("for robust parameter selection and strategy validation.")
    print()

    try:
        # Example 1: Basic CV-integrated optimization
        print("Running CV-Integrated Optimization Example...")
        example_integrated_cv_optimization()

        # Example 2: CV methods comparison
        # print("\nRunning CV Methods Comparison Example...")
        # example_cv_methods_comparison()

        # Example 3: Robust workflow (uncomment to run)
        # print("\nRunning Robust Optimization Workflow Example...")
        # example_robust_workflow()

        # Example 4: Custom CV configuration (uncomment to run)
        # print("\nRunning Custom CV Configuration Example...")
        # example_custom_cv_configuration()

        # Example 5: Parameter stability analysis (uncomment to run)
        # print("\nRunning Parameter Stability Analysis Example...")
        # example_parameter_stability_analysis()

        print(f"\n{'='*80}")
        print("ALL INTEGRATED CV EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("Check the results in: data/backtest_results/cv_optimization/")
        print()
        print("Key Benefits of the CV-Integrated Approach:")
        print("✓ Robust parameter selection with time-series awareness")
        print("✓ Statistical significance testing")
        print("✓ Multiple validation methods for comprehensive testing")
        print("✓ Automated workflow with best practices")
        print("✓ Protection against overfitting")
        print("✓ Confidence intervals for performance estimates")
        print()
        print("Next Steps:")
        print("1. Review cross-validation results and parameter stability")
        print("2. Select the most robust parameters")
        print("3. Run final out-of-sample testing")
        print("4. Deploy with confidence!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure QuestDB is running and data is available")
        import traceback
        traceback.print_exc()
