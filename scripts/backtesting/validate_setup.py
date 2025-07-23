#!/usr/bin/env python3
"""
Backtesting Setup Validator

This script validates that your environment is properly set up
for running the enhanced backtesting workflow.
"""

import sys
import importlib
from pathlib import Path


def check_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} {description} - {e}")
        return False


def check_file_exists(file_path, description=""):
    """Check if a file exists."""
    path = Path(file_path)
    if path.exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (not found)")
        return False


def check_directory_structure():
    """Check required directory structure."""
    print("DIRECTORY STRUCTURE")
    print("-" * 40)

    required_dirs = [
        "data/processed",
        "data/splits/train",
        "results/backtests",
        "scripts/backtesting",
        "src/strategies/implementations",
        "src/bt_engine"
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
            # Create the directory
            path.mkdir(parents=True, exist_ok=True)
            print(f"  → Created {dir_path}/")

    return all_exist


def check_core_dependencies():
    """Check core dependencies."""
    print("\nCORE DEPENDENCIES")
    print("-" * 40)

    core_modules = [
        ("polars", "- Data processing"),
        ("numpy", "- Numerical computations"),
        ("vectorbt", "- Backtesting engine"),
        ("pathlib", "- File path handling"),
        ("datetime", "- Date/time handling"),
        ("json", "- JSON handling"),
        ("time", "- Time utilities")
    ]

    all_good = True
    for module, desc in core_modules:
        if not check_import(module, desc):
            all_good = False

    return all_good


def check_optimization_dependencies():
    """Check optimization-specific dependencies."""
    print("\nOPTIMIZATION DEPENDENCIES")
    print("-" * 40)

    optimization_modules = [
        ("sklearn.model_selection", "- Random search"),
        ("scipy.stats", "- Statistical distributions"),
        ("skopt", "- Bayesian optimization"),
        ("skopt.space", "- Search space definitions"),
        ("skopt.utils", "- Optimization utilities")
    ]

    all_good = True
    for module, desc in optimization_modules:
        if not check_import(module, desc):
            all_good = False

    return all_good


def check_project_modules():
    """Check project-specific modules."""
    print("\nPROJECT MODULES")
    print("-" * 40)

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

    project_modules = [
        ("config.settings", "- Project settings"),
        ("src.strategies.implementations.cvd_bb_pullback", "- CVD BB strategy"),
        ("src.bt_engine.vectorbt_engine", "- VectorBT engine"),
        ("src.data.pipeline.data_preparation", "- Data preparation"),
        ("src.data.config.data_config", "- Data configuration")
    ]

    all_good = True
    for module, desc in project_modules:
        if not check_import(module, desc):
            all_good = False

    return all_good


def check_data_availability():
    """Check for available data files."""
    print("\nDATA AVAILABILITY")
    print("-" * 40)

    data_locations = [
        ("data/processed", "Processed data files"),
        ("data/raw", "Raw data files"),
        ("data/splits/train", "Training data splits")
    ]

    has_data = False
    for location, description in data_locations:
        path = Path(location)
        if path.exists():
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                print(
                    f"✓ {description}: {len(parquet_files)} parquet files found")
                has_data = True
            else:
                print(f"○ {description}: Directory exists but no parquet files")
        else:
            print(f"✗ {description}: Directory not found")

    if not has_data:
        print("\nNote: No data files found. You may need to:")
        print("1. Run data preparation: python scripts/prepare_data.py")
        print("2. Or place existing parquet files in data/processed/")

    return has_data


def check_script_files():
    """Check that required script files exist."""
    print("\nSCRIPT FILES")
    print("-" * 40)

    script_files = [
        ("scripts/backtesting/run_cvd_bb_backtest.py", "Quick backtest script"),
        ("scripts/backtesting/run_enhanced_cvd_bb_backtest.py",
         "Enhanced backtest script"),
        ("scripts/prepare_data.py", "Data preparation script")
    ]

    all_exist = True
    for file_path, description in script_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist


def main():
    """Run all validation checks."""
    print("BACKTESTING SETUP VALIDATION")
    print("=" * 60)
    print("Checking your environment setup for enhanced backtesting...")

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Dependencies", check_core_dependencies),
        ("Optimization Dependencies", check_optimization_dependencies),
        ("Project Modules", check_project_modules),
        ("Script Files", check_script_files),
        ("Data Availability", check_data_availability)
    ]

    all_passed = True
    results = {}

    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"✗ Error during {check_name}: {e}")
            results[check_name] = False
            all_passed = False

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<8} {check_name}")

    print(
        f"\nOverall Status: {'✓ READY' if all_passed else '✗ SETUP REQUIRED'}")

    if not all_passed:
        print(f"\n{'='*60}")
        print("SETUP RECOMMENDATIONS")
        print(f"{'='*60}")

        if not results.get("Core Dependencies", True):
            print("1. Install core dependencies:")
            print("   pip install polars numpy vectorbt")

        if not results.get("Optimization Dependencies", True):
            print("2. Install optimization dependencies:")
            print("   pip install scikit-learn scipy scikit-optimize")
            print("   Or: pip install -r requirements.txt")

        if not results.get("Data Availability", True):
            print("3. Prepare data:")
            print(
                "   python scripts/prepare_data.py --symbol BTC-USDT-SWAP --start 2022-01-01 --end 2022-12-31")

        if not results.get("Project Modules", True):
            print(
                "4. Check project structure and ensure you're running from the project root")

    else:
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("Your environment is ready! You can now:")
        print("1. Run quick test: python scripts/backtesting/run_cvd_bb_backtest.py")
        print("2. Run enhanced optimization: python scripts/backtesting/run_enhanced_cvd_bb_backtest.py")
        print("3. Follow the complete workflow: python scripts/backtesting/workflow_example.py")


if __name__ == "__main__":
    main()
