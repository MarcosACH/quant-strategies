"""
CVD BB Pullback Strategy Backtest Script

This script runs a backtest of the CVD Bollinger Band Pullback strategy
using parquet data files from the data directory.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from config.settings import settings
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.bt_engine.vectorbt_engine import VectorBTEngine
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def load_parquet_data(file_path: Path) -> pd.DataFrame:
    """Load and validate parquet data file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_parquet(file_path)

    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [
        col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded data: {data.shape[0]:,} rows, {data.shape[1]} columns")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    return data


def run_backtest(
    data: pd.DataFrame,
    param_ranges: dict,
    initial_cash: float = 1000,
    fee_pct: float = 0.05,
    risk_pct: float = 1.0,
    ticker: str = "BTC-USDT-PERP",
    exchange: str = "okx"
) -> pd.DataFrame:
    """Run the backtest with specified parameters."""

    strategy = CVDBBPullbackStrategy()
    engine = VectorBTEngine(
        initial_cash=initial_cash,
        fee_pct=fee_pct,
        frequency='1min'
    )

    print(f"Starting backtest...")

    start_date = data.index.min().strftime("%Y%m%d")
    end_date = data.index.max().strftime("%Y%m%d")
    date_range = f"{start_date}_{end_date}"

    results = engine.simulate_portfolios(
        strategy=strategy,
        data=data,
        param_dict=param_ranges,
        ticker=ticker,
        sizing_method="Risk percent",
        risk_pct=risk_pct,
        exchange_broker=exchange,
        date_range=date_range,
        save_results=False,
        indicator_batch_size=50
    )

    return results


def analyze_results(results: pd.DataFrame) -> None:
    """Analyze and display backtest results."""
    print(f"\nBACKTEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total parameter combinations tested: {len(results):,}")

    print(f"\nTOP 5 STRATEGIES BY SHARPE RATIO:")
    top_sharpe = results.nlargest(5, 'sharpe_ratio')
    for i, (idx, row) in enumerate(top_sharpe.iterrows(), 1):
        print(f"{i}. Sharpe: {row['sharpe_ratio']:.3f} | "
              f"Return: {row['total_return_pct']:.1f}% | "
              f"DD: {row['max_drawdown_pct']:.1f}% | "
              f"Trades: {row['total_trades']:.0f} | "
              f"WR: {row['win_rate_pct']:.1f}%")

    print(f"\nTOP 5 STRATEGIES BY TOTAL RETURN:")
    top_return = results.nlargest(5, 'total_return_pct')
    for i, (idx, row) in enumerate(top_return.iterrows(), 1):
        print(f"{i}. Return: {row['total_return_pct']:.1f}% | "
              f"Sharpe: {row['sharpe_ratio']:.3f} | "
              f"DD: {row['max_drawdown_pct']:.1f}% | "
              f"Trades: {row['total_trades']:.0f}")

    print(f"\nPERFORMANCE STATISTICS:")
    print(f"Average Return: {results['total_return_pct'].mean():.1f}%")
    print(f"Average Sharpe: {results['sharpe_ratio'].mean():.3f}")
    print(f"Average Max Drawdown: {results['max_drawdown_pct'].mean():.1f}%")
    print(f"Average Win Rate: {results['win_rate_pct'].mean():.1f}%")


def main():
    """Main function to run the backtest."""
    start_time = time.time()
    data_path = settings.DATA_ROOT_PATH / "processed" / \
        "features" / "okx_btc_usdt_perp_1m_2022_10_27-2022_12_27.parquet"

    try:
        data = load_parquet_data(data_path)

        strategy = CVDBBPullbackStrategy()
        # param_ranges = strategy.param_ranges
        param_ranges_small = {
            "bbands_length": np.arange(25, 160, 10),
            "bbands_stddev": np.arange(2.0, 6.0, 0.5),
            "cvd_length": [40],  # np.arange(35, 60, 5),
            "atr_length": [10],  # np.arange(5, 25, 5),
            "sl_coef": [2.0],  # np.arange(2.0, 3.5, 0.5),
            "tpsl_ratio": [2.5],  # np.arange(3.0, 5.5, 0.5)
        }
        print("Running FULL OPTIMIZATION")

        total_combinations = np.prod(
            [len(v) for v in param_ranges_small.values()])
        print(f"Total parameter combinations: {total_combinations:,}")

        results = run_backtest(
            data=data,
            param_ranges=param_ranges_small,
            initial_cash=1000,
            fee_pct=0.05,
            risk_pct=1.0
        )

        analyze_results(results)

        print(f"\nBacktest completed successfully!")
        print(f"Results saved to: results/backtests/")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"Total calculation time: {duration:.2f} seconds ({time.strftime("%H:%M:%S", time.gmtime(duration))})"
    )


if __name__ == "__main__":
    main()
