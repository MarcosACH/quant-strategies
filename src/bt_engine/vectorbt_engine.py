import gc
import itertools
import numpy as np
import polars as pl
from sklearn.model_selection import ParameterSampler
import vectorbt as vbt
from typing import Optional, List, Dict, Any, Iterator, Callable
from joblib import Parallel, delayed
from tqdm import tqdm
from config.settings import settings


class VectorBTEngine:
    """
    Generic vectorbt backtesting engine for all trading strategies.

    This class provides a unified interface for backtesting any strategy
    that implements the StrategyBase interface.
    """

    def __init__(
        self,
        initial_cash: float = None,
        fee_pct: float = None,
        frequency: str = None,
        cash_sharing: bool = True,
        use_numba: bool = True
    ):
        """
        Initialize the backtesting engine.

        Args:
            initial_cash: Starting cash amount
            fee_pct: Trading fee percentage
            frequency: Data frequency (e.g., '1D', '1H')
            cash_sharing: Whether to share cash across assets
            use_numba: Whether to use numba compilation
        """
        self.initial_cash = initial_cash or settings.INITIAL_CASH
        self.fee_pct = fee_pct or settings.DEFAULT_FEE_PCT
        self.frequency = frequency or settings.DEFAULT_FREQUENCY
        self.cash_sharing = cash_sharing
        self.use_numba = use_numba
        self.fee_decimal = self.fee_pct / 100.0

    def simulate_portfolios(
        self,
        strategy,
        data: pl.DataFrame,
        ticker: str,
        param_dict: Optional[Dict[str, Any]] = None,
        param_combinations: Optional[ParameterSampler] = None,
        sizing_method: str = "Value-based",
        risk_pct: float = np.nan,
        risk_nominal: float = np.nan,
        position_size_value: float = np.nan,
        min_size_value: float = 0.0001,
        max_size_value: float = np.inf,
        size_granularity: float = 0.0001,
        indicator_batch_size: int = 500,
        exchange_broker: str = "generic",
        date_range: str = "undefined_date_range",
        stats_subset: Optional[List[str]] = None,
        save_results: bool = True
    ) -> pl.DataFrame:
        """
        Run portfolio simulation for a strategy with parameter optimization.

        Args:
            strategy: Strategy instance implementing StrategyBase
            data: OHLCV data DataFrame
            param_dict: Dictionary of parameter ranges for optimization
            ticker: Asset ticker symbol
            sizing_method: Position sizing method
            risk_pct: Risk percentage for risk-based sizing
            risk_nominal: Risk nominal amount
            position_size_value: Fixed position size value
            min_size_value: Minimum position size
            max_size_value: Maximum position size
            size_granularity: Position size granularity
            indicator_batch_size: Batch size for indicator processing
            exchange_broker: Exchange/broker identifier
            date_range: Date range identifier
            stats_subset: Subset of statistics to calculate
            save_results: Whether to save results to file

        Returns:
            DataFrame with backtest results
        """
        if not param_dict and not param_combinations:
            raise ValueError(
                "Either param_dict or param_combinations must be provided")
        if param_dict and param_combinations:
            print(
                "Warning: Both param_dict and param_combinations provided. Using param_dict only.")

        print(f"Starting {strategy.name} Backtesting...")

        stat_names = [
            "Period", "End Value", "Total Return [%]", "Benchmark Return [%]",
            "Total Fees Paid", "Max Drawdown [%]", "Max Drawdown Duration",
            "Total Trades", "Win Rate [%]", "Best Trade [%]", "Worst Trade [%]",
            "Avg Winning Trade [%]", "Avg Losing Trade [%]", "Profit Factor",
            "Expectancy", "Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio"
        ]

        if stats_subset:
            stat_names = [name for name in stat_names if name in stats_subset]

        total_combinations = 1
        if param_dict:
            for values in param_dict.values():
                total_combinations *= len(values)
        else:
            total_combinations = len(param_combinations)

        print(f"Total parameter combinations: {total_combinations}")
        print(f"Processing in indicator batches of {indicator_batch_size}")

        results = []
        if save_results:
            filepath = self._get_results_filepath(
                exchange_broker, ticker, self.frequency, strategy.name, date_range
            )
            print(f"Results will be saved to: {filepath}")

        open_prices = data["open"].to_numpy()
        high_prices = data["high"].to_numpy()
        low_prices = data["low"].to_numpy()
        close_prices = data["close"].to_numpy()
        volume = data["volume"].to_numpy()

        indicator = strategy.create_indicator()
        order_func_nb = strategy.get_order_func_nb()

        exits_state = np.dtype([
            ("active_tp_price", np.float64),
            ("active_sl_price", np.float64)
        ])

        param_generator = self._generate_param_combinations(param_dict)
        total_processed = 0

        with tqdm(total=total_combinations, desc="Backtesting Progress") as pbar:
            while True:
                batch_params = []
                if param_dict:
                    for _ in range(indicator_batch_size):
                        try:
                            batch_params.append(next(param_generator))
                        except StopIteration:
                            break
                else:
                    batch_params = param_combinations[total_processed:
                                                      total_processed + indicator_batch_size]

                if not batch_params:
                    break

                batch_results = self._simulate_indicator_batch(
                    open_prices, high_prices, low_prices, close_prices, volume,
                    indicator, batch_params, order_func_nb, exits_state,
                    sizing_method, risk_pct, risk_nominal, position_size_value,
                    min_size_value, max_size_value, size_granularity,
                    stat_names
                )

                results.extend(batch_results)
                total_processed += len(batch_results)
                pbar.update(len(batch_params))

                if save_results:
                    self._save_batch_results(
                        batch_results, filepath, total_processed == len(batch_results))

                del batch_params, batch_results
                gc.collect()

        print(f"All batches processed. Total combinations: {total_processed}")

        final_results = pl.DataFrame(results)

        return final_results

    def _generate_param_combinations(self, param_dict: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Generate parameter combinations for optimization."""
        param_names = list(param_dict.keys())
        param_values = [param_dict[name] for name in param_names]

        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))

    def _simulate_indicator_batch(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        indicator: vbt.IndicatorFactory,
        batch_params: List[Dict],
        order_func_nb: Callable,
        exits_state: np.dtype,
        sizing_method: str,
        risk_pct: float,
        risk_nominal: float,
        position_size_value: float,
        min_size_value: float,
        max_size_value: float,
        size_granularity: float,
        stat_names: List[str]
    ) -> List[Dict]:
        """Process a batch of parameter combinations."""

        param_arrays = {
            key: [params[key] for params in batch_params]
            for key in batch_params[0].keys()
        }

        ind = indicator.run(
            open_prices, high_prices, low_prices, close_prices, volume,
            **param_arrays,
            param_product=False
        )

        long_entries_arr = ind.long_entries.to_numpy(dtype=np.bool_)
        short_entries_arr = ind.short_entries.to_numpy(dtype=np.bool_)
        long_tp_price_arr = ind.long_tp_price.to_numpy(dtype=np.float64)
        long_sl_price_arr = ind.long_sl_price.to_numpy(dtype=np.float64)
        short_tp_price_arr = ind.short_tp_price.to_numpy(dtype=np.float64)
        short_sl_price_arr = ind.short_sl_price.to_numpy(dtype=np.float64)

        num_cols = ind.long_entries.shape[1]
        columns = ind.wrapper.columns

        rep_eval_str = "np.full(wrapper.shape_2d[1], dtype=exits_state, fill_value=False)"

        jobs = [
            delayed(self._simulate_single_portfolio)(
                rep_eval_str, order_func_nb, exits_state,
                (
                    long_entries_arr[:, col],
                    short_entries_arr[:, col],
                    long_tp_price_arr[:, col],
                    long_sl_price_arr[:, col],
                    short_tp_price_arr[:, col],
                    short_sl_price_arr[:, col],
                ),
                high_prices, low_prices, close_prices,
                sizing_method, risk_pct, risk_nominal, position_size_value,
                min_size_value, max_size_value, size_granularity,
                stat_names, columns[col], columns
            )
            for col in range(num_cols)
        ]

        results = Parallel(n_jobs=settings.MAX_PARALLEL_JOBS,
                           backend="loky")(jobs)

        del ind, long_entries_arr, long_tp_price_arr, long_sl_price_arr
        del short_entries_arr, short_tp_price_arr, short_sl_price_arr
        gc.collect()

        return results

    def _simulate_single_portfolio(
        self,
        rep_eval_str: str,
        order_func_nb: Callable,
        exits_state: np.dtype,
        col_data: tuple,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        sizing_method: str,
        risk_pct: float,
        risk_nominal: float,
        position_size_value: float,
        min_size_value: float,
        max_size_value: float,
        size_granularity: float,
        stat_names: List[str],
        param_values: Any,
        columns: Any
    ) -> Dict[str, Any]:
        """Simulate a single portfolio configuration."""

        le, se, ltp, lsl, stp, ssl = col_data

        rep_eval = vbt.RepEval(
            rep_eval_str,
            mapping=dict(exits_state=exits_state, np=np)
        )

        pf = vbt.Portfolio.from_order_func(
            close_prices,
            order_func_nb,
            rep_eval,
            le, se, ltp, lsl, stp, ssl,
            high_prices, low_prices, close_prices,
            self.fee_decimal,
            sizing_method,
            risk_pct,
            risk_nominal,
            position_size_value,
            min_size_value,
            max_size_value,
            size_granularity,
            init_cash=self.initial_cash,
            cash_sharing=self.cash_sharing,
            freq=self.frequency,
            use_numba=self.use_numba
        )

        param_dict = self._extract_params(param_values, columns)

        self._calculate_stats(pf, param_dict, stat_names)

        del le, se, ltp, lsl, stp, ssl, rep_eval, pf
        gc.collect()

        return param_dict

    def _extract_params(self, param_values: Any, columns: Any) -> Dict[str, Any]:
        """Extract parameters from vectorbt column structure."""
        if hasattr(param_values, "_asdict"):
            return param_values._asdict()
        elif isinstance(param_values, dict):
            return param_values
        elif isinstance(param_values, tuple) and hasattr(columns, "names"):
            return dict(zip(columns.names, param_values))
        else:
            return {"param": param_values}

    def _calculate_stats(self, pf: vbt.Portfolio, param_dict: Dict[str, Any], stat_names: List[str]) -> None:
        """Calculate and store portfolio statistics."""
        stats = pf.stats()
        stat_names_set = set(stat_names)

        stat_mapping = {
            "period": "Period",
            "end_value": "End Value",
            "total_return_pct": "Total Return [%]",
            "benchmark_return_pct": "Benchmark Return [%]",
            "total_fees_paid": "Total Fees Paid",
            "max_drawdown_pct": "Max Drawdown [%]",
            "max_drawdown_duration": "Max Drawdown Duration",
            "total_trades": "Total Trades",
            "win_rate_pct": "Win Rate [%]",
            "best_trade_pct": "Best Trade [%]",
            "worst_trade_pct": "Worst Trade [%]",
            "avg_winning_trade_pct": "Avg Winning Trade [%]",
            "avg_losing_trade_pct": "Avg Losing Trade [%]",
            "profit_factor": "Profit Factor",
            "expectancy": "Expectancy",
            "sharpe_ratio": "Sharpe Ratio",
            "calmar_ratio": "Calmar Ratio",
            "omega_ratio": "Omega Ratio",
            "sortino_ratio": "Sortino Ratio"
        }

        for key, stat_name in stat_mapping.items():
            if stat_name in stat_names_set:
                param_dict[key] = stats.get(stat_name, np.nan)

    def _get_results_filepath(
        self,
        exchange_broker: str,
        ticker: str,
        frequency: str,
        strategy_name: str,
        date_range: str
    ) -> str:
        """Generate filepath for results."""
        filename = f"{exchange_broker}_{ticker}_{frequency}_{strategy_name}_{date_range}_stats.csv"
        filepath = settings.RESULTS_ROOT_PATH / "backtests" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return str(filepath)

    def _save_batch_results(self, batch_results: List[Dict], filepath: str, is_first_batch: bool):
        """Save batch results to CSV file."""
        batch_df = pl.DataFrame(batch_results)

        if is_first_batch:
            batch_df.to_csv(filepath, index=False,
                            float_format="%.6f", mode="w")
        else:
            batch_df.to_csv(filepath, index=False,
                            float_format="%.6f", mode="a", header=False)

    def load_results(self, filepath: str) -> pl.DataFrame:
        """Load results from CSV file with optimized dtypes."""
        try:
            dtype_hints = {
                "end_value": "float32",
                "total_return_pct": "float32",
                "total_fees_paid": "float32",
                "max_drawdown_pct": "float32",
                "total_trades": "int32",
                "win_rate_pct": "float32",
                "sharpe_ratio": "float32",
                "profit_factor": "float32",
                "expectancy": "float32",
                "calmar_ratio": "float32",
                "omega_ratio": "float32",
                "sortino_ratio": "float32"
            }

            return pl.read_csv(filepath, dtype=dtype_hints)

        except Exception as e:
            print(
                f"Warning: Could not optimize dtypes ({e}). Loading with default types...")
            return pl.read_csv(filepath)
