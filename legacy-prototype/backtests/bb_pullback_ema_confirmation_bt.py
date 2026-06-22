import gc
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import vectorbt as vbt
from numba import njit
from typing import Optional, List, Dict, Any, Iterator
import plotly.graph_objects as go
from vectorbt.portfolio.nb import order_nb, NoOrder
from vectorbt.portfolio.enums import SizeType, Direction
from joblib import Parallel, delayed
from tqdm import tqdm


@njit
def order_func_nb(c, last_exits_state, long_entries, short_entries, long_tp_price, long_sl_price,
                  short_tp_price, short_sl_price, high_prices, low_prices, close_prices, fee_decimal, position_size_value=np.nan, min_size_value=0.0001, max_size_value=np.inf, size_granularity=0.0001):
    col, i, pos = c.col, c.i, c.position_now
    exits_state = last_exits_state[col]
    high, low, close = high_prices[i], low_prices[i], close_prices[i]
    pos_size = c.cash_now if np.isnan(
        position_size_value) else position_size_value
    min_pos_size = min_size_value / close
    max_pos_size = max_size_value / close

    if pos == 0:
        long_signal = long_entries[i]
        short_signal = short_entries[i] if not long_signal else False

        if long_signal:
            exits_state.active_tp_price = long_tp_price[i]
            exits_state.active_sl_price = long_sl_price[i]

            return order_nb(pos_size, close, SizeType.Value, Direction.LongOnly,
                            fee_decimal, min_size=min_pos_size, max_size=max_pos_size, size_granularity=size_granularity)
        elif short_signal:
            exits_state.active_tp_price = short_tp_price[i]
            exits_state.active_sl_price = short_sl_price[i]

            return order_nb(pos_size, close, SizeType.Value, Direction.ShortOnly,
                            fee_decimal, min_size=min_pos_size, max_size=max_pos_size, size_granularity=size_granularity)
        return NoOrder

    sl_price, tp_price = exits_state.active_sl_price, exits_state.active_tp_price

    if pos > 0:
        if low <= sl_price:
            return order_nb(-np.inf, sl_price, SizeType.Amount, Direction.LongOnly,
                            fee_decimal)
        elif high >= tp_price:
            return order_nb(-np.inf, tp_price, SizeType.Amount, Direction.LongOnly,
                            fee_decimal)
    else:
        if high >= sl_price:
            return order_nb(-np.inf, sl_price, SizeType.Amount, Direction.ShortOnly,
                            fee_decimal)
        elif low <= tp_price:
            return order_nb(-np.inf, tp_price, SizeType.Amount, Direction.ShortOnly,
                            fee_decimal)

    return NoOrder


def generate_param_combinations(param_dict: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]

    for combination in itertools.product(*param_values):
        yield dict(zip(param_names, combination))


def simulate_portfolios(
    data,
    indicator,
    order_func_nb,
    fee_pct,
    initial_cash,
    frequency,
    param_dict,
    ticker,
    strat_name,
    position_size_value=np.nan,
    min_size_value=0.0001,
    max_size_value=np.inf,
    size_granularity=0.0001,
    indicator_batch_size=500,
    cash_sharing=True,
    use_numba=True,
    exchange_broker="okx",
    date_range="undefined_date_range",
    stats_subset: Optional[List[str]] = None
):
    print(
        f"Starting {strat_name.replace("_", " ").capitalize()} Backtesting...")

    fee_decimal = fee_pct / 100.0

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
    for values in param_dict.values():
        total_combinations *= len(values)

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Processing in indicator batches of {indicator_batch_size}")

    filepath = f"results/stats/{exchange_broker}_{ticker}_{frequency}_{strat_name}_{date_range}_stats.csv"
    is_first_write = True
    total_processed = 0

    if Path(filepath).exists():
        print(f"File {filepath} exists and will be overwritten")
    else:
        print(
            f"Creating new file: {filepath}")

    param_generator = generate_param_combinations(param_dict)

    high_prices = data["high"].to_numpy(dtype=np.float64)
    low_prices = data["low"].to_numpy(dtype=np.float64)
    close_prices = data["close"].to_numpy(dtype=np.float64)

    exits_state = np.dtype([
        ("active_tp_price", np.float64),
        ("active_sl_price", np.float64)
    ])

    with tqdm(total=total_combinations, desc="Backtesting Progress") as pbar:
        while True:
            batch_params = []
            for _ in range(indicator_batch_size):
                try:
                    batch_params.append(next(param_generator))
                except StopIteration:
                    break

            if not batch_params:
                break

            batch_results = _simulate_indicator_batch(
                high_prices,
                low_prices,
                close_prices,
                indicator,
                batch_params,
                order_func_nb,
                exits_state,
                fee_decimal,
                position_size_value,
                min_size_value,
                max_size_value,
                size_granularity,
                initial_cash,
                frequency,
                cash_sharing,
                use_numba,
                stat_names
            )

            batch_df = pd.DataFrame(batch_results)

            if is_first_write:
                batch_df.to_csv(filepath, index=False,
                                float_format="%.6f", mode="w")
                is_first_write = False
            else:
                batch_df.to_csv(filepath, index=False,
                                float_format="%.6f", mode="a", header=False)

            total_processed += len(batch_results)
            pbar.update(len(batch_params))

            del batch_params, batch_results, batch_df
            gc.collect()

    print(f"All batches processed. Total combinations: {total_processed}")
    print("Loading final results...")

    final_results = _load_final_csv(filepath, total_processed)

    return final_results


def _simulate_portfolio(
    rep_eval_str,
    order_func_nb,
    exits_state,
    col_data,
    high_prices,
    low_prices,
    close_prices,
    fee_decimal,
    position_size_value,
    min_size_value,
    max_size_value,
    size_granularity,
    initial_cash,
    frequency,
    cash_sharing,
    use_numba,
    stat_names,
    param_values,
    columns
):
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
        fee_decimal,
        position_size_value,
        min_size_value,
        max_size_value,
        size_granularity,
        init_cash=initial_cash,
        cash_sharing=cash_sharing,
        freq=frequency,
        use_numba=use_numba
    )

    del le, se, ltp, lsl, stp, ssl, rep_eval

    param_dict = _extract_params(param_values, columns)
    _calculate_stats(pf, param_dict, stat_names)

    del pf
    gc.collect()

    return param_dict


def _simulate_indicator_batch(
    high_prices,
    low_prices,
    close_prices,
    indicator,
    batch_params,
    order_func_nb,
    exits_state,
    fee_decimal,
    position_size_value,
    min_size_value,
    max_size_value,
    size_granularity,
    initial_cash,
    frequency,
    cash_sharing,
    use_numba,
    stat_names
):
    param_arrays = {
        key: [params[key] for params in batch_params]
        for key in batch_params[0].keys()
    }

    ind = indicator.run(
        high_prices,
        low_prices,
        close_prices,
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

    job_generator = (
        delayed(_simulate_portfolio)(
            rep_eval_str,
            order_func_nb,
            exits_state,
            (
                long_entries_arr[:, col],
                short_entries_arr[:, col],
                long_tp_price_arr[:, col],
                long_sl_price_arr[:, col],
                short_tp_price_arr[:, col],
                short_sl_price_arr[:, col],
            ),
            high_prices,
            low_prices,
            close_prices,
            fee_decimal,
            position_size_value,
            min_size_value,
            max_size_value,
            size_granularity,
            initial_cash,
            frequency,
            cash_sharing,
            use_numba,
            stat_names,
            columns[col],
            columns
        )
        for col in range(num_cols)
    )

    results = Parallel(n_jobs=-1, backend="loky")(job_generator)

    batch_results = results

    del ind
    del long_entries_arr, long_tp_price_arr, long_sl_price_arr
    del short_entries_arr, short_tp_price_arr, short_sl_price_arr
    del job_generator
    del results
    gc.collect()

    return batch_results


def _load_final_csv(filepath, expected_rows: int) -> pd.DataFrame:
    print(f"Loading final CSV with {expected_rows} rows...")

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

        if expected_rows > 100000:
            print("Large dataset detected. Using chunked reading...")
            chunks = []
            chunk_size = 10000
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=dtype_hints):
                chunks.append(chunk)
            final_df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            final_df = pd.read_csv(filepath, dtype=dtype_hints)

    except Exception as e:
        print(
            f"Warning: Could not optimize dtypes ({e}). Loading with default types...")
        final_df = pd.read_csv(filepath)

    print(f"DataFrame loaded successfully with shape: {final_df.shape}")
    return final_df


def _extract_params(param_values, columns) -> Dict[str, Any]:
    if hasattr(param_values, "_asdict"):
        return param_values._asdict()
    elif isinstance(param_values, dict):
        return param_values
    elif isinstance(param_values, tuple) and hasattr(columns, "names"):
        return dict(zip(columns.names, param_values))
    else:
        return {"param": param_values}


def _calculate_stats(pf, param_dict: Dict[str, Any], stat_names: List[str]) -> None:
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


def print_results(results: pd.DataFrame, result_nr=None, columns=[]):
    if columns:
        data = results.loc[:, columns]
    else:
        data = results
    if result_nr is not None:
        if result_nr < 1 or result_nr > len(data):
            raise IndexError(
                f"result_nr {result_nr} is out of range. Valid range is 1 to {len(data)}.")
        print(data.iloc[result_nr - 1])
    else:
        print(data)


def print_positions(portfolios, portfolio_nr, save_locally=False, exchange_broker=None, ticker=None, timeframe=None, strat_name=None, date_range=None):
    positions = portfolios[portfolio_nr].get_positions()
    positions_df = positions.records_readable
    positions_df["Return"] = positions_df["Return"] * 100
    positions_df = positions_df.rename(columns={"Return": "Return (%)"})
    print(positions_df)

    if save_locally:
        filepath = f"results/positions/{exchange_broker}_{ticker}_{timeframe}_{strat_name}_{date_range}_positions.csv"
        positions_df.to_csv(filepath, index=False, float_format="%.6f")
        print(f"Positions saved to {filepath}")


def print_orders(portfolios, portfolio_nr, save_locally=False, exchange_broker=None, ticker=None, timeframe=None, strat_name=None, date_range=None):
    orders = portfolios[portfolio_nr].get_orders()
    orders_df = orders.records_readable
    print(orders_df)

    if save_locally:
        filepath = f"results/orders/{exchange_broker}_{ticker}_{timeframe}_{strat_name}_{date_range}_orders.csv"
        orders_df.to_csv(filepath, index=False, float_format="%.6f")
        print(f"Orders saved to {filepath}")


def plot_portfolio(data, indicator, portfolios, portfolio_nr, strat_name, use_ohlc=False):
    bar_index = np.arange(len(data))
    close_bar = pd.Series(data.close.values, index=bar_index)
    fast_ema_bar = pd.Series(
        indicator.fast_ema.values, index=bar_index)
    slow_ema_bar = pd.Series(
        indicator.slow_ema.values, index=bar_index)

    fig = portfolios[portfolio_nr].plot_trades(width=1900, height=1000)

    if use_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=bar_index,
                open=data.open.values,
                high=data.high.values,
                low=data.low.values,
                close=data.close.values,
                name="OHLC"
            )
        )
    else:
        fig = close_bar.vbt.plot(
            fig=fig, trace_kwargs=dict(
                name="Close Price", line=dict(color="blue"))
        )

    fig = fast_ema_bar.vbt.plot(
        fig=fig, trace_kwargs=dict(
            name="Fast EMA", line=dict(color="orange"))
    )
    fig = slow_ema_bar.vbt.plot(
        fig=fig, trace_kwargs=dict(
            name="Slow EMA", line=dict(color="purple"))
    )

    fig.update_layout(title=f"{strat_name.replace("_", " ").capitalize()} Results",
                      xaxis_title="Bar Number", yaxis_title="Price")
    fig.show()
