import optuna
import polars as pl
import numpy as np
import vectorbt as vbt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.strategies.implementations.cvd_bb_pullback import CVDBBPullbackStrategy
    from src.data.config import DataConfig, DataSplitConfig
    from src.data.pipeline import DataPreparationPipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root or the modules exist.")
    sys.exit(1)


def simulate_portfolio(
    data,
    indicator,
    params,
    order_func_nb,
    fee_decimal,
    sizing_method,
    risk_pct,
    initial_cash,
    frequency,
    cash_sharing,
    use_numba,
    stat_names
):

    open_prices = data["open"].to_numpy()
    high_prices = data["high"].to_numpy()
    low_prices = data["low"].to_numpy()
    close_prices = data["close"].to_numpy()
    volume = data["volume"].to_numpy()

    ind = indicator.run(
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        volume,
        **params
    )

    exits_state = np.dtype([
        ("active_tp_price", np.float64),
        ("active_sl_price", np.float64)
    ])

    rep_eval = vbt.RepEval(
        "np.full(wrapper.shape_2d[1], dtype=exits_state, fill_value=False)",
        mapping=dict(exits_state=exits_state, np=np)
    )

    le = ind.long_entries.to_numpy(dtype=np.bool_)
    se = ind.short_entries.to_numpy(dtype=np.bool_)
    ltp = ind.long_tp_price.to_numpy(dtype=np.float64)
    lsl = ind.long_sl_price.to_numpy(dtype=np.float64)
    stp = ind.short_tp_price.to_numpy(dtype=np.float64)
    ssl = ind.short_sl_price.to_numpy(dtype=np.float64)

    pf = vbt.Portfolio.from_order_func(
        close_prices,
        order_func_nb,
        rep_eval,
        le, se, ltp, lsl, stp, ssl,
        high_prices, low_prices, close_prices,
        fee_decimal,
        sizing_method,
        risk_pct,
        init_cash=initial_cash,
        cash_sharing=cash_sharing,
        freq=frequency,
        use_numba=use_numba
    )

    param_dict = {}
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

    return param_dict


def objective(trial, data, indicator, order_func_nb, fee_decimal, sizing_method, risk_pct, initial_cash, frequency, cash_sharing, use_numba, stat_names):
    bbands_length = trial.suggest_int("bbands_length", 25, 45, step=10)
    bbands_stddev = trial.suggest_float("bbands_stddev", 2.0, 3.0, step=0.5)
    cvd_length = trial.suggest_int("cvd_length", 35, 45, step=5)
    atr_length = trial.suggest_int("atr_length", 5, 15, step=5)
    sl_coef = trial.suggest_float("sl_coef", 2.0, 3.0, step=0.5)
    tpsl_ratio = trial.suggest_float("tpsl_ratio", 3.0, 4.0, step=0.5)

    param_ranges = {
        "bbands_length": bbands_length,
        "bbands_stddev": bbands_stddev,
        "cvd_length": cvd_length,
        "atr_length": atr_length,
        "sl_coef": sl_coef,
        "tpsl_ratio": tpsl_ratio
    }

    results = simulate_portfolio(data, indicator, param_ranges, order_func_nb, fee_decimal,
                                 sizing_method, risk_pct, initial_cash, frequency, cash_sharing, use_numba, stat_names)
    return -results["max_drawdown_pct"], results["sharpe_ratio"]


if __name__ == "__main__":
    indicator = CVDBBPullbackStrategy().create_indicator()
    order_func_nb = CVDBBPullbackStrategy().get_order_func_nb()
    params = {
        "bbands_length": 50,  # []
        "bbands_stddev": 2.0,
        "cvd_length": 35,
        "atr_length": 5,
        "sl_coef": 2.0,
        "tpsl_ratio": 3.0
    }

    split_config = DataSplitConfig(
        train_pct=100,
        validation_pct=None,
        test_pct=None,
        purge_days=1
    )

    data_config = DataConfig(
        symbol="BTC-USDT-SWAP",
        exchange="OKX",
        split_config=split_config,
        config_name="cvd_bb_pullback_optuna_test",
        description="Test configuration for CVD BB Pullback strategy with Optuna",
    )

    pipeline = DataPreparationPipeline(data_config)

    prepared_datasets = pipeline.prepare_data(save_to_disk=False)

    # results = simulate_portfolio(
    #     data=prepared_datasets["train"],
    #     indicator=indicator,
    #     params=params,
    #     order_func_nb=order_func_nb,
    #     fee_decimal=0.0005,
    #     sizing_method="Risk percent",
    #     risk_pct=1.0,
    #     initial_cash=1000,
    #     frequency="1h",
    #     cash_sharing=True,
    #     use_numba=True,
    #     stat_names=["Sharpe Ratio", "Max Drawdown [%]", "Total Return [%]"]
    # )

    # print("Backtest Results:", results)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(
        lambda trial: objective(
            trial, prepared_datasets["train"], indicator, order_func_nb, 0.0005, "Risk percent", 1.0, 1000, "1h", True, True, [
                "Max Drawdown [%]", "Sharpe Ratio"]
        ),
        n_trials=100,
        n_jobs=-1
    )
    # study.optimize(objective, n_trials=100, n_jobs=-1)

    df = study.trials_dataframe()
    pl_df = pl.from_pandas(df)
    print(pl_df)
