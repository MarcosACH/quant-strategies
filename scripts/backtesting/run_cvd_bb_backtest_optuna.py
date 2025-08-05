import optuna
import polars as pl
import numpy as np
import vectorbt as vbt
import sys
from pathlib import Path
from functools import partial

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


def objective(
    trial,
    data,
    indicator,
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

    return results["max_drawdown_pct"], results["sharpe_ratio"]


def run_optimization_with_sampler(
    data,
    indicator,
    order_func_nb,
    fee_decimal,
    sizing_method,
    risk_pct,
    initial_cash,
    frequency,
    cash_sharing,
    use_numba,
    stat_names,
    sampler_name,
    n_trials=100,
    param_grid=None,
    plot_param_importances=False,
    plot_pareto_front=False,
    storage_url=None
):
    if sampler_name not in ["tpe", "grid", "random", "cmaes", "gp", "nsgaii", "qmc"]:
        raise ValueError(
            f"Unsupported sampler: {sampler_name}. Supported samplers are: {['tpe', 'grid', 'random', 'cmaes', 'gp', 'nsgaii', 'qmc']}")

    if sampler_name == "grid" and param_grid is None:
        raise ValueError("param_grid must be provided for grid sampler")

    if sampler_name == "cmaes":
        print("WARNING: If the study is being used for multi-objective optimization, CmaEsSampler cannot be used.")

    samplers = {
        "tpe": optuna.samplers.TPESampler(multivariate=True, seed=42),
        "grid": optuna.samplers.GridSampler(param_grid) if sampler_name == "grid" else None,
        "random": optuna.samplers.RandomSampler(seed=42),
        "cmaes": optuna.samplers.CmaEsSampler(seed=42),
        "gp": optuna.samplers.GPSampler(seed=42),
        "nsgaii": optuna.samplers.NSGAIISampler(population_size=50, seed=42),
        "qmc": optuna.samplers.QMCSampler(qmc_type="sobol", seed=42)
    }

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=samplers[sampler_name],
        study_name=f"trading_strategy_{sampler_name}",
        storage=storage_url,
        load_if_exists=True  # This allows resuming studies
    )

    bound_objective = partial(
        objective,
        data=data,
        indicator=indicator,
        order_func_nb=order_func_nb,
        fee_decimal=fee_decimal,
        sizing_method=sizing_method,
        risk_pct=risk_pct,
        initial_cash=initial_cash,
        frequency=frequency,
        cash_sharing=cash_sharing,
        use_numba=use_numba,
        stat_names=stat_names
    )

    study.optimize(bound_objective, n_trials=n_trials, n_jobs=-1)

    if plot_param_importances or plot_pareto_front:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        if plot_param_importances and plot_pareto_front:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Parameter Importances', 'Pareto Front'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )

            param_fig = optuna.visualization.plot_param_importances(study)
            for trace in param_fig.data:
                fig.add_trace(trace, row=1, col=1)

            pareto_fig = optuna.visualization.plot_pareto_front(study)
            for trace in pareto_fig.data:
                fig.add_trace(trace, row=1, col=2)

            fig.update_layout(
                title_text=f"Optimization Results - {sampler_name.upper()} Sampler",
                showlegend=False,
                height=1000,
                width=1800
            )

            fig.update_xaxes(title_text="Importance", row=1, col=1)
            fig.update_xaxes(title_text="Max Drawdown [%]", row=1, col=2)
            fig.update_yaxes(title_text="Parameters", row=1, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)

            fig.show()
        else:
            if plot_param_importances:
                optuna.visualization.plot_param_importances(study).show()
            if plot_pareto_front:
                optuna.visualization.plot_pareto_front(study).show()

    df = study.trials_dataframe()
    pl_df = pl.from_pandas(df)

    return study, pl_df


if __name__ == "__main__":
    indicator = CVDBBPullbackStrategy().create_indicator()
    order_func_nb = CVDBBPullbackStrategy().get_order_func_nb()

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

    # Create SQLite database for Optuna Dashboard
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "results" / "optuna_studies.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path}"
    
    print(f"Optuna studies will be saved to: {db_path}")
    print(f"To view in Optuna Dashboard, use storage URL: {storage_url}")

    strategies = ["gp"]

    results = {}
    for strategy in strategies:
        print(f"\nRunning optimization with {strategy.upper()} sampler...")
        study, pl_df = run_optimization_with_sampler(
            data=prepared_datasets["train"],
            indicator=indicator,
            order_func_nb=order_func_nb,
            fee_decimal=0.0005,
            sizing_method="Risk percent",
            risk_pct=1.0,
            initial_cash=1000,
            frequency="1h",
            cash_sharing=True,
            use_numba=True,
            stat_names=["Max Drawdown [%]", "Sharpe Ratio"],
            sampler_name=strategy,
            n_trials=20,
            plot_param_importances=True,
            plot_pareto_front=True,
            storage_url=storage_url
        )
        results[strategy] = {
            "study": study,
            "dataframe": pl_df,
            "pareto_trials": study.best_trials
        }

        print(f"\n=== Pareto-optimal solutions for {strategy.upper()} ===")
        for i, trial in enumerate(study.best_trials):
            max_dd, sharpe = trial.values
            print(f"Solution {i+1}:")
            print(f"  Max Drawdown: {max_dd:.2f}%")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print(f"  Parameters: {trial.params}")
            print()

    print(f"\n{'='*60}")
    print("OPTUNA DASHBOARD INSTRUCTIONS:")
    print(f"{'='*60}")
    print(f"1. Database location: {db_path}")
    print(f"2. Storage URL: {storage_url}")
    print("3. To view in VS Code Optuna Dashboard:")
    print("   - Open Command Palette (Ctrl+Shift+P)")
    print("   - Type 'Optuna Dashboard: Open'")
    print("   - Enter the storage URL when prompted")
    print("4. Or start dashboard manually:")
    print(f"   optuna-dashboard {storage_url}")
    print(f"{'='*60}")
