import numpy as np
import pandas as pd
import vectorbt as vbt
# from strategies.bb_crosses_ema_confirmation import indicator
# from backtests.bb_crosses_ema_confirmation import order_func_nb, simulate_portfolios, print_results, plot_portfolio
# from strategies.bb_pullback_ema_confirmation import indicator
# from backtests.bb_pullback_ema_confirmation import order_func_nb, simulate_portfolios, print_results, plot_portfolio
# from strategies.ema_crosses_st import indicator
# from backtests.ema_crosses_bt import order_func_nb, simulate_portfolios, print_results
from strategies.cumulative_volume_delta_bb_pulback_st import indicator
from backtests.cumulative_volume_delta_bb_pullback_bt import order_func_nb, simulate_portfolios, print_results, plot_portfolio

data = pd.read_parquet(
    "data/processed/okx_btc_usdt_perp_1m_2022_10_27-2022_12_27.parquet")

data = data.resample("5min").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
})

# data.dropna(inplace=True)

if __name__ == "__main__":
    import time
    start_time = time.time()

    # param_dict = {
    #     "fast_ema_length": np.arange(10, 30, 5),
    #     "slow_ema_length": np.arange(30, 50, 5),
    #     "bbands_length": [10],  # np.arange(10, 30, 5),
    #     "bbands_stddev": np.arange(1, 3, 0.5),
    #     "backcandles": [15],
    #     "atr_length": [10],  # np.arange(10, 30, 5),
    #     "sl_coef": [1],  # np.arange(1, 2.5, 0.5),
    #     "tpsl_ratio": [2.5]
    # }

    # results, portfolios = simulate_portfolios(
    #     indicator=indicator,
    #     data=data,
    #     order_func_nb=order_func_nb,
    #     prices_state=prices_state,
    #     fee_pct=0.1,
    #     initial_cash=1000,
    #     frequency="1min",
    #     param_dict=param_dict,
    #     position_size=np.nan,
    #     min_size=10,
    #     indicator_batch_size=500,
    #     save_portfolios=True,
    #     exchange_broker="okx",
    #     filename="bb_pullback_ema_confirmation",
    #     date_range="2022_10_27-2022_10_28"
    # )
    # print_results(results, columns=["total_return_pct"])

    # ind = indicator.run(
    #     data["high"],
    #     data["low"],
    #     data["close"],
    #     fast_ema_length=10,
    #     slow_ema_length=30,
    #     bbands_length=10,
    #     bbands_stddev=1.0,
    #     backcandles=15,
    #     atr_length=10,
    #     sl_coef=1.0,
    #     tpsl_ratio=2.5,
    #     param_product=False
    # )

    # plot_portfolio(data, ind, portfolios, 0)

# --------------------------------------------------------------------------------------------

    # param_dict = {
    #     "ema_length": np.arange(135, 165, 5),
    #     "bbands_length": np.arange(15, 45, 5),
    #     "bbands_stddev": [2.0],  # np.arange(1, 3, 0.5),
    #     "atr_length": [14],  # np.arange(10, 30, 5),
    #     "sl_coef": [2.5],  # np.arange(1, 2.5, 0.5),
    #     "tpsl_ratio": [2.5]
    # }

    # results, portfolios = simulate_portfolios(
    #     indicator=indicator,
    #     data=data,
    #     order_func_nb=order_func_nb,
    #     prices_state=prices_state,
    #     fee_pct=0.1,
    #     initial_cash=1000,
    #     frequency="1min",
    #     param_dict=param_dict,
    #     position_size=np.nan,
    #     min_size=10,
    #     indicator_batch_size=500,
    #     save_portfolios=True,
    #     exchange_broker="okx",
    #     filename="bb_pullback_ema_confirmation",
    #     date_range="2022_10_27-2022_10_28"
    # )
    # print_results(results, columns=["total_return_pct"])

    # ind = indicator.run(
    #     data["high"],
    #     data["low"],
    #     data["close"],
    #     ema_length=135,
    #     bbands_length=15,
    #     bbands_stddev=2.0,
    #     atr_length=14,
    #     sl_coef=2.5,
    #     tpsl_ratio=2.5,
    #     param_product=False
    # )

    # plot_portfolio(data, ind, portfolios, 0)

# ----------------------------------------------------------------------------------------------

    # param_dict = {
    #     "fast_ema_length": np.arange(10, 60, 5),
    #     "slow_ema_length": np.arange(80, 120, 5),
    #     "atr_length": np.arange(10, 40, 5),
    #     "sl_coef": np.arange(1, 4, 0.5),
    #     "tpsl_ratio": np.arange(1.5, 5, 0.5)
    # }

    # results = simulate_portfolios(
    #     data,
    #     indicator,
    #     order_func_nb,
    #     fee_pct=0.1,
    #     initial_cash=1000,
    #     frequency="1min",
    #     param_dict=param_dict,
    #     position_size_value=np.nan,
    #     min_size_value=10,
    #     indicator_batch_size=100,
    #     save_portfolios=False,
    #     exchange_broker="okx",
    #     ticker="btc-usdt-perp",
    #     strat_name="ema_crosses",
    #     date_range="2022_10_27-2022_10_28"
    # )
    # print_results(results, columns=["total_return_pct"])

    # ind = indicator.run(
    #     data["high"],
    #     data["low"],
    #     data["close"],
    #     fast_ema_length=np.arange(10, 50, 5),
    #     slow_ema_length=np.arange(80, 120, 5),
    #     atr_length=[14],  # np.arange(10, 40, 5),
    #     sl_coef=[2.0],  # np.arange(1, 4, 0.5),
    #     tpsl_ratio=[1.5],  # np.arange(1.5, 3, 0.5),
    #     param_product=True
    # )

    # plot_portfolio(data, ind, portfolios, 0)

# -----------------------------------------------------------------------------------

    param_dict = {
        "bbands_length": np.arange(25, 160, 10),
        "bbands_stddev": np.arange(2.0, 6.0, 0.5),
        "cvd_length": np.arange(35, 60, 5),
        "atr_length": [10],  # np.arange(5, 25, 5),
        "sl_coef": np.arange(2.0, 3.5, 0.5),
        "tpsl_ratio": np.arange(3.0, 5.5, 0.5)
    }

    results = simulate_portfolios(
        data,
        indicator,
        order_func_nb,
        fee_pct=0.05,
        initial_cash=1000,
        frequency="5min",
        param_dict=param_dict,
        sizing_method="Risk percent",
        risk_pct=1.0,
        risk_nominal=10.0,
        position_size_value=np.nan,
        min_size_value=10,
        indicator_batch_size=300,
        exchange_broker="okx",
        ticker="btc-usdt-perp",
        strat_name="cumulative_volume_delta_bb_pullback",
        date_range="2022_10_27-2022_12_27"
    )
    print_results(results, columns=["total_return_pct"])

    # ind = indicator.run(
    #     data["open"],
    #     data["high"],
    #     data["low"],
    #     data["close"],
    #     data["volume"],
    #     bbands_length=[15],
    #     bbands_stddev=[1.0],
    #     cvd_length=[50],
    #     atr_length=[14],
    #     sl_coef=[2.0],
    #     tpsl_ratio=[2.0],
    #     param_product=True
    # )

    # exits_state = np.dtype([
    #     ("active_tp_price", np.float64),
    #     ("active_sl_price", np.float64)
    # ])

    # rep_eval = vbt.RepEval(
    #     "np.full(wrapper.shape_2d[1], dtype=exits_state, fill_value=False)",
    #     mapping=dict(exits_state=exits_state, np=np)
    # )

    # portfolio = vbt.Portfolio.from_order_func(
    #     data["close"].to_numpy(dtype=np.float64),
    #     order_func_nb,
    #     rep_eval,
    #     ind.long_entries.to_numpy(dtype=np.bool_),
    #     ind.short_entries.to_numpy(dtype=np.bool_),
    #     ind.long_tp_price.to_numpy(dtype=np.float64),
    #     ind.long_sl_price.to_numpy(dtype=np.float64),
    #     ind.short_tp_price.to_numpy(dtype=np.float64),
    #     ind.short_sl_price.to_numpy(dtype=np.float64),
    #     data["high"].to_numpy(dtype=np.float64),
    #     data["low"].to_numpy(dtype=np.float64),
    #     data["close"].to_numpy(dtype=np.float64),
    #     0.001,
    #     "Risk percent",
    #     0.1,
    #     10.0,
    #     np.nan,
    #     10,
    #     np.inf,
    #     0.0001,
    #     init_cash=1000,
    #     cash_sharing=True,
    #     freq="1min",
    #     use_numba=True
    # )

    # plot_portfolio(
    #     data, ind, portfolio, "cumulative_volume_delta_bb_pullback"
    # )

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"Total calculation time: {duration:.2f} seconds ({time.strftime("%H:%M:%S", time.gmtime(duration))})"
    )
    print("Optimization process finished.")
