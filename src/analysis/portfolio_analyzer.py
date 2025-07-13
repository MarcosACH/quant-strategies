import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Any
import vectorbt as vbt
from config.settings import settings


class PortfolioAnalyzer:
    """
    Portfolio analysis and visualization utilities.

    This class provides methods for analyzing backtest results,
    generating reports, and creating visualizations.
    """

    def __init__(self):
        self.results_path = settings.RESULTS_ROOT_PATH

    def print_results(
        self,
        results: pd.DataFrame,
        result_nr: Optional[int] = None,
        columns: List[str] = None
    ) -> None:
        """
        Print backtest results in a formatted way.

        Args:
            results: DataFrame with backtest results
            result_nr: Specific result number to print (1-indexed)
            columns: Specific columns to display
        """
        if columns:
            data = results.loc[:, columns]
        else:
            data = results

        if result_nr is not None:
            if result_nr < 1 or result_nr > len(data):
                raise IndexError(
                    f"result_nr {result_nr} is out of range. Valid range is 1 to {len(data)}."
                )
            print(data.iloc[result_nr - 1])
        else:
            print(data)

    def print_positions(
        self,
        portfolios: Any,
        portfolio_nr: int,
        save_locally: bool = False,
        exchange_broker: str = None,
        ticker: str = None,
        timeframe: str = None,
        strat_name: str = None,
        date_range: str = None
    ) -> None:
        """
        Print portfolio positions with optional saving.

        Args:
            portfolios: Portfolio object or list of portfolios
            portfolio_nr: Portfolio number to analyze
            save_locally: Whether to save to CSV
            exchange_broker: Exchange/broker identifier
            ticker: Asset ticker
            timeframe: Data timeframe
            strat_name: Strategy name
            date_range: Date range identifier
        """
        positions = portfolios[portfolio_nr].get_positions()
        positions_df = positions.records_readable
        positions_df["Return"] = positions_df["Return"] * 100
        positions_df = positions_df.rename(columns={"Return": "Return (%)"})
        print(positions_df)

        if save_locally and all([exchange_broker, ticker, timeframe, strat_name, date_range]):
            filepath = (
                self.results_path / "positions" /
                f"{exchange_broker}_{ticker}_{timeframe}_{strat_name}_{date_range}_positions.csv"
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)
            positions_df.to_csv(filepath, index=False, float_format="%.6f")
            print(f"Positions saved to {filepath}")

    def print_orders(
        self,
        portfolios: Any,
        portfolio_nr: int,
        save_locally: bool = False,
        exchange_broker: str = None,
        ticker: str = None,
        timeframe: str = None,
        strat_name: str = None,
        date_range: str = None
    ) -> None:
        """
        Print portfolio orders with optional saving.

        Args:
            portfolios: Portfolio object or list of portfolios
            portfolio_nr: Portfolio number to analyze
            save_locally: Whether to save to CSV
            exchange_broker: Exchange/broker identifier
            ticker: Asset ticker
            timeframe: Data timeframe
            strat_name: Strategy name
            date_range: Date range identifier
        """
        orders = portfolios[portfolio_nr].get_orders()
        orders_df = orders.records_readable
        print(orders_df)

        if save_locally and all([exchange_broker, ticker, timeframe, strat_name, date_range]):
            filepath = (
                self.results_path / "orders" /
                f"{exchange_broker}_{ticker}_{timeframe}_{strat_name}_{date_range}_orders.csv"
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)
            orders_df.to_csv(filepath, index=False, float_format="%.6f")
            print(f"Orders saved to {filepath}")

    def plot_portfolio(
        self,
        data: pd.DataFrame,
        indicator: Any,
        portfolio: vbt.Portfolio,
        strat_name: str,
        use_ohlc: bool = False,
        width: int = 1900,
        height: int = 1600,
        save_plot: bool = False,
        save_path: str = None
    ) -> go.Figure:
        """
        Create comprehensive portfolio visualization.

        Args:
            data: OHLCV data
            indicator: Strategy indicator object
            portfolio: VectorBT portfolio object
            strat_name: Strategy name for titles
            use_ohlc: Whether to show OHLC candlesticks
            width: Plot width
            height: Plot height
            save_plot: Whether to save plot
            save_path: Path to save plot

        Returns:
            Plotly figure object
        """
        bar_index = np.arange(len(data))

        # Extract indicator data
        volume_delta = indicator.volume_delta.values
        atr_values = indicator.atr.values

        # Create subplots
        fig = make_subplots(
            rows=6, cols=1,
            subplot_titles=(
                f"{strat_name.replace('_', ' ').capitalize()} - Price and Trades",
                "Cumulative Volume Delta with Bollinger Bands",
                "Volume Delta Bars",
                "Average True Range (ATR)",
                "Trade PnL",
                "Portfolio Value"
            ),
            vertical_spacing=0.08
        )

        # Add price and trades plot
        portfolio_fig = portfolio.plot_trades(width=width, height=height)
        for trace in portfolio_fig.data:
            trace.showlegend = True
            fig.add_trace(trace, row=1, col=1)

        # Add shapes (trade markers) if they exist
        if portfolio_fig.layout.shapes:
            for shape in portfolio_fig.layout.shapes:
                shape_dict = shape.to_plotly_json()
                shape_dict["xref"] = "x"
                shape_dict["yref"] = "y"
                fig.add_shape(shape_dict, row=1, col=1)

        # Add OHLC if requested
        if use_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=bar_index,
                    open=data.open.values,
                    high=data.high.values,
                    low=data.low.values,
                    close=data.close.values,
                    name="OHLC",
                    showlegend=True
                ),
                row=1, col=1
            )

        # Add CVD and Bollinger Bands
        fig.add_trace(go.Scatter(
            x=bar_index,
            y=indicator.cumulative_volume_delta.values,
            mode="lines",
            name="Cumulative Volume Delta",
            line=dict(color="blue", width=2),
            showlegend=True
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=bar_index,
            y=indicator.upper_bband.values,
            mode="lines",
            name="Upper BB",
            line=dict(color="red", width=1),
            showlegend=True
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=bar_index,
            y=indicator.lower_bband.values,
            mode="lines",
            name="Lower BB",
            line=dict(color="red", width=1),
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.1)",
            showlegend=True
        ), row=2, col=1)

        # Add volume delta bars
        fig.add_trace(go.Bar(
            x=bar_index[volume_delta >= 0],
            y=volume_delta[volume_delta >= 0],
            name="Positive Volume Delta",
            marker_color="green",
            width=0.8,
            showlegend=True
        ), row=3, col=1)

        fig.add_trace(go.Bar(
            x=bar_index[volume_delta < 0],
            y=volume_delta[volume_delta < 0],
            name="Negative Volume Delta",
            marker_color="red",
            width=0.8,
            showlegend=True
        ), row=3, col=1)

        fig.add_hline(y=0, line_dash="dash", line_color="black",
                      line_width=1, row=3, col=1)

        # Add ATR
        fig.add_trace(go.Scatter(
            x=bar_index,
            y=atr_values,
            mode="lines",
            name="ATR",
            line=dict(color="purple", width=2),
            showlegend=True
        ), row=4, col=1)

        # Add trade PnL
        trades_fig = portfolio.plot_trade_pnl(width=width, height=height)
        for trace in trades_fig.data:
            trace.showlegend = True
            if hasattr(trace, "y") and trace.y is not None:
                trace.y = [y * 100 for y in trace.y]  # Convert to percentage
            fig.add_trace(trace, row=5, col=1)

        # Add portfolio value
        value_fig = portfolio.plot_value(width=width, height=height)
        for trace in value_fig.data:
            trace.showlegend = True
            fig.add_trace(trace, row=6, col=1)

        # Update layout
        fig.update_layout(
            title=f"{strat_name.replace('_', ' ').capitalize()} Results",
            width=width,
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            dragmode="pan"
        )

        # Update axes
        fig.update_xaxes(title_text="Bar Number", row=6, col=1)
        fig.update_xaxes(fixedrange=False)

        # Link x-axes
        for row in range(1, 7):
            fig.update_xaxes(matches="x", row=row, col=1)

        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1, fixedrange=False)
        fig.update_yaxes(title_text="Cumulative Volume Delta",
                         row=2, col=1, fixedrange=False)
        fig.update_yaxes(title_text="Volume Delta",
                         row=3, col=1, fixedrange=False)
        fig.update_yaxes(title_text="ATR Value", row=4,
                         col=1, fixedrange=False)
        fig.update_yaxes(title_text="Trade PnL (%)",
                         row=5, col=1, fixedrange=False)
        fig.update_yaxes(title_text="Value ($)", row=6,
                         col=1, fixedrange=False)

        # Save plot if requested
        if save_plot:
            if save_path is None:
                save_path = self.results_path / "plots" / \
                    f"{strat_name}_analysis.html"

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            print(f"Plot saved to {save_path}")

        return fig

    def generate_performance_report(
        self,
        results: pd.DataFrame,
        strategy_name: str,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            results: Backtest results DataFrame
            strategy_name: Name of the strategy
            save_report: Whether to save report to file

        Returns:
            Dictionary containing performance metrics and analysis
        """
        report = {
            "strategy_name": strategy_name,
            "total_combinations": len(results),
            "performance_summary": {},
            "top_performers": {},
            "risk_analysis": {},
            "optimization_insights": {}
        }

        # Performance summary
        if "sharpe_ratio" in results.columns:
            report["performance_summary"]["avg_sharpe"] = results["sharpe_ratio"].mean()
            report["performance_summary"]["best_sharpe"] = results["sharpe_ratio"].max()
            report["performance_summary"]["worst_sharpe"] = results["sharpe_ratio"].min()

        if "total_return_pct" in results.columns:
            report["performance_summary"]["avg_return"] = results["total_return_pct"].mean()
            report["performance_summary"]["best_return"] = results["total_return_pct"].max()
            report["performance_summary"]["worst_return"] = results["total_return_pct"].min()

        if "max_drawdown_pct" in results.columns:
            report["performance_summary"]["avg_drawdown"] = results["max_drawdown_pct"].mean()
            report["performance_summary"]["worst_drawdown"] = results["max_drawdown_pct"].max()

        # Top performers
        if "sharpe_ratio" in results.columns:
            top_sharpe = results.nlargest(5, "sharpe_ratio")
            report["top_performers"]["by_sharpe"] = top_sharpe.to_dict(
                "records")

        if "total_return_pct" in results.columns:
            top_returns = results.nlargest(5, "total_return_pct")
            report["top_performers"]["by_return"] = top_returns.to_dict(
                "records")

        # Risk analysis
        if "max_drawdown_pct" in results.columns and "total_return_pct" in results.columns:
            # Calculate risk-adjusted returns
            results_copy = results.copy()
            results_copy["risk_adjusted_return"] = (
                results_copy["total_return_pct"] /
                (results_copy["max_drawdown_pct"] + 1)
            )
            top_risk_adj = results_copy.nlargest(5, "risk_adjusted_return")
            report["risk_analysis"]["top_risk_adjusted"] = top_risk_adj.to_dict(
                "records")

        # Save report if requested
        if save_report:
            report_path = self.results_path / "reports" / \
                f"{strategy_name}_performance_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"Performance report saved to {report_path}")

        return report

    def compare_strategies(
        self,
        results_dict: Dict[str, pd.DataFrame],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance across multiple strategies.

        Args:
            results_dict: Dictionary mapping strategy names to results DataFrames
            metrics: List of metrics to compare

        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = ["sharpe_ratio", "total_return_pct",
                       "max_drawdown_pct", "win_rate_pct"]

        comparison_data = []

        for strategy_name, results in results_dict.items():
            strategy_summary = {"strategy": strategy_name}

            for metric in metrics:
                if metric in results.columns:
                    strategy_summary[f"{metric}_mean"] = results[metric].mean()
                    strategy_summary[f"{metric}_max"] = results[metric].max()
                    strategy_summary[f"{metric}_std"] = results[metric].std()

            comparison_data.append(strategy_summary)

        return pd.DataFrame(comparison_data)
