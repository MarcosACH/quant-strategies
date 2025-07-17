# Quantitative Trading Strategy Development Framework

A comprehensive, production-ready framework for developing, backtesting, and deploying quantitative trading strategies using VectorBT.

## 🏗️ Architecture Overview

This framework follows a 5-phase development workflow with proper separation of concerns:

1. **Research & Development** - Data exploration, feature engineering, initial strategy development
2. **Parameter Optimization** - Grid search, Bayesian optimization with cross-validation
3. **Validation & Robustness Testing** - Out-of-sample validation, walk-forward analysis
4. **Final Testing & Finalization** - Statistical significance testing, transaction cost analysis
5. **Implementation & Monitoring** - Paper trading, live deployment, continuous monitoring

## 📁 Project Structure

```
strategies-development/
├── config/                     # Configuration management
│   ├── settings.py             # Central configuration
│   ├── data_sources.yaml       # Data source configurations
│   ├── validation_rules.yaml   # Validation criteria
│   └── strategy_configs/       # Strategy-specific configs
├── src/                        # Source code
│   ├── strategies/             # Strategy implementations
│   │   ├── base/              # Base classes and utilities
│   │   └── implementations/    # Concrete strategy classes
│   ├── backtesting/           # Backtesting engine
│   ├── data/                  # Data pipeline
│   │   ├── collectors/        # Data collection
│   │   ├── processors/        # Data processing
│   │   └── storage/           # Data persistence
│   ├── features/              # Feature engineering
│   ├── optimization/          # Parameter optimization
│   ├── validation/            # Validation framework
│   ├── analysis/              # Performance analysis
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for research
├── scripts/                   # Automation scripts
├── data/                      # Data storage
├── results/                   # Analysis results
└── reports/                   # Generated reports
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd strategies-development

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and settings
```

### 3. Data Preparation

**New**: Use the integrated data configuration system to prepare your datasets:

```bash
# Interactive notebook (recommended)
jupyter notebook notebooks/01_data_exploration/data_configuration_interface.ipynb

# Command line interface
python scripts/prepare_data.py --symbol BTC-USDT-SWAP --start 2023-01-01 --end 2024-06-30
```

The data configuration system provides:

- **Automated data quality validation**
- **Proper time-based data splitting** (train/validation/test)
- **Purging between splits** to prevent look-ahead bias
- **Configurable parameters** for different strategy types
- **Reproducible workflows** with saved configurations

### 4. Run Your First Backtest

```python
from src.strategies.implementations import CVDBBPullbackStrategy
from src.backtesting import VectorBTEngine
from src.analysis import PortfolioAnalyzer
import pandas as pd
import yfinance as yf

# Load data
data = yf.download("AAPL", start="2022-01-01", end="2024-01-01")

# Initialize strategy
strategy = CVDBBPullbackStrategy()

# Create backtesting engine
engine = VectorBTEngine(initial_cash=10000, fee_pct=0.1)

# Define parameter ranges for optimization
param_ranges = {
    'bbands_length': [20, 30, 40],
    'bbands_stddev': [1.5, 2.0, 2.5],
    'cvd_length': [30, 50, 70]
}

# Run backtest
results = engine.simulate_portfolios(
    strategy=strategy,
    data=data,
    param_dict=param_ranges,
    ticker="AAPL"
)

# Analyze results
analyzer = PortfolioAnalyzer()
analyzer.print_results(results)
```

## 🧬 Strategy Implementation

### Creating a New Strategy

1. **Inherit from StrategyBase**:

```python
from src.strategies.base import StrategyBase

class MyStrategy(StrategyBase):
    @property
    def default_params(self):
        return {'param1': 10, 'param2': 0.5}
    
    @property
    def param_ranges(self):
        return {'param1': [5, 10, 15], 'param2': [0.3, 0.5, 0.7]}
    
    def create_indicator(self):
        # Implement your indicator logic
        pass
    
    def get_order_func_nb(self):
        # Return your numba-compiled order function
        pass
```

2. **Implement Signal Logic**:

```python
@staticmethod
@njit
def _get_signals(data):
    # Your signal generation logic here
    return long_entries, short_entries
```

3. **Create Order Function**:

```python
@njit
def order_func_nb(c, signals, ...):
    # Your order execution logic here
    pass
```

### Example: CVD Bollinger Band Pullback Strategy

The included CVD BB Pullback strategy demonstrates:

- ✅ Volume analysis with Cumulative Volume Delta
- ✅ Mean reversion using Bollinger Bands
- ✅ ATR-based stop losses and take profits
- ✅ Risk-based position sizing
- ✅ Comprehensive parameter optimization

**Strategy Logic**:

- **Long Entry**: CVD crosses above lower Bollinger Band after being below
- **Short Entry**: CVD crosses below upper Bollinger Band after being above
- **Stop Loss**: ATR-based distance from entry
- **Take Profit**: Risk-reward ratio multiple of stop loss

## 📊 Features

### Backtesting Engine

- **VectorBT Integration**: Lightning-fast vectorized backtesting
- **Multiple Position Sizing Methods**: Value-based, risk percentage, risk nominal
- **Transaction Cost Modeling**: Realistic fees, slippage, and market impact
- **Parallel Processing**: Multi-core optimization support

### Risk Management

- **Position Sizing**: Advanced sizing algorithms with risk controls
- **Portfolio-Level Risk**: Correlation and diversification analysis
- **Drawdown Controls**: Maximum drawdown limits and monitoring
- **Dynamic Risk Adjustment**: Volatility-based position sizing

### Optimization & Validation

- **Parameter Optimization**: Grid search and Bayesian optimization
- **Cross-Validation**: Time-series aware validation methods
- **Walk-Forward Analysis**: Realistic out-of-sample testing
- **Robustness Testing**: Parameter sensitivity and Monte Carlo analysis
- **Statistical Significance**: Comprehensive statistical testing

### Data Management

- **Multiple Data Sources**: Yahoo Finance, Alpha Vantage, CCXT, Polygon
- **Data Quality Checks**: Automated validation and cleaning
- **Storage Options**: SQLite, HDF5, Parquet with compression
- **Caching System**: Intelligent data caching for performance

### Analysis & Reporting

- **Performance Metrics**: 20+ risk-adjusted performance measures
- **Interactive Visualizations**: Plotly-based charts and analysis
- **Automated Reports**: PDF and HTML report generation
- **Strategy Comparison**: Multi-strategy performance comparison

## 🎯 Best Practices

### Strategy Development

1. **Start Simple**: Begin with basic logic, add complexity gradually
2. **Avoid Overfitting**: Use proper validation and out-of-sample testing
3. **Document Everything**: Maintain clear documentation of assumptions
4. **Test Thoroughly**: Validate edge cases and error conditions

### Data Management

1. **Quality First**: Always validate data quality before backtesting
2. **Point-in-Time**: Ensure no look-ahead bias in historical data
3. **Survivorship Bias**: Account for delisted assets in historical analysis
4. **Regular Updates**: Implement automated data update pipelines

### Risk Management

1. **Position Sizing**: Never risk more than you can afford to lose
2. **Diversification**: Avoid concentration in single assets or strategies
3. **Stress Testing**: Test strategies under extreme market conditions
4. **Regular Monitoring**: Implement automated monitoring and alerts

## 🔧 Configuration

### Environment Variables (.env)

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key

# Database
DATABASE_URL=sqlite:///data/trading.db

# Trading Settings
INITIAL_CASH=10000
DEFAULT_FEE_PCT=0.1
PAPER_TRADING=true

# Risk Management
MAX_POSITION_SIZE_PCT=10
MAX_PORTFOLIO_RISK_PCT=25
```

### Strategy Configuration (YAML)

```yaml
validation_criteria:
  minimum_sharpe_ratio: 1.0
  maximum_drawdown_pct: 20.0
  minimum_trades: 100

risk_management:
  maximum_leverage: 3.0
  minimum_diversification_ratio: 0.3
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Volatility, VaR, Expected Shortfall, Maximum Drawdown
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega ratios
- **Trade Analysis**: Win rate, profit factor, average trade duration
- **Advanced**: Information ratio, Treynor ratio, Jensen's alpha

## 🧪 Testing & Validation

### Unit Testing

```bash
pytest tests/
```

### Strategy Validation

```bash
python scripts/validation/validate_strategy.py --strategy cvd_bb_pullback
```

### Performance Benchmarking

```bash
python scripts/analysis/benchmark_performance.py
```

## 📚 Documentation

- **API Documentation**: Auto-generated with Sphinx
- **Strategy Guide**: Detailed implementation guidelines
- **Best Practices**: Production deployment recommendations
- **Examples**: Complete working examples for each component

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: <support@activequants.com>

## 🏆 Acknowledgments

- **VectorBT**: For providing an excellent backtesting framework
- **Quantitative Trading Community**: For inspiration and best practices
- **Open Source Contributors**: For making this project possible

---

**⚠️ Disclaimer**: This framework is for educational and research purposes. Trading involves substantial risk of loss. Always conduct thorough testing before deploying strategies with real capital.
