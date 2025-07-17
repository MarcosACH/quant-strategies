# Data Configuration & Preparation System

This system provides a comprehensive solution for data selection, preparation, and splitting in the quantitative strategy development workflow.

## Overview

The Data Configuration & Preparation System bridges the gap between raw data collection and strategy development by providing:

1. **Centralized Configuration Management** - Define all data parameters in one place
2. **Automated Data Quality Validation** - Ensure data integrity before use
3. **Proper Data Splitting** - Time-based splits with purging to prevent leakage
4. **Reproducible Workflows** - Save and load configurations for consistent results

## Architecture

```
src/data/
├── config/
│   ├── data_config.py      # Main configuration classes
│   ├── data_validator.py   # Data quality validation
│   └── __init__.py
├── pipeline/
│   ├── data_preparation.py # Complete preparation pipeline
│   └── __init__.py
└── query/
    └── questdb_market_data_query.py  # Your existing QuestDB query service
```

## Key Components

### 1. DataConfig Class

The central configuration class that manages all data parameters:

```python
from src.data.config.data_config import DataConfig, DataSplitConfig

# Create split configuration
split_config = DataSplitConfig(
    train_pct=0.6,
    validation_pct=0.2,
    test_pct=0.2,
    purge_days=1
)

# Create data configuration
config = DataConfig(
    symbol="BTC-USDT-SWAP",
    exchange="OKX",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
    timeframe="1h",
    split_config=split_config,
    config_name="btc_1h_2023_2024"
)
```

### 2. DataValidator Class

Provides comprehensive data quality validation:

```python
from src.data.config.data_validator import DataValidator

validator = DataValidator(config)
validation_results = validator.validate_data_quality(df)
cleaned_data = validator.clean_data(df)
```

### 3. DataPreparationPipeline Class

Orchestrates the complete data preparation workflow:

```python
from src.data.pipeline.data_preparation import DataPreparationPipeline

pipeline = DataPreparationPipeline(config)
prepared_datasets = pipeline.prepare_data(save_to_disk=True)
```

## Usage Methods

### Method 1: Interactive Jupyter Notebook (Recommended)

Use the interactive notebook for data exploration and configuration:

```bash
jupyter notebook notebooks/01_data_exploration/data_configuration_interface.ipynb
```

This notebook provides:

- Interactive parameter selection
- Data preview and quality checks
- Configuration validation and saving
- Step-by-step preparation workflow

### Method 2: Command Line Interface

For automated workflows, use the command-line script:

```bash
python scripts/prepare_data.py --symbol BTC-USDT-SWAP --start 2023-01-01 --end 2024-06-30 --timeframe 1h
```

Available options:

- `--symbol`: Trading symbol (required)
- `--start`: Start date (required)
- `--end`: End date (required)
- `--exchange`: Exchange name (default: OKX)
- `--timeframe`: Data timeframe (default: 1h)
- `--train-pct`: Training set percentage (default: 0.6)
- `--validation-pct`: Validation set percentage (default: 0.2)
- `--test-pct`: Test set percentage (default: 0.2)
- `--config-name`: Configuration name (auto-generated if not provided)
- `--dry-run`: Show configuration without preparing data

### Method 3: Programmatic Usage

For custom workflows, use the classes directly:

```python
from datetime import datetime, timezone
from src.data.config.data_config import DataConfig, DataSplitConfig
from src.data.pipeline.data_preparation import DataPreparationPipeline

# Create configuration
config = DataConfig(
    symbol="BTC-USDT-SWAP",
    exchange="OKX",
    start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
    end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
    timeframe="1h",
    split_config=DataSplitConfig(train_pct=0.6, validation_pct=0.2, test_pct=0.2),
    config_name="my_config"
)

# Prepare data
pipeline = DataPreparationPipeline(config)
datasets = pipeline.prepare_data()

# Access prepared datasets
train_data = datasets["train"]
validation_data = datasets["validation"]
test_data = datasets["test"]
```

## Data Splitting Strategy

The system implements proper time-based data splitting with the following features:

### Time-Based Splits (Not Random)

- Respects temporal order of data
- Prevents look-ahead bias
- Maintains market regime representation

### Purging Between Splits

- Adds gap days between splits to prevent information leakage
- Configurable purge period (default: 1 day)
- Ensures clean separation between train/validation/test

### Flexible Split Ratios

- Default: 60% train, 20% validation, 20% test
- Configurable percentages
- Optional validation set (can be None)

### Example Split Timeline

```
2023-01-01 =====[TRAIN 60%]===== 2023-09-12 [PURGE] 2023-09-13 ===[VAL 20%]=== 2024-01-28 [PURGE] 2024-01-29 ===[TEST 20%]=== 2024-06-30
```

## Data Quality Validation

The system performs comprehensive data quality checks:

### Continuity Checks

- Identifies gaps in time series data
- Validates expected timeframe intervals
- Reports missing data periods

### Outlier Detection

- Statistical outlier identification
- Configurable standard deviation thresholds
- Return-based outlier analysis

### OHLC Consistency

- Validates high >= low, open, close
- Checks for negative or zero prices
- Ensures volume >= 0

### Basic Quality Metrics

- Minimum data point requirements
- Null value detection
- Duplicate timestamp identification

## Configuration Management

### Saving Configurations

```python
# Save to default location
config_path = config.save_config()

# Save to specific location
config_path = config.save_config("/path/to/config.yaml")
```

### Loading Configurations

```python
# Load from file
config = DataConfig.load_config("config.yaml")

# Use with pipeline
pipeline = DataPreparationPipeline(config)
```

### Configuration Files

Configurations are saved in YAML format with all parameters:

```yaml
data_config:
  symbol: "BTC-USDT-SWAP"
  exchange: "OKX"
  start_date: "2023-01-01T00:00:00+00:00"
  end_date: "2024-06-30T00:00:00+00:00"
  timeframe: "1h"
  config_name: "btc_1h_2023_2024"
  description: "BTC-USDT-SWAP 1h data for strategy development"

split_config:
  train_pct: 0.6
  validation_pct: 0.2
  test_pct: 0.2
  purge_days: 1

quality_config:
  min_data_points: 1000
  max_gap_minutes: 120
  outlier_std_threshold: 5.0
  apply_data_cleaning: true
  fill_missing_method: "forward"
```

## Output Structure

Prepared datasets are saved in the following structure:

```
data/processed/{config_name}/
├── config.yaml                                    # Configuration file
├── metadata.json                                  # Dataset metadata
├── {symbol}_{exchange}_train_{timeframe}.parquet  # Training data
├── {symbol}_{exchange}_validation_{timeframe}.parquet  # Validation data
└── {symbol}_{exchange}_test_{timeframe}.parquet   # Test data
```

## Integration with Development Workflow

This system directly supports the development workflow phases:

### Phase 1: Research & Development ✅

- ✅ Data Collection & Preparation
- ✅ Data Quality Validation
- ✅ Data Splitting Strategy

### Phase 2: Strategy Development

Use prepared datasets in strategy development:

```python
# Load prepared data
pipeline = DataPreparationPipeline(config)
train_data = pipeline.load_prepared_dataset("config_name", "train")
validation_data = pipeline.load_prepared_dataset("config_name", "validation")
test_data = pipeline.load_prepared_dataset("config_name", "test")
```

### Phase 3: Validation & Testing

- Test set remains untouched until final validation
- Validation set used for hyperparameter tuning
- Proper temporal separation maintained

## Best Practices

### 1. Configuration Naming

Use descriptive names that include:

- Symbol
- Timeframe
- Date range
- Purpose (if specific)

Example: `btc_usdt_swap_1h_2023_2024_momentum_study`

### 2. Data Quality Checks

Always review validation reports before proceeding:

- Check for data gaps
- Verify expected record counts
- Review outlier analysis

### 3. Split Ratios

Consider your use case:

- **Research**: 60/20/20 (standard)
- **Limited data**: 70/15/15
- **Extensive validation**: 50/25/25

### 4. Purge Periods

Adjust based on strategy frequency:

- **High-frequency**: 1-2 days
- **Daily strategies**: 3-5 days
- **Weekly strategies**: 7-14 days

### 5. Timeframe Selection

Choose appropriate timeframes for your strategy:

- **Scalping**: 1m, 5m
- **Intraday**: 15m, 30m, 1h
- **Swing**: 4h, 1d
- **Position**: 1d, 1w

## Examples

See `examples/data_config_examples.py` for complete examples of:

- BTC configuration for long-term strategies
- ETH configuration with custom splits
- Short-term trading configuration

## Error Handling

The system includes comprehensive error handling:

### Configuration Validation

- Date range validation
- Split percentage validation
- Parameter constraint checking

### Data Quality Errors

- Insufficient data handling
- Missing column detection
- Quality threshold violations

### Pipeline Errors

- QuestDB connection issues
- Data processing errors
- File system errors

## Next Steps

After preparing your data:

1. **Feature Engineering**: Use `notebooks/01_data_exploration/feature_exploration.ipynb`
2. **Strategy Development**: Use `notebooks/02_strategy_development/strategy_prototyping.ipynb`
3. **Parameter Optimization**: Use `notebooks/03_optimization/parameter_optimization.ipynb`

## Support

For issues or questions:

1. Check the validation reports for data quality issues
2. Review configuration parameters
3. Ensure QuestDB connectivity
4. Verify file permissions for data storage

The system is designed to be robust and provide clear error messages to help diagnose issues quickly.
