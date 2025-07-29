# Time-Series Cross-Validation System

## Overview

The Time-Series Cross-Validation system provides robust validation methods specifically designed for quantitative trading strategies. It implements best practices for financial time-series data to prevent data leakage, ensure temporal consistency, and provide reliable out-of-sample performance estimates.

## Key Features

### 🎯 **Time-Series Aware Validation**

- Maintains temporal order (no random shuffling)
- Implements purging periods to prevent data leakage
- Supports embargo periods for additional safety
- Respects market regime changes

### 📊 **Multiple Validation Methods**

#### 1. **Rolling Window Cross-Validation**

- Fixed-size training and test windows
- Windows roll forward through time
- Best for testing adaptability to recent market conditions
- Recommended for strategies that need to adapt quickly

#### 2. **Expanding Window Cross-Validation**  

- Training window grows over time
- Fixed-size test windows
- Best for testing performance with increasing data
- Recommended for strategies that benefit from more historical data

#### 3. **Blocked Time-Series Cross-Validation**

- Divides data into sequential blocks
- Each block serves as test set with all previous blocks as training
- Best for testing across different time periods
- Recommended for regime-based testing

### 🔬 **Statistical Validation**

- Confidence intervals for performance estimates
- Statistical significance testing (t-tests)
- Multiple testing corrections
- Robust standard error calculations

### ⚙️ **Integration Ready**

- Seamless integration with existing optimization workflow
- Compatible with all parameter selection methods
- Automated result saving and reporting
- Comprehensive logging and monitoring

## Architecture

```
src/optimization/
├── cross_validator.py          # Core CV implementation
├── cv_integration.py          # Integration with existing workflow
└── parameters_selector.py     # Enhanced parameter selection
```

### Core Components

1. **TimeSeriesCrossValidator**: Main CV engine
2. **ValidationMethod**: Enum of available CV methods
3. **TimeSeriesSplitter**: Abstract base for splitting strategies
4. **CVIntegratedOptimizer**: Complete optimization workflow
5. **CVSummary**: Comprehensive results container

## Usage Examples

### Basic Cross-Validation

```python
from src.optimization import TimeSeriesCrossValidator, ValidationMethod

# Set up cross-validator
cv = TimeSeriesCrossValidator(
    strategy=strategy,
    engine=engine,
    validation_method=ValidationMethod.ROLLING_WINDOW,
    n_splits=5,
    train_size_months=6,
    test_size_months=2,
    purge_days=1
)

# Run cross-validation
results = cv.cross_validate(
    data=data,
    param_combinations=param_combinations,
    optimization_metric="sharpe_ratio"
)
```

### Integrated Optimization Workflow

```python
from src.optimization import CVIntegratedOptimizer

# Set up integrated optimizer
optimizer = CVIntegratedOptimizer(
    strategy=strategy,
    query_service=query_service,
    symbol="BTC-USDT-SWAP",
    start_date="2022-01-01",
    end_date="2022-12-31"
)

# Run complete optimization
result = optimizer.run_cv_optimization(
    param_ranges=param_ranges,
    cv_method="rolling_window",
    optimization_metric="sharpe_ratio"
)
```

### Robust Validation Workflow

```python
# Multi-stage validation workflow
workflow_results = optimizer.run_robust_optimization_workflow(
    param_ranges=param_ranges,
    optimization_metric="sharpe_ratio"
)
```

## Best Practices

### 🎯 **Method Selection**

| Use Case | Recommended Method | Reason |
|----------|-------------------|---------|
| High-frequency strategies | Rolling Window | Adapts to recent market conditions |
| Long-term strategies | Expanding Window | Benefits from more historical data |
| Regime-based strategies | Blocked Time-Series | Tests across different periods |
| Initial exploration | Blocked Time-Series | Fast and comprehensive |
| Final validation | Rolling Window | Most realistic deployment scenario |

### 📏 **Parameter Guidelines**

#### Training Window Size

- **Minimum**: 3 months for 1h data
- **Recommended**: 4-6 months for 1h data
- **Rule**: At least 1000 data points

#### Test Window Size

- **Minimum**: 2 weeks for 1h data
- **Recommended**: 1-2 months for 1h data
- **Rule**: At least 500 data points

#### Number of Splits

- **Minimum**: 3 splits
- **Recommended**: 5-7 splits
- **Maximum**: Limited by data availability

#### Purging Period

- **Minimum**: 1 day
- **Recommended**: 1-3 days for intraday strategies
- **Rule**: Enough to prevent signal leakage

### 🔍 **Validation Checklist**

#### Before Running CV

- [ ] Sufficient data available (>6 months)
- [ ] Data quality validated
- [ ] Parameter ranges reasonable
- [ ] CV method appropriate for strategy type

#### During CV

- [ ] Monitor split sizes (train/test)
- [ ] Check for data leakage warnings
- [ ] Validate performance consistency
- [ ] Monitor statistical significance

#### After CV

- [ ] Review confidence intervals
- [ ] Check parameter stability
- [ ] Validate statistical significance
- [ ] Compare across CV methods
- [ ] Run final out-of-sample test

### ⚠️ **Common Pitfalls**

#### Data Leakage

- **Problem**: Using future information in past predictions
- **Solution**: Proper purging and embargo periods

#### Insufficient Data

- **Problem**: Too few data points in train/test sets
- **Solution**: Adjust window sizes or use fewer splits

#### Overfitting to CV

- **Problem**: Optimizing for CV performance only
- **Solution**: Use nested CV or hold-out final test set

#### Ignoring Market Regimes

- **Problem**: CV splits don't capture different market conditions
- **Solution**: Use longer periods or blocked CV

## Performance Interpretation

### Statistical Significance

- **p < 0.01**: Highly significant, strong evidence
- **p < 0.05**: Significant, reasonable evidence  
- **p < 0.10**: Marginal, weak evidence
- **p > 0.10**: Not significant, insufficient evidence

### Confidence Intervals

- **Narrow**: Stable performance, reliable estimate
- **Wide**: Unstable performance, uncertain estimate
- **Overlaps zero**: Strategy may not be profitable

### Cross-Validation Score

- **CV > 0**: Positive expected performance
- **CV >> std**: Consistent positive performance
- **CV < std**: Unreliable performance

## Integration with Existing Workflow

### Phase 1: Initial Screening

```python
# Use blocked CV for fast parameter screening
cv_result = optimizer.run_cv_optimization(
    param_ranges=broad_ranges,
    cv_method="blocked_timeseries",
    n_param_combinations=50
)
```

### Phase 2: Refined Testing

```python
# Use rolling window CV for realistic testing
refined_result = optimizer.run_cv_optimization(
    param_ranges=refined_ranges,
    cv_method="rolling_window",
    n_param_combinations=20
)
```

### Phase 3: Final Validation

```python
# Test best parameters with expanding window
final_result = optimizer.run_cv_optimization(
    param_ranges=best_params_only,
    cv_method="expanding_window",
    run_final_validation=True
)
```

## Output Files

The system automatically saves comprehensive results:

### CV Results

- `cv_optimization_{symbol}_{method}_{timestamp}.json`
- `{config_name}_{method}_summary_{timestamp}.json`
- `{config_name}_{method}_detailed_{timestamp}.json`

### Comparison Results

- `cv_methods_comparison_{symbol}_{timestamp}.json`
- `robust_workflow_{symbol}_{timestamp}.json`

### Result Structure

```json
{
  "validation_method": "rolling_window",
  "best_parameters": {...},
  "cv_score": 1.234,
  "cv_std": 0.156,
  "confidence_interval": [1.078, 1.390],
  "statistical_significance": {
    "is_significant": true,
    "p_value": 0.023,
    "t_statistic": 2.456
  },
  "overall_metrics": {...}
}
```

## Advanced Features

### Parameter Sensitivity Analysis

```python
# Analyze parameter stability across CV methods
stability_results = example_parameter_stability_analysis()
```

### Custom CV Configuration

```python
# Custom configuration for specific needs
result = optimizer.run_cv_optimization(
    cv_method=ValidationMethod.ROLLING_WINDOW,
    n_splits=8,
    train_size_months=2,
    test_size_months=0.5,
    purge_days=3
)
```

### Multi-Method Comparison

```python
# Compare all CV methods on same data
comparison = optimizer.compare_cv_methods(
    param_ranges=param_ranges,
    optimization_metric="sharpe_ratio"
)
```

## Future Enhancements

### Planned Features

- [ ] Monte Carlo cross-validation
- [ ] Nested cross-validation for hyperparameter tuning
- [ ] Walk-forward analysis integration
- [ ] Visualization dashboards
- [ ] Multi-asset cross-validation
- [ ] Online learning adaptation

### Visualization Support

- [ ] CV score distributions
- [ ] Parameter sensitivity plots
- [ ] Performance stability charts
- [ ] Split timeline visualization

## Troubleshooting

### Common Issues

#### "No valid splits generated"

- **Cause**: Insufficient data or too large window sizes
- **Solution**: Reduce window sizes or use more data

#### "CV results not significant"

- **Cause**: Strategy may not have edge or insufficient data
- **Solution**: Review strategy logic or collect more data

#### "High CV standard deviation"

- **Cause**: Unstable strategy performance
- **Solution**: Review parameter sensitivity or strategy robustness

#### "Memory errors during CV"

- **Cause**: Too many parameter combinations or large datasets
- **Solution**: Reduce batch sizes or use parameter sampling

### Performance Tips

- Use parameter sampling for initial exploration
- Implement parallel processing for large parameter spaces
- Cache data loading for multiple CV runs
- Use reduced datasets for development/testing

## Conclusion

The Time-Series Cross-Validation system provides a robust foundation for validating trading strategies with financial time-series data. By following the best practices and using the appropriate validation methods, you can:

- ✅ Prevent overfitting and data leakage
- ✅ Get reliable out-of-sample performance estimates
- ✅ Build confidence in strategy deployment
- ✅ Make statistically informed decisions
- ✅ Integrate seamlessly with existing workflows

The system is designed to scale with your needs, from simple parameter validation to comprehensive multi-stage optimization workflows.
