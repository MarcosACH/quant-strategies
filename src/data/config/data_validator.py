"""
Data Validator Module

This module provides validation utilities for data quality assessment
and continuity checking in the quantitative strategy development workflow.
"""

from typing import Dict, Any
import polars as pl
from polars import col

from src.data.config.data_config import DataConfig


class DataValidator:
    """Data quality validator for market data."""

    def __init__(self, config: DataConfig):
        """
        Initialize validator with data configuration.

        Args:
            config: DataConfig instance with validation parameters
        """
        self.config = config

    def validate_data_quality(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.

        Args:
            df: Polars DataFrame with market data

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "data_range": {
                "start": df["timestamp"].min() if len(df) > 0 else None,
                "end": df["timestamp"].max() if len(df) > 0 else None
            }
        }

        if len(df) < self.config.min_data_points:
            validation_results["is_valid"] = False
            validation_results["issues"].append(
                f"Insufficient data points: {len(df)} < {self.config.min_data_points}"
            )

        if len(df) == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("No data returned")
            return validation_results

        required_columns = ["timestamp", "open",
                            "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["issues"].append(
                f"Missing columns: {missing_columns}")

        null_counts = df.null_count()
        has_nulls = any(null_counts.row(0))

        if has_nulls:
            null_info = {col: count for col, count in zip(
                df.columns, null_counts.row(0)) if count > 0}
            validation_results["warnings"].append(
                f"Null values found: {null_info}")

        duplicate_count = len(df) - df.unique(subset=["timestamp"]).height
        if duplicate_count > 0:
            validation_results["warnings"].append(
                f"Duplicate timestamps: {duplicate_count}")

        continuity_results = self._check_continuity(df)
        validation_results.update(continuity_results)

        outlier_results = self._check_outliers(df)
        validation_results.update(outlier_results)

        ohlc_results = self._check_ohlc_consistency(df)
        validation_results.update(ohlc_results)

        return validation_results

    def _check_continuity(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Check data continuity and identify gaps."""
        if len(df) < 2:
            return {"continuity_check": "insufficient_data"}

        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }

        expected_interval_ms = timeframe_minutes[self.config.timeframe] * 60 * 1000

        df_with_diff = df.with_columns([
            col("timestamp").diff().alias("time_diff")
        ])

        df_with_diff = df_with_diff.with_columns([
            col("time_diff").dt.total_milliseconds().alias("diff_ms")
        ])

        gaps = df_with_diff.filter(
            (col("diff_ms") > expected_interval_ms) &
            (col("diff_ms").is_not_null())
        )

        large_gaps = gaps.filter(
            col("diff_ms") > self.config.max_gap_minutes * 60 * 1000
        )

        return {
            "continuity_check": {
                "total_gaps": len(gaps),
                "large_gaps": len(large_gaps),
                "expected_interval_ms": expected_interval_ms,
                "gap_details": [
                    {
                        "timestamp": row["timestamp"],
                        "gap_minutes": row["diff_ms"] / (60 * 1000),
                        "gap_multiple": row["diff_ms"] / expected_interval_ms
                    }
                    for row in large_gaps.to_dicts()
                ]
            }
        }

    def _check_outliers(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Check for outliers in price data."""
        if len(df) < 10:
            return {"outlier_check": "insufficient_data"}

        df_returns = df.with_columns([
            (col("close") / col("close").shift(1) - 1).alias("returns")
        ])

        returns_stats = df_returns.select([
            col("returns").mean().alias("mean"),
            col("returns").std().alias("std"),
            col("returns").quantile(0.01).alias("q01"),
            col("returns").quantile(0.99).alias("q99")
        ])

        stats = returns_stats.to_dicts()[0]

        outliers = df_returns.filter(
            (col("returns").abs() > stats["std"] * self.config.outlier_std_threshold) &
            (col("returns").is_not_null())
        )

        return {
            "outlier_check": {
                "total_outliers": len(outliers),
                "outlier_threshold": self.config.outlier_std_threshold,
                "returns_stats": stats,
                "outlier_dates": [
                    {
                        "timestamp": row["timestamp"],
                        "return": row["returns"],
                        "close_price": row["close"]
                    }
                    for row in outliers.to_dicts()
                ]
            }
        }

    def _check_ohlc_consistency(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Check OHLC price consistency."""
        if len(df) == 0:
            return {"ohlc_check": "no_data"}

        invalid_ohlc = df.filter(
            (col("high") < col("low")) |
            (col("high") < col("open")) |
            (col("high") < col("close")) |
            (col("low") > col("open")) |
            (col("low") > col("close"))
        )

        invalid_prices = df.filter(
            (col("open") <= 0) |
            (col("high") <= 0) |
            (col("low") <= 0) |
            (col("close") <= 0) |
            (col("volume") < 0)
        )

        return {
            "ohlc_check": {
                "invalid_ohlc_count": len(invalid_ohlc),
                "invalid_price_count": len(invalid_prices),
                "total_records": len(df),
                "invalid_ohlc_dates": [
                    row["timestamp"] for row in invalid_ohlc.to_dicts()
                ][:10]  # Limit to first 10 for brevity
            }
        }

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean data based on configuration settings.

        Args:
            df: Raw market data DataFrame

        Returns:
            Cleaned DataFrame
        """
        if not self.config.apply_data_cleaning:
            return df

        cleaned_df = df.clone()

        cleaned_df = cleaned_df.unique(subset=["timestamp"])

        cleaned_df = cleaned_df.sort("timestamp")

        cleaned_df = cleaned_df.filter(
            (col("high") >= col("low")) &
            (col("high") >= col("open")) &
            (col("high") >= col("close")) &
            (col("low") <= col("open")) &
            (col("low") <= col("close")) &
            (col("open") > 0) &
            (col("high") > 0) &
            (col("low") > 0) &
            (col("close") > 0) &
            (col("volume") >= 0)
        )

        if self.config.fill_missing_method == "forward":
            cleaned_df = cleaned_df.fill_null(strategy="forward")
        elif self.config.fill_missing_method == "backward":
            cleaned_df = cleaned_df.fill_null(strategy="backward")
        elif self.config.fill_missing_method == "interpolate":
            # For numeric columns, use interpolation
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col_name in numeric_cols:
                if col_name in cleaned_df.columns:
                    cleaned_df = cleaned_df.with_columns(
                        col(col_name).interpolate().alias(col_name)
                    )
        elif self.config.fill_missing_method == "drop":
            cleaned_df = cleaned_df.drop_nulls()

        return cleaned_df

    def generate_data_report(self, df: pl.DataFrame) -> str:
        """
        Generate a comprehensive data quality report.

        Args:
            df: Market data DataFrame

        Returns:
            Formatted report string
        """
        validation_results = self.validate_data_quality(df)

        report = f"""
DATA QUALITY REPORT
==================
Configuration: {self.config.config_name}
Symbol: {self.config.symbol} ({self.config.exchange})
Timeframe: {self.config.timeframe}
Date Range: {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')}

SUMMARY
-------
Total Records: {validation_results['total_records']:,}
Expected Records: {self.config.get_expected_data_points():,}
Data Coverage: {(validation_results['total_records'] / self.config.get_expected_data_points() * 100):.1f}%
Overall Status: {'✓ VALID' if validation_results['is_valid'] else '✗ INVALID'}

DATA RANGE
----------
Start: {validation_results['data_range']['start']}
End: {validation_results['data_range']['end']}

ISSUES
------
"""

        if validation_results['issues']:
            for issue in validation_results['issues']:
                report += f"• {issue}\n"
        else:
            report += "No critical issues found.\n"

        report += "\nWARNINGS\n--------\n"
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                report += f"• {warning}\n"
        else:
            report += "No warnings.\n"

        if 'continuity_check' in validation_results:
            continuity = validation_results['continuity_check']
            if isinstance(continuity, dict):
                report += f"\nCONTINUITY\n----------\n"
                report += f"Total Gaps: {continuity['total_gaps']}\n"
                report += f"Large Gaps: {continuity['large_gaps']}\n"

                if continuity['gap_details']:
                    report += "\nLarge Gap Details:\n"
                    for gap in continuity['gap_details'][:5]:
                        report += f"  {gap['timestamp']}: {gap['gap_minutes']:.0f} min gap\n"

        if 'outlier_check' in validation_results:
            outlier = validation_results['outlier_check']
            if isinstance(outlier, dict):
                report += f"\nOUTLIERS\n--------\n"
                report += f"Total Outliers: {outlier['total_outliers']}\n"
                report += f"Threshold: {outlier['outlier_threshold']} standard deviations\n"

        if 'ohlc_check' in validation_results:
            ohlc = validation_results['ohlc_check']
            if isinstance(ohlc, dict):
                report += f"\nOHLC CONSISTENCY\n---------------\n"
                report += f"Invalid OHLC: {ohlc['invalid_ohlc_count']}\n"
                report += f"Invalid Prices: {ohlc['invalid_price_count']}\n"

        return report
