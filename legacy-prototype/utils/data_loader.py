import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loader errors."""
    pass


def data_loader(
    file_path: Union[str, Path],
    datetime_column: str = "datetime",
    set_datetime_as_index: bool = True,
    resample_freq: Optional[str] = None,
    resample_agg: Optional[Dict[str, Union[str, List[str]]]] = None,
    columns_to_load: Optional[List[str]] = None,
    date_range: Optional[Tuple[Union[str, datetime],
                               Union[str, datetime]]] = None,
    normalize_columns: bool = True,
    handle_duplicates: str = "keep_last",
    sort_datetime_descending: bool = True,
    fill_missing: Optional[str] = None,
    validate_data: bool = True,
    drop_zero_volume: bool = False,
    numeric_dtype: Optional[str] = None,
    column_dtypes: Optional[Dict[str, str]] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load market price data from parquet or CSV files into a pandas DataFrame.

    This function provides comprehensive data loading capabilities for financial
    market data with built-in data cleaning, validation, and preprocessing features.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file (parquet or CSV)
    datetime_column : str, default "datetime"
        Name of the datetime column
    set_datetime_as_index : bool, default True
        Whether to set the datetime column as the DataFrame index
    resample_freq : Optional[str], default None
        Frequency for resampling (e.g., "D" for daily, "W" for weekly, "M" for monthly)
        If None, no resampling is performed
    resample_agg : Optional[Dict[str, Union[str, List[str]]]], default None
        Aggregation methods for resampling. If None, uses default aggregations:
        - price columns (open, high, low, close): "ohlc"
        - volume: "sum"
        - other numeric: "mean"
    columns_to_load : Optional[List[str]], default None
        Specific columns to load. If None, loads all columns
    date_range : Optional[Tuple[Union[str, datetime], Union[str, datetime]]], default None
        Tuple of (start_date, end_date) to filter data
    normalize_columns : bool, default True
        Whether to convert column names to lowercase and replace spaces with underscores
    handle_duplicates : str, default "keep_last"
        How to handle duplicate timestamps ("keep_first", "keep_last", "drop")
    sort_datetime_descending : bool, default True
        Whether to sort the data by datetime in descending order or ascending order
    fill_missing : Optional[str], default None
        Method to fill missing values ("forward", "backward", "interpolate")
    validate_data : bool, default True
        Whether to perform data validation checks
    drop_zero_volume : bool, default False
        Whether to drop rows where the volume column is zero (after loading and processing)
    numeric_dtype : Optional[str], default None
        If provided, cast all numeric columns to this dtype (e.g., "float64", "float32", "int64")
    column_dtypes : Optional[Dict[str, str]], default None
        Dictionary mapping column names to dtypes (e.g., {"close": "float64", "volume": "int64"})
    **kwargs : Any
        Additional arguments passed to pandas read functions

    Returns
    -------
    pd.DataFrame
        Processed market data DataFrame

    Raises
    ------
    DataLoaderError
        If file cannot be read or required columns are missing
    ValueError
        If invalid parameters are provided
    FileNotFoundError
        If the specified file does not exist

    Examples
    --------
    >>> # Load daily OHLCV data with basic processing
    >>> df = data_loader("market_data.parquet")

    >>> # Load specific columns with custom date range
    >>> df = data_loader(
    ...     "prices.csv",
    ...     columns_to_load=["datetime", "open", "high", "low", "close"],
    ...     date_range=("2023-01-01", "2023-12-31")
    ... )

    >>> # Resample to weekly data with custom aggregations
    >>> df = data_loader(
    ...     "minute_data.parquet",
    ...     resample_freq="W",
    ...     resample_agg={"price": "ohlc", "volume": "sum"}
    ... )
    """

    _validate_inputs(
        file_path, datetime_column, handle_duplicates, fill_missing, resample_freq
    )

    file_path = Path(file_path)

    try:
        logger.info(f"Loading data from {file_path}")
        df = _load_file(file_path, **kwargs)

        if df.empty:
            raise DataLoaderError("Loaded DataFrame is empty")

        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")

        if normalize_columns:
            df = _normalize_column_names(df)
            if datetime_column in df.columns:
                pass
            else:
                normalized_datetime = datetime_column.lower().replace(" ", "_")
                if normalized_datetime in df.columns:
                    datetime_column = normalized_datetime

        if datetime_column not in df.columns:
            available_cols = list(df.columns)
            raise DataLoaderError(
                f"Datetime column '{datetime_column}' not found. "
                f"Available columns: {available_cols}"
            )

        df = _process_datetime_column(df, datetime_column)

        if date_range is not None:
            df = _filter_date_range(df, datetime_column, date_range)

        if handle_duplicates != "none":
            df = _handle_duplicates(df, datetime_column, handle_duplicates)

        if set_datetime_as_index:
            df = df.set_index(datetime_column)
            datetime_column = df.index.name

        if fill_missing is not None:
            df = _fill_missing_values(df, fill_missing)

        if resample_freq is not None:
            _validate_resample_frequency(
                df, resample_freq, set_datetime_as_index, datetime_column)
            df = _resample_data(df, resample_freq,
                                resample_agg, set_datetime_as_index)

        if drop_zero_volume:
            volume_col = None
            for col in df.columns:
                if str(col).lower() in ["volume", "vol"] or "volume" in str(col).lower() or "vol" in str(col).lower():
                    volume_col = col
                    break
            if volume_col is not None:
                before = len(df)
                df = df[df[volume_col] != 0]
                after = len(df)
                logger.info(f"Dropped {before - after} rows with zero volume.")
            else:
                logger.warning(
                    "drop_zero_volume=True but no volume column found.")

        if sort_datetime_descending:
            df = df.sort_values(datetime_column, ascending=False)
        else:
            df = df.sort_values(datetime_column, ascending=True)

        if column_dtypes is not None:
            for col, dtype in column_dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(
                            f"Could not cast column '{col}' to dtype '{dtype}': {e}")
        if numeric_dtype is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                try:
                    df[col] = df[col].astype(numeric_dtype)
                except Exception as e:
                    logger.warning(
                        f"Could not cast numeric column '{col}' to dtype '{numeric_dtype}': {e}")

        if validate_data:
            _validate_market_data(df)

        logger.info(f"Data processing completed. Final shape: {df.shape}")

        if columns_to_load is not None:
            return df[columns_to_load]
        else:
            return df

    except Exception as e:
        if isinstance(e, (DataLoaderError, ValueError, FileNotFoundError)):
            raise
        else:
            raise DataLoaderError(
                f"Unexpected error loading data: {str(e)}") from e


def _validate_inputs(
    file_path: Union[str, Path],
    datetime_column: str,
    handle_duplicates: str,
    fill_missing: Optional[str],
    resample_freq: Optional[str]
) -> None:
    """Validate input parameters."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if not path.suffix.lower() in [".parquet", ".csv"]:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    valid_duplicate_methods = ["keep_first", "keep_last", "drop", "none"]
    if handle_duplicates not in valid_duplicate_methods:
        raise ValueError(
            f"handle_duplicates must be one of {valid_duplicate_methods}, "
            f"got '{handle_duplicates}'"
        )

    if fill_missing is not None:
        valid_fill_methods = ["forward", "backward", "interpolate"]
        if fill_missing not in valid_fill_methods:
            raise ValueError(
                f"fill_missing must be one of {valid_fill_methods} or None, "
                f"got '{fill_missing}'"
            )

    if resample_freq is not None:
        try:
            pd.date_range("2020-01-01", periods=2, freq=resample_freq)
        except ValueError as e:
            raise ValueError(
                f"Invalid resample frequency '{resample_freq}': {e}")


def _load_file(
    file_path: Path,
    **kwargs: Any
) -> pd.DataFrame:
    """Load data from file based on extension."""

    try:
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return df

    except Exception as e:
        raise DataLoaderError(
            f"Failed to load file {file_path}: {str(e)}") from e


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""

    normalized_columns = {}
    for col in df.columns:
        normalized = str(col).lower().replace(" ", "_").replace("-", "_")
        normalized = "".join(c if c.isalnum() or c ==
                             "_" else "_" for c in normalized)
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        normalized = normalized.strip("_")

        normalized_columns[col] = normalized

    df = df.rename(columns=normalized_columns)
    logger.debug(f"Normalized column names: {list(df.columns)}")

    return df


def _process_datetime_column(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Process and validate datetime column."""

    try:
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df[datetime_column] = pd.to_datetime(
                df[datetime_column])  # format="%d.%m.%Y %H:%M:%S"

        if df[datetime_column].dt.tz is not None:
            df[datetime_column] = df[datetime_column].dt.tz_localize(None)

        return df

    except Exception as e:
        raise DataLoaderError(
            f"Failed to process datetime column '{datetime_column}': {str(e)}"
        ) from e


def _filter_date_range(
    df: pd.DataFrame,
    datetime_column: str,
    date_range: Tuple[Union[str, datetime], Union[str, datetime]]
) -> pd.DataFrame:
    """Filter data by date range."""

    start_date, end_date = date_range

    try:
        start_date = pd.to_datetime(start_date)  # format="%d.%m.%Y %H:%M:%S"
        end_date = pd.to_datetime(end_date)  # format="%d.%m.%Y %H:%M:%S"

        if start_date > end_date:
            raise ValueError("Start date must be before end date")

        mask = (df[datetime_column] >= start_date) & (
            df[datetime_column] <= end_date)
        filtered_df = df[mask].copy()

        logger.info(
            f"Filtered data from {start_date.date()} to {end_date.date()}. "
            f"Rows: {len(df)} -> {len(filtered_df)}"
        )

        return filtered_df

    except Exception as e:
        raise DataLoaderError(f"Failed to filter date range: {str(e)}") from e


def _handle_duplicates(
    df: pd.DataFrame,
    datetime_column: str,
    method: str
) -> pd.DataFrame:
    """Handle duplicate timestamps."""

    duplicates_count = df[datetime_column].duplicated().sum()

    if duplicates_count == 0:
        return df

    logger.warning(f"Found {duplicates_count} duplicate timestamps")

    if method == "keep_first":
        df = df.drop_duplicates(subset=[datetime_column], keep="first")
    elif method == "keep_last":
        df = df.drop_duplicates(subset=[datetime_column], keep="last")
    elif method == "drop":
        df = df.drop_duplicates(subset=[datetime_column], keep=False)

    logger.info(f"After handling duplicates: {len(df)} rows remaining")

    return df


def _fill_missing_values(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Fill missing values using specified method."""

    missing_count = df.isnull().sum().sum()

    if missing_count == 0:
        return df

    logger.info(
        f"Filling {missing_count} missing values using method: {method}")

    if method == "forward":
        df = df.fillna(method="ffill")
    elif method == "backward":
        df = df.fillna(method="bfill")
    elif method == "interpolate":
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate()

    return df


def _validate_resample_frequency(
    df: pd.DataFrame,
    resample_freq: str,
    datetime_is_index: bool,
    datetime_column: str
) -> None:
    """
    Validate that the resample frequency is not higher than the original data frequency.
    Prevents upsampling (e.g., 5min to 1min) which would create artificial data points.
    """

    if len(df) < 2:
        logger.warning(
            "Cannot validate resample frequency: DataFrame has less than 2 rows")
        return

    try:
        if datetime_is_index:
            datetime_series = df.index
        else:
            datetime_series = df[datetime_column]

        time_diffs = pd.Series(datetime_series).diff().dropna().abs()

        if len(time_diffs) == 0:
            logger.warning(
                "Cannot validate resample frequency: No time differences found")
            return

        original_freq_seconds = time_diffs.mode().iloc[0].total_seconds()

        resample_freq_seconds = pd.Timedelta(pd.date_range(
            "2020-01-01", periods=2, freq=resample_freq).freq).total_seconds()

        if resample_freq_seconds < original_freq_seconds:
            raise DataLoaderError(
                f"Cannot upsample data: resample frequency '{resample_freq}' "
                f"({resample_freq_seconds}s) is higher than original data frequency "
                f"(~{original_freq_seconds}s). Upsampling would create artificial data points."
            )

        logger.info(
            f"Resample frequency validation passed: {resample_freq} "
            f"({resample_freq_seconds}s) >= original frequency (~{original_freq_seconds}s)"
        )

    except Exception as e:
        if isinstance(e, DataLoaderError):
            raise
        else:
            logger.warning(f"Could not validate resample frequency: {str(e)}")


def _resample_data(
    df: pd.DataFrame,
    freq: str,
    agg_methods: Optional[Dict[str, Union[str, List[str]]]],
    datetime_is_index: bool
) -> pd.DataFrame:
    """Resample data to specified frequency."""

    if not datetime_is_index:
        raise DataLoaderError("Cannot resample: datetime must be set as index")

    agg = {}
    for col in df.columns:
        col_lower = col.lower()
        if agg_methods and col in agg_methods:
            agg[col] = agg_methods[col]
        elif col_lower == "open":
            agg[col] = "first"
        elif col_lower == "high":
            agg[col] = "max"
        elif col_lower == "low":
            agg[col] = "min"
        elif col_lower == "close":
            agg[col] = "last"
        elif col_lower in ["volume", "vol", "count", "trades"] or "volume" in col_lower or "vol" in col_lower or "count" in col_lower or "trades" in col_lower:
            agg[col] = "sum"
        else:
            agg[col] = "mean"

    try:
        resampled_df = df.resample(freq).agg(agg)
        resampled_df = resampled_df.dropna(how="all")
        logger.info(f"Resampled data shape: {resampled_df.shape}")
        return resampled_df

    except Exception as e:
        raise DataLoaderError(f"Failed to resample data: {str(e)}") from e


def _validate_market_data(df: pd.DataFrame) -> None:
    """Perform basic validation checks on market data."""

    warnings_list = []

    columns_lower = [str(col).lower() for col in df.columns]

    price_columns = [col for col in df.columns
                     if any(price_term in str(col).lower()
                            for price_term in ["price", "open", "high", "low", "close"])]

    for col in price_columns:
        if df[col].dtype in ["int64", "float64"] and (df[col] < 0).any():
            warnings_list.append(
                f"Found negative values in price column '{col}'")

    volume_columns = [col for col in df.columns
                      if any(vol_term in str(col).lower()
                             for vol_term in ["volume", "vol"])]

    for col in volume_columns:
        if df[col].dtype in ["int64", "float64"]:
            zero_volume_count = (df[col] == 0).sum()
            if zero_volume_count > 0:
                warnings_list.append(
                    f"Found {zero_volume_count} zero volume values in column '{col}' "
                    f"({zero_volume_count/len(df)*100:.2f}% of total records)"
                )

    high_cols = [col for col in df.columns if "high" in str(col).lower()]
    low_cols = [col for col in df.columns if "low" in str(col).lower()]

    for high_col in high_cols:
        for low_col in low_cols:
            if (df[high_col] < df[low_col]).any():
                warnings_list.append(
                    f"Found cases where {high_col} < {low_col}")

    for col in price_columns:
        if len(df) > 1:
            pct_change = df[col].pct_change().abs()
            extreme_changes = (pct_change > 0.5).sum()
            if extreme_changes > 0:
                warnings_list.append(
                    f"Found {extreme_changes} extreme price changes (>50%) in column '{col}'"
                )

    if warnings_list:
        logger.warning("Data validation warnings:")
        for warning in warnings_list:
            logger.warning(f"  - {warning}")
    else:
        logger.info("Data validation passed without warnings")
