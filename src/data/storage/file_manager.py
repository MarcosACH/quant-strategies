import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
# import h5py
import pickle
import json
from datetime import datetime
import gzip
import shutil

from config.settings import settings


class FileManager:
    """
    File system operations and data serialization manager.

    This class handles saving and loading of various data formats
    including CSV, HDF5, Parquet, and pickle files with compression
    and organization utilities.
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize file manager.

        Args:
            base_path: Base directory for data storage
        """
        self.base_path = base_path or settings.DATA_ROOT_PATH
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_path,
            self.base_path / "raw",
            self.base_path / "processed",
            self.base_path / "features",
            self.base_path / "backtest_results",
            self.base_path / "cache"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # def save_market_data(
    #     self,
    #     data: pd.DataFrame,
    #     symbol: str,
    #     timeframe: str,
    #     source: str = "unknown",
    #     format: str = "parquet",
    #     compress: bool = True
    # ) -> Path:
    #     """
    #     Save market data to file.

    #     Args:
    #         data: OHLCV DataFrame
    #         symbol: Asset symbol
    #         timeframe: Data timeframe
    #         source: Data source identifier
    #         format: File format ('csv', 'parquet', 'hdf5')
    #         compress: Whether to compress the file

    #     Returns:
    #         Path to saved file
    #     """
    #     # Create filename
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"{symbol}_{timeframe}_{source}_{timestamp}"

    #     if format == "parquet":
    #         filepath = self.base_path / "raw" / f"{filename}.parquet"
    #         if compress:
    #             data.to_parquet(filepath, compression='gzip')
    #         else:
    #             data.to_parquet(filepath)

    #     elif format == "csv":
    #         filepath = self.base_path / "raw" / f"{filename}.csv"
    #         if compress:
    #             data.to_csv(filepath, compression='gzip')
    #         else:
    #             data.to_csv(filepath)

    #     elif format == "hdf5":
    #         filepath = self.base_path / "raw" / f"{filename}.h5"
    #         with h5py.File(filepath, 'w') as f:
    #             # Store data arrays
    #             for col in data.columns:
    #                 f.create_dataset(
    #                     col, data=data[col].values, compression='gzip' if compress else None)

    #             # Store metadata
    #             f.attrs['symbol'] = symbol
    #             f.attrs['timeframe'] = timeframe
    #             f.attrs['source'] = source
    #             f.attrs['created_at'] = timestamp

    #     else:
    #         raise ValueError(f"Unsupported format: {format}")

    #     return filepath

    # def load_market_data(
    #     self,
    #     filepath: Union[str, Path],
    #     format: str = None
    # ) -> pd.DataFrame:
    #     """
    #     Load market data from file.

    #     Args:
    #         filepath: Path to data file
    #         format: File format (auto-detected if None)

    #     Returns:
    #         OHLCV DataFrame
    #     """
    #     filepath = Path(filepath)

    #     if format is None:
    #         format = filepath.suffix.lower().lstrip('.')

    #     if format == "parquet":
    #         return pd.read_parquet(filepath)

    #     elif format == "csv":
    #         return pd.read_csv(filepath, index_col=0, parse_dates=True)

    #     elif format in ["hdf5", "h5"]:
    #         data_dict = {}
    #         with h5py.File(filepath, 'r') as f:
    #             for key in f.keys():
    #                 data_dict[key] = f[key][:]

    #         return pd.DataFrame(data_dict)

    #     else:
    #         raise ValueError(f"Unsupported format: {format}")

    # def save_backtest_results(
    #     self,
    #     results: pd.DataFrame,
    #     strategy_name: str,
    #     symbol: str,
    #     timeframe: str,
    #     date_range: str,
    #     compress: bool = True
    # ) -> Path:
    #     """
    #     Save backtest results to file.

    #     Args:
    #         results: Backtest results DataFrame
    #         strategy_name: Strategy name
    #         symbol: Asset symbol
    #         timeframe: Data timeframe
    #         date_range: Date range identifier
    #         compress: Whether to compress the file

    #     Returns:
    #         Path to saved file
    #     """
    #     filename = f"{strategy_name}_{symbol}_{timeframe}_{date_range}_results.parquet"
    #     filepath = self.base_path / "backtest_results" / filename

    #     if compress:
    #         results.to_parquet(filepath, compression='gzip')
    #     else:
    #         results.to_parquet(filepath)

    #     return filepath

    # def load_backtest_results(self, filepath: Union[str, Path]) -> pd.DataFrame:
    #     """
    #     Load backtest results from file.

    #     Args:
    #         filepath: Path to results file

    #     Returns:
    #         Backtest results DataFrame
    #     """
    #     return pd.read_parquet(filepath)

    def save_features(
        self,
        features: pd.DataFrame,
        feature_set_name: str,
        symbol: str,
        timeframe: str,
        compress: bool = True
    ) -> Path:
        """
        Save feature data to file.

        Args:
            features: Features DataFrame
            feature_set_name: Name of the feature set
            symbol: Asset symbol
            timeframe: Data timeframe
            compress: Whether to compress the file

        Returns:
            Path to saved file
        """
        filename = f"{feature_set_name}_{symbol}_{timeframe}_features.parquet"
        filepath = self.base_path / "features" / filename

        if compress:
            features.to_parquet(filepath, compression='gzip')
        else:
            features.to_parquet(filepath)

        return filepath

    def load_features(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load feature data from file.

        Args:
            filepath: Path to features file

        Returns:
            Features DataFrame
        """
        return pd.read_parquet(filepath)

    def save_object(
        self,
        obj: Any,
        filename: str,
        subfolder: str = "cache",
        compress: bool = True
    ) -> Path:
        """
        Save arbitrary Python object to file.

        Args:
            obj: Object to save
            filename: Filename (without extension)
            subfolder: Subfolder within base path
            compress: Whether to compress the file

        Returns:
            Path to saved file
        """
        filepath = self.base_path / subfolder / f"{filename}.pkl"

        if compress:
            with gzip.open(f"{filepath}.gz", 'wb') as f:
                pickle.dump(obj, f)
            return Path(f"{filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            return filepath

    def load_object(self, filepath: Union[str, Path]) -> Any:
        """
        Load Python object from file.

        Args:
            filepath: Path to object file

        Returns:
            Loaded object
        """
        filepath = Path(filepath)

        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    def save_json(
        self,
        data: Dict[str, Any],
        filename: str,
        subfolder: str = "cache",
        compress: bool = False
    ) -> Path:
        """
        Save dictionary to JSON file.

        Args:
            data: Dictionary to save
            filename: Filename (without extension)
            subfolder: Subfolder within base path
            compress: Whether to compress the file

        Returns:
            Path to saved file
        """
        filepath = self.base_path / subfolder / f"{filename}.json"

        if compress:
            with gzip.open(f"{filepath}.gz", 'wt') as f:
                json.dump(data, f, indent=2, default=str)
            return Path(f"{filepath}.gz")
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return filepath

    def load_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load dictionary from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Loaded dictionary
        """
        filepath = Path(filepath)

        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt') as f:
                return json.load(f)
        else:
            with open(filepath, 'r') as f:
                return json.load(f)

    def list_files(
        self,
        subfolder: str = "",
        pattern: str = "*",
        extension: str = None
    ) -> List[Path]:
        """
        List files in a directory.

        Args:
            subfolder: Subfolder within base path
            pattern: File pattern to match
            extension: File extension filter

        Returns:
            List of file paths
        """
        search_path = self.base_path / subfolder

        if extension:
            pattern = f"*.{extension.lstrip('.')}"

        return list(search_path.glob(pattern))

    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {"exists": False}

        stat = filepath.stat()

        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "extension": filepath.suffix,
            "stem": filepath.stem
        }

    def cleanup_cache(self, max_age_days: int = 7):
        """
        Clean up old cache files.

        Args:
            max_age_days: Maximum age of files to keep
        """
        cache_path = self.base_path / "cache"
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)

        for file in cache_path.rglob("*"):
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                file.unlink()

    def compress_directory(self, directory: Union[str, Path], output_path: Union[str, Path]):
        """
        Compress a directory to a tar.gz archive.

        Args:
            directory: Directory to compress
            output_path: Path for the compressed archive
        """
        shutil.make_archive(str(output_path), 'gztar', str(directory))

    def get_disk_usage(self) -> Dict[str, float]:
        """
        Get disk usage statistics for the data directory.

        Returns:
            Dictionary with usage statistics in MB
        """
        total_size = 0
        file_count = 0

        for file in self.base_path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
                file_count += 1

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "total_files": file_count,
            "average_file_size_mb": (total_size / file_count) / (1024 * 1024) if file_count > 0 else 0
        }
