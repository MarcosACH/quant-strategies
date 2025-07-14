"""
Utility functions for Jupyter notebooks in the quant-strategies framework.

This module provides consistent path resolution and setup functions
for notebooks running from different subdirectories.
"""

import sys
import os
from pathlib import Path


def setup_notebook_environment():
    """
    Set up the notebook environment with proper path resolution.

    This function:
    1. Determines the project root directory
    2. Adds the project root to sys.path for imports
    3. Returns the project root path for use in notebooks

    Returns:
        Path: The project root directory path
    """
    notebook_dir = Path(os.getcwd())

    current_dir = notebook_dir
    max_levels = 5

    for _ in range(max_levels):
        if (current_dir / "setup.py").exists() or (current_dir / "pyproject.toml").exists():
            project_root = current_dir
            break
        if (current_dir / "src").exists() and (current_dir / "config").exists():
            project_root = current_dir
            break
        if current_dir.parent == current_dir:
            project_root = notebook_dir.parent.parent
            break
        current_dir = current_dir.parent
    else:
        project_root = notebook_dir.parent.parent

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root


def get_data_path(project_root: Path, *path_parts: str) -> Path:
    """
    Get a path relative to the project's data directory.

    Args:
        project_root: The project root directory
        *path_parts: Path components relative to the data directory

    Returns:
        Path: The resolved data path
    """
    return project_root / "data" / Path(*path_parts)


def get_results_path(project_root: Path, *path_parts: str) -> Path:
    """
    Get a path relative to the project's results directory.

    Args:
        project_root: The project root directory
        *path_parts: Path components relative to the results directory

    Returns:
        Path: The resolved results path
    """
    return project_root / "results" / Path(*path_parts)


def list_available_data_files(project_root: Path, pattern: str = "*.parquet") -> list:
    """
    List available data files matching a pattern.

    Args:
        project_root: The project root directory
        pattern: Glob pattern to match files (default: "*.parquet")

    Returns:
        list: List of matching file paths
    """
    data_dir = project_root / "data"
    files = []

    search_dirs = [
        data_dir / "processed" / "features",
        data_dir / "raw" / "market_data",
        data_dir / "raw" / "external",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            files.extend(search_dir.glob(pattern))

    return sorted(files)


def print_notebook_setup_info(project_root: Path):
    """
    Print useful setup information for debugging.

    Args:
        project_root: The project root directory
    """
    print("=== Notebook Environment Setup ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Project root exists: {project_root.exists()}")
    print(
        f"Python path includes project root: {str(project_root) in sys.path}")

    key_dirs = ["src", "config", "data", "notebooks", "results"]
    print("\n=== Project Structure Check ===")
    for dir_name in key_dirs:
        dir_path = project_root / dir_name
        print(f"{dir_name}/: {'✓' if dir_path.exists() else '✗'}")

    data_files = list_available_data_files(project_root)
    print(f"\n=== Available Data Files ({len(data_files)}) ===")
    for file_path in data_files[:10]:
        rel_path = file_path.relative_to(project_root)
        print(f"  {rel_path}")
    if len(data_files) > 10:
        print(f"  ... and {len(data_files) - 10} more files")
