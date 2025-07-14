from abc import ABC, abstractmethod
from typing import Dict, Any, List
import vectorbt as vbt
from pathlib import Path
import yaml


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all strategies must implement
    and provides common functionality for parameter management.
    """

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters that override defaults
        """
        self.params = {**self.default_params, **params}
        self.name = self.__class__.__name__
        self.indicator = None
        self._validate_params()

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """
        Default parameter values for the strategy.

        Returns:
            Dict[str, Any]: Dictionary of parameter names and default values
        """
        pass

    @property
    @abstractmethod
    def param_ranges(self) -> Dict[str, List]:
        """
        Parameter ranges for optimization.

        Returns:
            Dict[str, List]: Dictionary of parameter names and their optimization ranges
        """
        pass

    @abstractmethod
    def create_indicator(self) -> vbt.IndicatorFactory:
        """
        Create the vectorbt indicator factory for this strategy.

        Returns:
            vbt.IndicatorFactory: Configured indicator factory
        """
        pass

    @abstractmethod
    def get_order_func_nb(self):
        """
        Get the numba-compiled order function for this strategy.

        Returns:
            Callable: Numba-compiled order function
        """
        pass

    def update_params(self, **new_params):
        """
        Update strategy parameters.

        Args:
            **new_params: New parameter values
        """
        self.params.update(new_params)
        self._validate_params()
        # Recreate indicator if it exists
        if hasattr(self, 'indicator') and self.indicator is not None:
            self.indicator = self.create_indicator()

    def _validate_params(self):
        """Validate parameter values (override in subclasses for custom validation)"""
        pass

    def get_param(self, param_name: str, default=None):
        """
        Get a parameter value.

        Args:
            param_name: Name of the parameter
            default: Default value if parameter doesn't exist

        Returns:
            Parameter value
        """
        return self.params.get(param_name, default)

    def save_config(self, filepath: str):
        """
        Save strategy configuration to YAML file.

        Args:
            filepath: Path to save configuration
        """
        config = {
            'strategy_name': self.name,
            'parameters': self.params,
            'param_ranges': self.param_ranges
        }

        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """
        Load strategy configuration from YAML file.

        Args:
            filepath: Path to configuration file

        Returns:
            Strategy instance with loaded configuration
        """
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        return cls(**config.get('parameters', {}))

    def __str__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"

    def __repr__(self):
        return self.__str__()
