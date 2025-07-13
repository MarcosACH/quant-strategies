from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import yaml


class BaseStrategyConfig(ABC):
    """
    Abstract base class for strategy configurations.

    Each strategy should have its own configuration class that inherits
    from this base class to ensure consistent parameter management.
    """

    def __init__(self, **kwargs):
        """Initialize configuration with custom parameters."""
        self.params = {**self.default_params, **kwargs}
        self.validate_params()

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values for the strategy."""
        pass

    @property
    @abstractmethod
    def param_ranges(self) -> Dict[str, List]:
        """Parameter ranges for optimization."""
        pass

    @property
    @abstractmethod
    def param_constraints(self) -> Dict[str, Any]:
        """Parameter constraints and validation rules."""
        pass

    def validate_params(self) -> None:
        """Validate parameter values against constraints."""
        constraints = self.param_constraints

        for param_name, value in self.params.items():
            if param_name in constraints:
                constraint = constraints[param_name]

                # Check minimum value
                if 'min' in constraint and value < constraint['min']:
                    raise ValueError(
                        f"{param_name} must be >= {constraint['min']}, got {value}")

                # Check maximum value
                if 'max' in constraint and value > constraint['max']:
                    raise ValueError(
                        f"{param_name} must be <= {constraint['max']}, got {value}")

                # Check data type
                if 'type' in constraint and not isinstance(value, constraint['type']):
                    raise TypeError(
                        f"{param_name} must be of type {constraint['type']}, got {type(value)}")

    def update_param(self, param_name: str, value: Any) -> None:
        """Update a single parameter value."""
        self.params[param_name] = value
        self.validate_params()

    def update_params(self, **kwargs) -> None:
        """Update multiple parameter values."""
        self.params.update(kwargs)
        self.validate_params()

    def get_param(self, param_name: str, default=None):
        """Get a parameter value."""
        return self.params.get(param_name, default)

    def save_config(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_data = {
            'strategy_config': self.__class__.__name__,
            'parameters': self.params,
            'param_ranges': self.param_ranges,
            'param_constraints': self.param_constraints
        }

        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data.get('parameters', {}))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'params': self.params.copy(),
            'param_ranges': self.param_ranges,
            'param_constraints': self.param_constraints
        }

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"

    def __repr__(self):
        return self.__str__()
