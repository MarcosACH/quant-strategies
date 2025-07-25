import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
from skopt.space import Real, Integer


class ParametersSelection:
    """
    Parameter selection engine for strategy optimization.

    This class generates parameter ranges for different optimization methods
    without performing the actual backtesting.
    """

    def __init__(self, param_ranges: Dict[str, List]):
        """Initialize the parameter selector."""
        self.param_ranges = param_ranges

    def get_grid_search_params(self, reduced: bool = False) -> Dict[str, List]:
        """
        Get parameter ranges for grid search.

        Args:
            reduced: Whether to use reduced parameter ranges for faster computation

        Returns:
            Dictionary with parameter ranges for grid search
        """
        if reduced:
            param_ranges = {k: v[:3] for k, v in self.param_ranges.items()}
            print("Using reduced parameter ranges for quick grid search...")
            total_combinations = np.prod(
                [len(v) for v in param_ranges.values()])
            print(
                f"Grid search: {total_combinations:,} parameter combinations")
        else:
            param_ranges = self.param_ranges
            total_combinations = np.prod(
                [len(v) for v in param_ranges.values()])
            print(
                f"Grid search (full): {total_combinations:,} parameter combinations")

        return param_ranges

    def get_random_search_params(self, n_iter: int = 100, custom_ranges: Optional[Dict[str, List]] = None) -> Dict[str, List]:
        """
        Generate random parameter combinations for random search.

        Args:
            n_iter: Number of random combinations to generate
            custom_ranges: Custom parameter ranges (uses base ranges if None)

        Returns:
            Dictionary with parameter ranges for random search
        """
        print(f"Generating {n_iter} random search parameters...")

        param_ranges = custom_ranges or self.param_ranges

        # Convert parameter ranges to distributions
        param_distributions = {}
        for param_name, param_values in param_ranges.items():
            if isinstance(param_values[0], (int, np.integer)):
                param_distributions[param_name] = randint(
                    min(param_values), max(param_values) + 1)
            else:
                param_distributions[param_name] = uniform(
                    min(param_values), max(param_values) - min(param_values))

        param_sampler = ParameterSampler(
            param_distributions, n_iter=n_iter, random_state=42)

        print(
            f"Generated {len(param_sampler):,} random parameter combinations")
        return list(param_sampler)

    def get_bayesian_optimization_params(self,
                                         n_iter: int = 50,
                                         custom_ranges: Optional[Dict[str, List]] = None) -> Tuple[List, List[str]]:
        """
        Get parameter space definition for Bayesian optimization.

        Args:
            n_iter: Number of optimization iterations
            custom_ranges: Custom parameter ranges (uses base ranges if None)

        Returns:
            Tuple of (search dimensions, parameter names)
        """
        print(
            f"Setting up Bayesian optimization space for {n_iter} iterations...")

        param_ranges = custom_ranges or self.param_ranges

        dimensions = []
        param_names = []

        for param_name, param_values in param_ranges.items():
            param_names.append(param_name)
            if isinstance(param_values[0], (int, np.integer)):
                dimensions.append(
                    Integer(min(param_values), max(param_values), name=param_name))
            else:
                dimensions.append(
                    Real(min(param_values), max(param_values), name=param_name))

        print(
            f"Bayesian optimization space configured with {len(dimensions)} parameters")
        return dimensions, param_names
