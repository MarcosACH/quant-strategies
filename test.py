import numpy as np
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint


param_ranges = {
    "bbands_length": np.arange(25, 150, 10),
    "bbands_stddev": np.arange(2.0, 6.0, 0.5),
    "cvd_length": [40],  # np.arange(35, 60, 5),
    "atr_length": [10],  # np.arange(5, 25, 5),
    "sl_coef": [2.0],  # np.arange(2.0, 3.5, 0.5),
    "tpsl_ratio": [2.5],  # np.arange(3.0, 5.5, 0.5)
}

param_distributions = {}
for param_name, param_values in param_ranges.items():
    if isinstance(param_values[0], (int, np.integer)):
        param_distributions[param_name] = randint(
            min(param_values), max(param_values) + 1)
    else:
        param_distributions[param_name] = uniform(
            min(param_values), max(param_values) - min(param_values))

param_sampler = ParameterSampler(
    param_distributions, n_iter=100, random_state=42)

# Convert to list to enable slicing
all_params = list(param_sampler)

batch_size = 50
total_processed = 0
while True:
    batch_params = all_params[total_processed:total_processed + batch_size]
    if not batch_params:
        break
    print(
        f"Processing batch {total_processed//batch_size + 1}: {len(batch_params)} parameters")
    print(f"Batch contents: {batch_params}")
    print(len(batch_params))
    total_processed += batch_size
