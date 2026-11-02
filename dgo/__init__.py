"""DGO and the fixed JQ100 configuration used in the paper."""
from .optimizer import minimize, OptimizeResult

def jq100(objective, bounds, *, max_evals=10000, seed=None,
          vectorized=False, initial_population=None):
    """Run the paper configuration; every objective point consumes budget."""
    return minimize(objective, bounds, population_size=40, max_evals=max_evals,
                    seed=seed, vectorized=vectorized,
                    initial_population=initial_population, probe=True,
                    immigrants=False, fixed_sweep_probability=0.5,
                    allocation='quota', radius_feedback='pair',
                    third_mode='interpolate', third_fraction=1.0)
