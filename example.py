"""Small usage example; this is not a paper benchmark."""
import numpy as np
from dgo import jq100

if __name__ == '__main__':
    result = jq100(lambda x: np.sum(x*x, axis=-1),
                   np.tile([-5.0, 5.0], (10, 1)),
                   max_evals=1000, seed=1, vectorized=True)
    print(f'Objective: {result.fun:.8g}; evaluations: {result.nfev}')
