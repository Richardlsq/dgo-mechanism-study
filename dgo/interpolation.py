import numpy as np

ETA = 1e-8


def proposals(centers, delta, parent, values, lower, span, mode, rng, upper=None):
    """Return eligible indices and candidate points without objective calls."""
    plus, minus = centers + delta, centers - delta
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gp, gm = values[:, 0]-parent, values[:, 1]-parent
        curvature = gp+gm
        threshold = 64*np.finfo(float).eps*(np.abs(values).sum(axis=1)+2*np.abs(parent))
        fraction = (gm-gp)/(2*curvature)
    valid = np.all((plus >= 0)&(plus <= 1)&(minus >= 0)&(minus <= 1), axis=1)
    valid &= np.isfinite(curvature)&np.isfinite(threshold)&np.isfinite(fraction)
    valid &= (curvature > threshold)&(np.abs(fraction) >= ETA)&(np.abs(fraction) <= 1-ETA)
    # Verify physical mapping retains the symmetric line at machine precision.
    upper = lower+span if upper is None else upper
    physical = np.clip(lower + span*centers, lower, upper)
    pp = np.clip(lower+span*plus, lower, upper)
    pm = np.clip(lower+span*minus, lower, upper)
    tolerance = 64*np.finfo(float).eps*np.maximum(1., np.abs(physical))
    valid &= np.all(np.abs((pp+pm)/2-physical) <= tolerance, axis=1)
    ids = np.flatnonzero(valid)
    if mode == "random":
        magnitudes = rng.uniform(ETA, 1-ETA, len(ids))
        fractions = magnitudes*np.where(rng.integers(0, 2, len(ids)), 1., -1.)
    else:
        fractions = fraction[ids]
    points = centers[ids]+fractions[:, None]*delta[ids]
    candidate = np.clip(lower+span*points, lower, upper)
    different = np.any(candidate != physical[ids], axis=1)
    different &= np.any(candidate != pp[ids], axis=1)&np.any(candidate != pm[ids], axis=1)
    return ids[different], points[different]
