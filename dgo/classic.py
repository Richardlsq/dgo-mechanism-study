"""JQC1 frozen-policy classical references; no objective-specific information.

This is a local implementation of explicitly declared DE/rand/1/bin and
constriction global-best PSO policies, not a replication of a named code release.
The caller supplies a point-counting vector evaluator and saved initial points.
"""
from __future__ import annotations

import math
import numpy as np

DE_F = 0.5
DE_CR = 0.9
PSO_PHI1 = 2.05
PSO_PHI2 = 2.05
PSO_PHI = PSO_PHI1 + PSO_PHI2
PSO_CHI = 2.0 / (PSO_PHI - 2.0 + math.sqrt(PSO_PHI * PSO_PHI - 4.0 * PSO_PHI))


def reflect(proposed, bounds, velocity=None):
    """Triangle-wave box reflection, including arbitrarily many crossings.

    For an interior folded point the velocity receives the triangle wave's
    slope (+1 or -1). At a folded endpoint, choose the inward one-sided
    velocity: +abs(v) at the lower bound and -abs(v) at the upper bound.
    This convention explicitly resolves exact-boundary parity ambiguity.
    """
    proposed = np.asarray(proposed, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    width = hi - lo
    if not np.isfinite(proposed).all():
        raise FloatingPointError('nonfinite proposed position')
    u = np.remainder((proposed - lo) / width, 2.0)
    folded = lo + width * np.where(u <= 1.0, u, 2.0 - u)
    # Suppress roundoff outside the declared physical box.
    folded = np.clip(folded, lo, hi)
    if velocity is None:
        return folded
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != proposed.shape or not np.isfinite(velocity).all():
        raise ValueError('invalid velocity')
    updated = velocity * np.where(u < 1.0, 1.0, -1.0)
    updated = np.where(u == 0.0, np.abs(velocity), updated)
    updated = np.where(u == 1.0, -np.abs(velocity), updated)
    return folded, updated


def de_step(population, bounds, rng, count=None):
    """Generate a synchronous DE/rand/1/bin batch; no fitness call or update.

    Targets are the index prefix 0..count-1. Each target samples three distinct
    donors excluding itself, then D crossover uniforms, then one forced index.
    """
    population = np.asarray(population, dtype=float)
    n, d = population.shape
    count = n if count is None else int(count)
    if not 1 <= count <= n or n < 4:
        raise ValueError('invalid DE batch size')
    candidates = np.empty((count, d), dtype=float)
    indices = np.arange(n)
    for i in range(count):
        a, b, c = rng.choice(indices[indices != i], size=3, replace=False)
        mutant = population[a] + DE_F * (population[b] - population[c])
        crossed = rng.random(d) < DE_CR
        crossed[int(rng.integers(d))] = True
        candidates[i] = np.where(crossed, mutant, population[i])
    return reflect(candidates, bounds)


def pso_step(position, velocity, personal_best, global_best, bounds, rng, count=None):
    """Generate a synchronous constriction-PSO batch, without fitness calls.

    Both random matrices are independent per particle and per coordinate.
    The supplied global_best is the pre-generation best, held for this batch.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    personal_best = np.asarray(personal_best, dtype=float)
    global_best = np.asarray(global_best, dtype=float)
    n, d = position.shape
    count = n if count is None else int(count)
    if not 1 <= count <= n:
        raise ValueError('invalid PSO batch size')
    r1 = rng.random((count, d))
    r2 = rng.random((count, d))
    new_velocity = PSO_CHI * (
        velocity[:count]
        + PSO_PHI1 * r1 * (personal_best[:count] - position[:count])
        + PSO_PHI2 * r2 * (global_best - position[:count])
    )
    return reflect(position[:count] + new_velocity, bounds, new_velocity)


def _evaluate(ev, points):
    """The external evaluator owns charging, recording and historical best."""
    start = int(ev.nfev)
    values = np.asarray(ev(points), dtype=float)
    if int(ev.nfev) != start + len(points):
        raise AssertionError('evaluator point count does not match batch')
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise FloatingPointError('invalid objective return')
    return values


def solve(method, ev, bounds, budget, seed, population):
    """Spend exactly budget points, including all supplied initialization.

    No target-based stopping, restarts, hidden evaluations, velocity clipping,
    objective names, known optima, or cross-method parameter selection.
    """
    if method not in ('DE', 'PSO', 'Random'):
        raise ValueError(f'unknown classical method: {method}')
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError('bounds must have shape (D, 2)')
    d = len(bounds)
    if d < 1 or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('invalid finite box')
    population = np.array(population, dtype=float, copy=True)
    n = 5 * d if method == 'DE' else 40
    if population.shape != (n, d) or not np.isfinite(population).all():
        raise ValueError(f'{method} requires an ({n}, {d}) population')
    if np.any(population < bounds[:, 0]) or np.any(population > bounds[:, 1]):
        raise ValueError('initial population outside box')
    if isinstance(budget, bool) or int(budget) != budget or budget < n:
        raise ValueError('budget must be integer and cover initialization')
    budget = int(budget)
    if int(ev.nfev) != 0:
        raise ValueError('fresh evaluator required')
    rng = np.random.default_rng(seed)
    values = _evaluate(ev, population)
    batches = 0
    accepted = 0
    last_size = 0
    if method == 'PSO':
        velocity = np.zeros_like(population)
        personal_best = population.copy()
        personal_values = values.copy()
    while int(ev.nfev) < budget:
        count = min(n, budget - int(ev.nfev))
        if method == 'DE':
            candidates = de_step(population, bounds, rng, count)
            candidate_values = _evaluate(ev, candidates)
            better = candidate_values < values[:count]
            ids = np.flatnonzero(better)
            # Every trial was generated before any replacement in this batch.
            population[ids] = candidates[ids]
            values[ids] = candidate_values[ids]
            accepted += len(ids)
        elif method == 'PSO':
            leader = personal_best[int(np.argmin(personal_values))].copy()
            candidates, new_velocity = pso_step(
                population, velocity, personal_best, leader, bounds, rng, count)
            candidate_values = _evaluate(ev, candidates)
            population[:count] = candidates
            velocity[:count] = new_velocity
            values[:count] = candidate_values
            ids = np.flatnonzero(candidate_values < personal_values[:count])
            personal_best[ids] = candidates[ids]
            personal_values[ids] = candidate_values[ids]
            accepted += len(ids)
        else:
            candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, d))
            _evaluate(ev, candidates)
        batches += 1
        last_size = count
    if int(ev.nfev) != budget:
        raise AssertionError('budget not completed exactly')
    stats = {'population_size': n, 'initial_evals': n,
             'post_initial_evals': budget - n, 'update_batches': batches,
             'final_batch_size': last_size,
             'partial_final_batch': bool(last_size and last_size < n)}
    if method == 'DE':
        stats.update(F=DE_F, CR=DE_CR, strict_replacements=accepted,
                     update='synchronous', boundary='triangle_reflection')
    elif method == 'PSO':
        stats.update(chi=PSO_CHI, phi1=PSO_PHI1, phi2=PSO_PHI2,
                     personal_best_improvements=accepted, topology='global_best',
                     update='synchronous', initial_velocity='zero',
                     velocity_clamp=False, boundary='triangle_reflection_velocity_parity',
                     exact_boundary_velocity='inward')
    return {'stop': 'budget', 'stats': stats, 'warnings': []}
