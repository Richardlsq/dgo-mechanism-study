"""Mechanism-oriented custom test problems; not official BBOB/CEC instances."""
from dataclasses import dataclass
import math
import numpy as np

# Fixed before parameter experiments. Geometry pairs share shift and rotation
# within an instance; they are deliberately paired, not independent families.
PROFILES = [
    dict(id='rosen_axis', family='rosenbrock', rotated=False, condition=1., group='curved'),
    dict(id='rosen_rot', family='rosenbrock', rotated=True, condition=1., group='curved'),
    dict(id='ellip_axis_1e6', family='ellipsoid', rotated=False, condition=1e6, group='quadratic'),
    dict(id='ellip_rot_1e6', family='ellipsoid', rotated=True, condition=1e6, group='quadratic'),
    dict(id='ellip_rot_1e2', family='ellipsoid', rotated=True, condition=1e2, group='quadratic'),
    dict(id='sphere', family='sphere', rotated=False, condition=1., group='quadratic'),
    dict(id='ackley', family='ackley', rotated=False, condition=1., group='control'),
    dict(id='rastrigin_rot', family='rastrigin', rotated=True, condition=1., group='control'),
]


@dataclass
class Problem:
    profile: dict
    bounds: np.ndarray
    shift: np.ndarray
    rotation: np.ndarray
    weights: np.ndarray

    def __call__(self, x):
        z = np.asarray(x, dtype=float)-self.shift
        if z.ndim != 2 or z.shape[1] != len(self.shift):
            raise ValueError('Expected a batch of physical coordinates')
        if self.profile['rotated']:
            z = z @ self.rotation
        family = self.profile['family']
        if family == 'rosenbrock':
            y = z+1
            return np.sum(100*(y[:,1:]-y[:,:-1]**2)**2+(y[:,:-1]-1)**2, axis=1)
        if family == 'ellipsoid':
            return (z*z) @ self.weights
        if family == 'sphere':
            return np.sum(z*z, axis=1)
        if family == 'ackley':
            return np.maximum(0.,-20*np.exp(-.2*np.sqrt(np.mean(z*z,axis=1)))-np.exp(np.mean(np.cos(2*np.pi*z),axis=1))+20+np.e)
        if family == 'rastrigin':
            return np.sum(z*z+10*(1-np.cos(2*np.pi*z)), axis=1)
        raise ValueError(family)


def generate(profile, dimension, instance_seed):
    rng = np.random.default_rng(np.random.SeedSequence([98001, instance_seed, dimension]))
    shift = rng.uniform(-3.,3.,dimension)
    q,r = np.linalg.qr(rng.normal(size=(dimension,dimension)))
    q *= np.where(np.diag(r)>=0,1.,-1.)
    return Problem(dict(profile), np.tile([-5.,5.],(dimension,1)), shift, q,
                   np.logspace(0.,np.log10(profile['condition']),dimension))


def save(path, problem):
    np.savez_compressed(path, bounds=problem.bounds, shift=problem.shift,
                        rotation=problem.rotation, weights=problem.weights)


def restore(path, profile):
    with np.load(path,allow_pickle=False) as d:
        return Problem(dict(profile), *(d[k].copy() for k in ['bounds','shift','rotation','weights']))


def independent_value(problem, x):
    z = [float(a)-float(b) for a,b in zip(x,problem.shift)]
    if problem.profile['rotated']:
        z = [math.fsum(z[k]*float(problem.rotation[k,j]) for k in range(len(z))) for j in range(len(z))]
    family = problem.profile['family']
    if family == 'rosenbrock':
        y=[v+1 for v in z]
        return math.fsum(100*(y[i+1]-y[i]**2)**2+(y[i]-1)**2 for i in range(len(y)-1))
    if family == 'ellipsoid':
        return math.fsum(v*v*float(w) for v,w in zip(z,problem.weights))
    if family == 'sphere':
        return math.fsum(v*v for v in z)
    if family == 'ackley':
        return max(0.,-20*math.exp(-.2*math.sqrt(math.fsum(v*v for v in z)/len(z)))-math.exp(math.fsum(math.cos(2*math.pi*v) for v in z)/len(z))+20+math.e)
    return math.fsum(v*v+10*(1-math.cos(2*math.pi*v)) for v in z)
