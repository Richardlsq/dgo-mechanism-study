# DGO mechanism study

This is the open-source code and results repository for the published arXiv preprint **[Local Budget Benefits, Population Commitment, and Solver Performance: A DGO Case Study with Controlled DE Transfer](https://arxiv.org/abs/2610.41099)** (arXiv:2610.41099), by **Shiqi Li**, School of Traffic and Transportation, Lanzhou Jiaotong University.

DGO is a population optimizer designed for this study. The paper distinguishes local evaluation gains, population writeback effects, and complete-solver performance. Positive and negative results are both included.

## Contents

```text
dgo/                   Core optimization and intervention code
results/               Scientific configurations, inputs and saved results
example.py             Small JQ100 usage example
summarize_results.py   Recompute BX1 and PC1 comparisons from saved data
requirements.txt      Core dependency
LICENSE               Project code license
README.md             Documentation for the whole package
```

## Run JQ100

Python 3.10 or later and NumPy are required. The study used Python 3.14.4 and NumPy 2.4.4; matching those versions is preferable for numerical comparisons.

```bash
python -m pip install -r requirements.txt
python example.py
```

```python
import numpy as np
from dgo import jq100

bounds = np.tile([-5.0, 5.0], (10, 1))
result = jq100(lambda x: np.sum(x*x, axis=-1), bounds,
               max_evals=1000, seed=1, vectorized=True)
print(result.x, result.fun, result.nfev)
```

The objective must be deterministic and return finite values. With `vectorized=True`, it receives a batch and returns one value per point; otherwise it receives a single point and returns a scalar. Bounds have shape `(dimension, 2)`. Initialization and every candidate evaluation count toward `max_evals`. An optional `initial_population` has shape `(40, dimension)` in physical coordinates.

`jq100` fixes population size 40, equal sweep/probe quotas, pair-based radius feedback, full interpolation fraction, and no scouts. `dgo.minimize` exposes the configurable underlying optimizer; its defaults are **not** JQ100. No parameter was retuned for this release.

| File | Role |
|---|---|
| `dgo/optimizer.py` | Original numerical core used by JQ100 |
| `dgo/interpolation.py` | Quadratic interpolation eligibility and candidate construction |
| `dgo/continuation.py` | DGO state capture, resumption and single writeback intervention |
| `dgo/de_transfer.py` | Synchronous DE with a diagnostic probe and writeback intervention (M3) |
| `dgo/classic.py` | Fixed DE/rand/1/bin, constriction PSO and uniform random search |
| `dgo/benchmarks.py` | Custom mechanism-test functions, not official BBOB functions |

Numerical implementations retain their study behavior; distribution changes are package imports and introductory documentation. `classic.solve` takes a point-counting evaluator with `nfev` and a supplied initial population. DE uses `5 * dimension` points; PSO and random search use 40. This is the JQC1 reference implementation, not PC1's separate DE40 policy. Continuation and DE-transfer modules expose their study-specific low-level APIs. Optional exploratory modules and experiment scheduling are outside this core distribution.

## Results and paper correspondence

| Directory | Evidence |
|---|---|
| `JQC1` | JQ100 versus classical DE, PSO and random search: 960 complete-policy runs |
| `JQX1` | JQ100, NQ, S0, L-SHADE and CMA-ES: 450 runs |
| `HXE1` | Earlier H10/S0 versus strong references: 360 runs |
| `J1`, `J2`, `J3` | Interpolation selection, short-horizon local value, and the 24-FE local study |
| `J5` | Two original-path tail cases and their continuations |
| `J6` | DGO local/continuation study: 48 states and paired futures |
| `M3` | Controlled DE transfer: 32 cases, including nonselected cases |
| `CT1` | Candidate-type/radius-policy study: 128 states and 256 paired futures |
| `PC1` | Matched-start, restart and parameter comparisons: 1,296 runs; serial timing |
| `BX1` | 24 boxed BBOB functions, dimensions 10 and 20: 1,728 runs |
| `S1` | Saved-data sensitivity, target-arrival and distribution analyses |
| `earlier_studies` | Run-level and local-budget summaries for earlier studies |
| `supplement` | Updated intervals, ordered-FE target counts and acceptance comparisons |

`config.json` retains scientific settings, instance/search seeds and budgets. Where supplied, `inputs/` contains numerical instance/initialization arrays; read NPZ files with `numpy.load(..., allow_pickle=False)`. `runs.jsonl.gz` contains one JSON object per run, readable with Python's `gzip` and `json` modules. Formal run records retain endpoints, available histories, actual FE, stopping information and timing. Continuations distinguish prefix FE from newly paid FE. CSV files have header rows; empty target-arrival entries indicate no observed hit. Nonselected cases remain in the case tables.

`fun` is the objective value on custom problems. BX1 supplies `raw_value`, `fopt`, `raw_error` and `error`; its objectives have offsets, and comparisons use nonnegative error relative to the supplied optimum. `L` and `C` denote L-SHADE and native CMA-ES. `Cnative` and `Crestart` are distinct policies. Use `nfev` for actual evaluations, rather than substituting the budget.

## Recompute key saved-data results

```bash
python summarize_results.py
```

This reconstructs 10 BX1 and six matched-start PC1 comparisons, including their shared-seed bootstrap intervals, and compares them with the supplied results. It performs no optimization or objective evaluations and writes no data.

BX1 uses the geometric mean of paired error ratios truncated at 0.01, with 24 equally weighted functions. The same resampled search-seed indices are used across functions, with instance resampling within functions. PC1 uses profile-specific targets and equal weights for six families. Do not pool rows as independent observations or replace these weights by row counts.

## Reading the data

- The adopted BX1/PC1 intervals are in `supplement/bx1_blocks.json` and `supplement/pc1_intervals.json`. BX1's `old_interval` fields retain the earlier calculation, not the final uncertainty analysis. Two BX1 search seeds cover a narrow empirical sensitivity range.
- Use `supplement/bx1_exact_first_targets.csv` and `bx1_exact_ecdf.csv` for ordered-FE statements. BX1's earlier `first_targets.csv` and `ecdf.csv` use batch-visible observations; the two clocks answer different questions.
- `BX1/observation_origin.csv` identifies 864 runs whose initial objective observations were reconstructed through separate evaluations. Terminal solutions and later observations were unchanged. These initial values are not contemporaneous observations from the original runs.
- CT1 compares type-specific writeback after a common paid diagnostic batch, not independently deployed candidate generators. Its stricter interpolation-rule comparison is in `supplement/ct1_acceptance.csv`. Some CT1 seed integers had been used for different purposes in M3; their constructed streams differ, but numerical seed disjointness is not claimed.
- Positive local findings, mixed continuation effects, early stops, unsuccessful targets and adverse complete-policy outcomes are retained. Intervals crossing zero or parity do not establish equivalence.
- Serial timing in `PC1/profile.json` is distinct from timings collected alongside other tasks. Nested objective/monitor time is part of solver time and should not be added to it.

This package supports running the DGO core and inspecting/reaggregating saved results. It omits full candidate streams, checkpoints and complete experiment drivers, so it is not a one-command rerun of the entire paper.

## External algorithms and license

Project code uses the MIT license in `LICENSE`, matching the public repository. Numerical files contain the author's generated research results. No third-party source, compiled library or runtime is bundled.

CMA-ES comparisons used [pycma 4.5.0](https://github.com/CMA-ES/pycma/releases/tag/r4.5.0), distributed under BSD-3-Clause. BX1 used [COCO](https://github.com/numbbo/coco), `coco-experiment` 2.8.3, with the `bbob-boxed` suite. Obtain these dependencies separately under their own terms; neither is needed to read the results or run JQ100.

The inspected [L-SHADE author distribution](https://ryojitanabe.github.io/publication) did not provide an explicit redistribution license. Its source, derived C++ adapter and binaries are excluded; numerical comparison results are included. Downloading the upstream implementation alone does not reconstruct the study-specific adapter. Exact rerunning of that arm requires the omitted implementation and adapter; no substitute is claimed equivalent.

## Citation

Shiqi Li. *Local Budget Benefits, Population Commitment, and Solver Performance: A DGO Case Study with Controlled DE Transfer*. arXiv:2610.41099, 2026. https://arxiv.org/abs/2610.41099

```bibtex
@misc{li2026localbudget,
  title={Local Budget Benefits, Population Commitment, and Solver Performance: A DGO Case Study with Controlled DE Transfer},
  author={Shiqi Li},
  year={2026},
  eprint={2610.41099},
  archivePrefix={arXiv},
  primaryClass={cs.NE},
  url={https://arxiv.org/abs/2610.41099}
}
```
