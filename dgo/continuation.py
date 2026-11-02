"""DGO research optimizer.

该实验原型组合 DE/current-to-pbest/1、随机双向试探、成功率控制和均匀侦察。
这些组成机制有既有研究；本文件不宣称首创、普适优势或有限预算全局最优。
核心只依赖 NumPy。目标函数必须确定、有限，求解范围为非退化有限盒约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
import copy
from .interpolation import proposals


@dataclass(frozen=True)
class OptimizeResult:
    """优化结果；历史记录按实际评估点数计费，而不是按 Python 调用次数。"""

    x: np.ndarray
    fun: float
    nfev: int
    history_evals: np.ndarray
    history_best: np.ndarray
    stats: dict[str, int | float]


def _reflect(points: np.ndarray) -> np.ndarray:
    """将任意有限坐标反射回单位盒，支持跨越多个边界。"""
    return 1.0 - np.abs(1.0 - np.remainder(points, 2.0))


def _sample_excluding(
    rng: np.random.Generator, population_size: int, blocked: np.ndarray
) -> np.ndarray:
    """每行从排除互异 blocked 索引后的集合均匀抽取一个索引。

    排除数至多为 3，通过跳过已排除位置实现 O(行数)，不构建 N×N 矩阵。
    """
    excluded = np.sort(blocked, axis=1)
    sampled = rng.integers(0, population_size - excluded.shape[1], len(excluded))
    for column in excluded.T:
        sampled += sampled >= column
    return sampled


def minimize(
    objective: Callable[[np.ndarray], np.ndarray | float],
    bounds: np.ndarray,
    *,
    population_size: int = 40,
    max_evals: int = 10000,
    seed: int | None = None,
    vectorized: bool = False,
    initial_population: np.ndarray | None = None,
    probe: bool = True,
    adaptive: bool = True,
    immigrants: bool = True,
    adapt_radius: bool = True,
    fixed_sweep_probability: float | None = None,
    observer: Callable | None = None,
    third_mode: str = "none",
    radius_feedback: str = "macro",
    allocation: str = "probability",
    third_fraction: float = 1.0,
    state_observer: Callable | None = None,
    trace_observer: Callable | None = None,
    resume_state: dict | None = None,
) -> OptimizeResult:
    """在盒约束内最小化确定性、无导数的实值目标函数。

    参数
    ----
    objective : callable
        vectorized=False 时接收形状 (D,) 并返回标量；为 True 时接收
        (M,D) 并返回 (M,)。每个输入点计一次评估，双向试探计两次。
        NaN、正负无穷和错误输出形状均抛出 ValueError；目标异常原样传播。
    bounds : array_like, shape (D,2)
        各维有限下界与上界，严格要求下界小于上界。
    population_size : int
        种群大小 N，至少为 4。默认 40。
    max_evals : int
        包含初始化的严格总评估点数 B，至少为 N；不会超出或漏用预算。
    seed : int or None
        局部随机数生成器种子，不修改 NumPy 的全局随机状态。
    vectorized : bool
        是否使用批量目标接口。真正批量运算应在目标函数内部实现。
    initial_population : array_like, optional
        原始物理坐标中的初始种群，形状 (N,D)，必须处于边界内。
    probe, adaptive, immigrants, adapt_radius : bool
        消融开关：双向试探、自适应算子调度、每轮均匀侦察、棍长自适应。
        禁用的评估预算自动交给剩余操作；adaptive=False 时两算子各半。

    返回
    ----
    OptimizeResult
        最佳解、最佳值、实际评估数、按批记录的历史最优及算子计数。

    说明
    ----
    每轮先评估一个均匀侦察点（启用时），然后选择一个搜索算子。
    扫棍最多评估 N 个目标，点棍最多试探 floor(N/2) 个个体的两个端点。
    两算子近似享有相同每轮预算；不足两个剩余点时使用一次扫棍。
    采用严格改善替换，目标函数的平移不影响成功计数或算子调度。
    """
    if allocation not in ("probability", "quota"):
        raise ValueError("allocation must be probability or quota")
    if not np.isfinite(third_fraction) or not 0 <= third_fraction <= 1:
        raise ValueError("third_fraction must be in [0,1]")
    if allocation == "quota" and (immigrants or not probe or fixed_sweep_probability != 0.5):
        raise ValueError("quota requires no scouts, probes enabled and fixed target 0.5")
    if radius_feedback not in ("macro", "pair"):
        raise ValueError("radius_feedback must be macro or pair")
    if third_mode not in ("none", "random", "interpolate"):
        raise ValueError("invalid third mode")
    third_rng = np.random.default_rng(None if seed is None else np.random.SeedSequence([seed, 94007]))
    if fixed_sweep_probability is not None:
        if not np.isfinite(fixed_sweep_probability) or not 0 <= fixed_sweep_probability <= 1:
            raise ValueError("fixed probability must be finite and in [0,1]")
    if observer is not None and immigrants:
        raise ValueError("S03 observer requires immigrants=False")
    if not callable(objective):
        raise TypeError("objective 必须可调用")
    box = np.asarray(bounds, dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] != 2:
        raise ValueError("bounds 必须具有非空形状 (D,2)")
    if not np.all(np.isfinite(box)) or np.any(box[:, 0] >= box[:, 1]):
        raise ValueError("bounds 必须有限且每一维下界严格小于上界")
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, Integral)
        or population_size < 4
    ):
        raise ValueError("population_size 必须是至少为 4 的整数")
    if (
        isinstance(max_evals, bool)
        or not isinstance(max_evals, Integral)
        or max_evals < population_size
    ):
        raise ValueError("max_evals 必须是至少为 population_size 的整数")
    n, dimension, budget = int(population_size), len(box), int(max_evals)
    lower = box[:, 0]
    with np.errstate(over="ignore"):
        span = box[:, 1] - lower
    if not np.all(np.isfinite(span)):
        raise ValueError("边界跨度溢出，请先对变量缩放")
    rng = np.random.default_rng(seed)
    if resume_state is not None:
        if not isinstance(resume_state, dict):
            raise TypeError("resume_state must be a round-end state dictionary")
        population = np.asarray(resume_state["population"], dtype=float).copy()
    elif initial_population is None:
        population = rng.random((n, dimension))
    else:
        initial = np.asarray(initial_population, dtype=float)
        if initial.shape != (n, dimension):
            raise ValueError("initial_population 必须具有形状 (N,D)")
        if (
            not np.all(np.isfinite(initial))
            or np.any(initial < lower)
            or np.any(initial > box[:, 1])
        ):
            raise ValueError("initial_population 必须有限且位于边界内")
        population = (initial - lower) / span

    nfev = 0
    best_value = float("inf")
    best_x: np.ndarray | None = None
    history_evals: list[int] = []
    history_best: list[float] = []
    stats: dict[str, int | float] = {
        "initial_evals": n,
        "sweep_evals": 0,
        "probe_evals": 0,
        "scout_evals": 0,
        "sweep_successes": 0,
        "probe_successes": 0,
        "scout_accepts": 0,
        "rounds": 0,
        "third_evals": 0,
        "third_improvements": 0,
    }

    def evaluate(unit_points: np.ndarray) -> np.ndarray:
        """批量评估、精确计费并保存所有已评估点中的历史最优。"""
        nonlocal nfev, best_value, best_x
        # 极端非对称边界下，加法舍入也可能越界；最终限幅保证回调输入可行。
        physical = np.clip(lower + span * unit_points, lower, box[:, 1])
        count = len(physical)
        if nfev + count > budget:
            raise RuntimeError("内部评估预算错误")
        if vectorized:
            values = np.asarray(objective(physical), dtype=float)
        else:
            values = np.asarray([objective(row) for row in physical], dtype=float)
        if values.shape != (count,):
            raise ValueError(f"目标输出形状应为 ({count},)，实际为 {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("目标输出必须有限，不能含 NaN 或正负无穷")
        nfev += count
        winner = int(np.argmin(values))
        if values[winner] < best_value:
            best_value = float(values[winner])
            best_x = physical[winner].copy()
        history_evals.append(nfev)
        history_best.append(best_value)
        return values

    if resume_state is None:
        fitness = evaluate(population)
    else:
        fitness = np.asarray(resume_state["fitness"], dtype=float).copy()
    radius_initial = 0.1 * np.sqrt(dimension)
    radius_max = 0.5 * np.sqrt(dimension)
    radii = np.full(n, radius_initial)
    # Q 值表示单位评估成功次数的指数滑动平均；正先验避免冷启动除零。
    scores = np.array([0.25, 0.25])
    elite_size = min(n, max(2, int(np.ceil(0.2 * n))))
    all_indices = np.arange(n)
    sweep_probability = 1.0 if not probe else 0.5

    last_updated = np.full(2, nfev, dtype=np.int64)

    if resume_state is not None:
        # Only a complete round-end state is resumable. A third_ready snapshot
        # has already paid objective calls but has not yet settled this round.
        restored = copy.deepcopy(resume_state)
        population = np.asarray(restored["population"], dtype=float).copy()
        fitness = np.asarray(restored["fitness"], dtype=float).copy()
        radii = np.asarray(restored["radii"], dtype=float).copy()
        scores = np.asarray(restored["scores"], dtype=float).copy()
        last_updated = np.asarray(restored["last_updated"], dtype=np.int64).copy()
        stats = copy.deepcopy(restored["stats"])
        if isinstance(restored["nfev"], bool) or not isinstance(restored["nfev"], Integral):
            raise ValueError("resume nfev must be an integer")
        nfev = int(restored["nfev"])
        best_value = float(restored["best_value"])
        best_x = np.asarray(restored["best_x"], dtype=float).copy()
        history_evals = list(restored["history_evals"])
        history_best = list(restored["history_best"])
        sweep_probability = float(restored["sweep_probability"])
        if population.shape != (n, dimension) or fitness.shape != (n,) or radii.shape != (n,):
            raise ValueError("resume population/fitness/radii shape mismatch")
        if not (np.all(np.isfinite(population)) and np.all((population >= 0) & (population <= 1))
                and np.all(np.isfinite(fitness)) and np.all(np.isfinite(radii))
                and np.all((radii >= 1e-12) & (radii <= radius_max))):
            raise ValueError("resume population/fitness/radii are invalid")
        if scores.shape != (2,) or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("resume scores are invalid")
        if last_updated.shape != (2,) or np.any(last_updated < 0) or np.any(last_updated > nfev):
            raise ValueError("resume last_updated is invalid")
        if not n <= nfev <= budget or not isinstance(stats, dict) or stats.get("initial_evals") != n:
            raise ValueError("resume budget/initial accounting mismatch")
        if any(isinstance(stats.get(key), bool) or not isinstance(stats.get(key), Integral)
               or stats[key] < 0 for key in ("rounds", "sweep_evals", "probe_evals", "scout_evals")):
            raise ValueError("resume round/cost counters are invalid")
        if n + stats["sweep_evals"] + stats["probe_evals"] + stats["scout_evals"] != nfev:
            raise ValueError("resume_state is not a settled round-end state")
        if best_x.shape != (dimension,) or not np.all(np.isfinite(best_x)) or not np.isfinite(best_value):
            raise ValueError("resume best is invalid")
        if np.any(best_x < lower) or np.any(best_x > box[:, 1]):
            raise ValueError("resume best_x is outside the physical box")
        if not 0 <= sweep_probability <= 1:
            raise ValueError("resume sweep_probability is invalid")
        if not history_evals or len(history_evals) != len(history_best):
            raise ValueError("resume history is missing or has mismatched lengths")
        if history_evals[-1] != nfev or history_best[-1] != best_value:
            raise ValueError("resume history does not end at the paid best/state")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in history_evals):
            raise ValueError("resume history_evals must contain integers")
        if np.any(np.diff(np.asarray(history_evals, dtype=np.int64)) <= 0):
            raise ValueError("resume history_evals must be increasing")
        if not np.all(np.isfinite(history_best)) or np.any(np.diff(history_best) > 0):
            raise ValueError("resume history_best must be finite and nonincreasing")
        rng.bit_generator.state = copy.deepcopy(restored["rng_state"])
        third_rng.bit_generator.state = copy.deepcopy(restored["third_rng_state"])

    def export_state():
        """Copy all mutable continuation state; coordinates in population are unit-box."""
        return copy.deepcopy(dict(
            population=population, fitness=fitness, radii=radii, scores=scores,
            last_updated=last_updated, stats=stats, nfev=nfev,
            best_value=best_value, best_x=best_x,
            history_evals=history_evals, history_best=history_best,
            rng_state=rng.bit_generator.state,
            third_rng_state=third_rng.bit_generator.state,
            sweep_probability=sweep_probability,
        ))

    def emit_trace(stage, details=None, event=None):
        if trace_observer is not None:
            # No internal array, nested dictionary or RNG state escapes by alias.
            trace_observer(copy.deepcopy(dict(stage=stage, state=export_state(),
                                             event=event, details=details)))

    def probe_trace_details():
        # Called only in the current P branch. Unpaid proposals are not reported
        # as evaluated third points, including the third_mode=none observer path.
        return dict(targets=targets, centers=centers, parent_values=parent_values,
                    directions=directions, delta=delta, endpoints=endpoints,
                    endpoint_values=endpoint_values, pair_points=pair_points,
                    pair_values=pair_values, pair_success=pair_success,
                    third_ids=third_ids,
                    third_points=third_points if extra else np.empty((0, dimension)),
                    third_values=third_values, third_better=third_better,
                    selected_points=selected_points, selected_values=selected_values,
                    success=success, radius_before=radius_before,
                    count=count, extra=extra, eligible=eligible,
                    improved_by_third=improved_by_third, best_side=best_side)

    while nfev < budget:
        if trace_observer is not None:
            emit_trace("pre")
        details = None
        stats["rounds"] += 1
        if immigrants:
            scout = rng.random((1, dimension))
            scout_value = evaluate(scout)[0]
            stats["scout_evals"] += 1
            worst = int(np.argmax(fitness))
            if scout_value < fitness[worst]:
                population[worst] = scout[0]
                fitness[worst] = scout_value
                radii[worst] = radius_initial
                stats["scout_accepts"] += 1
            if nfev == budget:
                break

        if allocation == "quota":
            s_count, p_count = stats["sweep_evals"], stats["probe_evals"]
            sweep_probability = 1.0 if s_count < p_count else 0.0 if s_count > p_count else 0.5
        elif fixed_sweep_probability is not None:
            sweep_probability = float(fixed_sweep_probability)
        elif not probe:
            sweep_probability = 1.0
        elif adaptive:
            total_score = float(scores.sum())
            sweep_probability = (
                0.2 + 0.6 * scores[0] / total_score if total_score > 0 else 0.5
            )
        else:
            sweep_probability = 0.5

        event = None
        if observer is not None or trace_observer is not None:
            event = {"round": int(stats["rounds"]), "nfev_before": nfev,
                     "allocation": allocation, "third_fraction": float(third_fraction),
                     "sweep_spent_before": int(stats["sweep_evals"]),
                     "probe_spent_before": int(stats["probe_evals"]),
                     "p_sweep": float(sweep_probability), "q_s_before": float(scores[0]),
                     "q_p_before": float(scores[1]), "age_s_evals": int(nfev-last_updated[0]),
                     "age_p_evals": int(nfev-last_updated[1]), "best_before": best_value}
        remaining = budget - nfev
        use_probe = probe and remaining >= 2 and rng.random() >= sweep_probability
        if use_probe:
            count = min(n // 2, remaining // 2)
            targets = rng.choice(n, count, replace=False)
            radius_before = radii[targets].copy()
            centers = population[targets].copy()
            parent_values = fitness[targets].copy()
            directions = rng.normal(size=(count, dimension))
            norms = np.linalg.norm(directions, axis=1)
            zero = norms == 0.0
            if np.any(zero):
                directions[zero, 0] = 1.0
                norms[zero] = 1.0
            directions /= norms[:, None]
            delta = radii[targets, None] * directions
            # 每行相邻存储正、负端点，保证配对与成本均可直接检查。
            endpoints = np.stack(
                (population[targets] + delta, population[targets] - delta), axis=1
            )
            endpoints = _reflect(endpoints.reshape(2 * count, dimension))
            endpoint_values = evaluate(endpoints).reshape(count, 2)
            best_side = np.argmin(endpoint_values, axis=1)
            selected_values = endpoint_values[np.arange(count), best_side]
            success = selected_values < fitness[targets]
            selected_points = endpoints.reshape(count, 2, dimension)[
                np.arange(count), best_side
            ]
            pair_success = success.copy()
            pair_values = selected_values.copy()
            pair_points = selected_points.copy()
            third_ids = np.empty(0, dtype=np.int64)
            third_points = np.empty((0, dimension))
            third_values = np.empty(0)
            third_better = np.empty(0, dtype=bool)
            extra = 0
            eligible = 0
            improved_by_third = 0
            if third_mode != "none" or state_observer is not None:
                ids, third_points = proposals(centers, delta, parent_values, endpoint_values,
                                               lower, span, third_mode, third_rng, upper=box[:, 1])
                eligible = len(ids)
                if state_observer is not None:
                    state_observer(dict(centers=centers.copy(), delta=delta.copy(),
                                        parent=parent_values.copy(), values=endpoint_values.copy(),
                                        eligible=ids.copy()))
                if third_mode != "none":
                    extra = min(len(ids), budget-nfev, int(np.floor(third_fraction*count)))
                    ids, third_points = ids[:extra], third_points[:extra]
                    third_ids = ids.copy()
                    if extra:
                        third_values = evaluate(third_points)
                        better = third_values < np.minimum(parent_values[ids], selected_values[ids])
                        improved_by_third = int(better.sum())
                        third_better = better.copy()
                        if trace_observer is not None:
                            emit_trace("third_ready", probe_trace_details(), event)
                        selected_points[ids[better]] = third_points[better]
                        selected_values[ids[better]] = third_values[better]
                        success = selected_values < parent_values
            stats["third_evals"] += extra
            stats["third_improvements"] += improved_by_third
            if event is not None:
                event.update(probe_targets=count, third_eligible=eligible,
                             third_evals=extra, third_improvements=improved_by_third)
            population[targets[success]] = selected_points[success]
            fitness[targets[success]] = selected_values[success]
            radius_success = pair_success if radius_feedback == "pair" else success
            if adapt_radius:
                radii[targets] = np.clip(
                    radii[targets] * np.exp(0.7 * (radius_success.astype(float) - 0.5)),
                    1e-12,
                    radius_max,
                )
            if event is not None:
                event.update(targets=targets.tolist(), radius_before=radius_before.tolist(),
                             radius_after=radii[targets].tolist(), radius_feedback=radius_feedback,
                             adapt_radius=bool(adapt_radius), parent_values=parent_values.tolist(),
                             pair_values=pair_values.tolist(), selected_values=selected_values.tolist(),
                             pair_success=pair_success.tolist(), macro_success=success.tolist(),
                             radius_success=radius_success.tolist(),
                             rescued_count=int((success & ~pair_success).sum()))
            evaluations, arm = 2 * count + extra, 1
            stats["probe_evals"] += evaluations
            stats["probe_successes"] += int(success.sum())
            if trace_observer is not None:
                details = probe_trace_details()
        else:
            count = min(n, remaining)
            targets = (
                all_indices if count == n else rng.choice(n, count, replace=False)
            )
            elite = np.argpartition(fitness, elite_size - 1)[:elite_size]
            elite_position = np.full(n, -1, dtype=int)
            elite_position[elite] = np.arange(elite_size)
            excluded_position = elite_position[targets]
            slots = rng.integers(0, elite_size - (excluded_position >= 0))
            slots += (excluded_position >= 0) & (slots >= excluded_position)
            leaders = elite[slots]
            first = _sample_excluding(rng, n, np.column_stack((targets, leaders)))
            second = _sample_excluding(
                rng, n, np.column_stack((targets, leaders, first))
            )
            factor = rng.uniform(0.4, 0.9, size=(count, 1))
            trial = population[targets] + factor * (
                population[leaders] - population[targets]
                + population[first] - population[second]
            )
            crossover = rng.random((count, dimension)) < 0.9
            crossover[np.arange(count), rng.integers(dimension, size=count)] = True
            trial = _reflect(np.where(crossover, trial, population[targets]))
            trial_values = evaluate(trial)
            success = trial_values < fitness[targets]
            population[targets[success]] = trial[success]
            fitness[targets[success]] = trial_values[success]
            evaluations, arm = count, 0
            stats["sweep_evals"] += evaluations
            stats["sweep_successes"] += int(success.sum())
            if trace_observer is not None:
                details = dict(targets=targets, elite=elite, leaders=leaders,
                               first=first, second=second, factor=factor,
                               crossover=crossover, trial=trial,
                               trial_values=trial_values, success=success)

        scores[arm] = 0.8 * scores[arm] + 0.2 * float(success.sum()) / evaluations
        last_updated[arm] = nfev
        if event is not None:
            event.update({"arm": "S" if arm == 0 else "P", "nfev_after": nfev,
                          "evaluations": evaluations, "successes": int(success.sum()),
                          "reward": float(success.sum()) / evaluations,
                          "q_s_after": float(scores[0]), "q_p_after": float(scores[1]),
                          "best_after": best_value, "best_gain": event["best_before"]-best_value})
            if observer is not None:
                observer(event)
        if trace_observer is not None:
            emit_trace("post", details, event)

    stats["sweep_probability"] = float(sweep_probability)
    stats["radius_min"] = float(radii.min())
    stats["radius_median"] = float(np.median(radii))
    stats["radius_max"] = float(radii.max())
    assert best_x is not None
    return OptimizeResult(
        x=best_x,
        fun=best_value,
        nfev=nfev,
        history_evals=np.asarray(history_evals, dtype=np.int64),
        history_best=np.asarray(history_best, dtype=float),
        stats=stats,
    )
