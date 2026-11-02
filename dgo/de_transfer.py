"""Fixed synchronous DE/rand/1/bin with exactly one diagnostic probe batch.

This module imports no DGO optimizer and performs no evaluation on import.
The interpolation helper is the frozen, read-only v0.3 mathematical rule.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .interpolation import proposals

STATE_KEYS={"population","fitness","nfev","best_value","best_x","history_evals","history_best",
            "rng_state","probe_rng_state","generation","probe_done","stats"}
STAT_KEYS={"initial_evals","de_evals","de_successes","probe_evals","probe_pair_evals","probe_third_evals",
           "probe_successes","potential_third_improvements","committed_third_candidates","suppressed_commits","probe_batches"}


@dataclass(frozen=True)
class OptimizeResult:
    x:np.ndarray
    fun:float
    nfev:int
    history_evals:np.ndarray
    history_best:np.ndarray
    stats:dict
    final_state:dict


def reflect(points):return 1.-np.abs(1.-np.mod(points,2.))


def validate_state(state,box,budget):
    """Reject invalid settled resume states before any objective invocation."""
    if not isinstance(state,dict) or set(state)!=STATE_KEYS:raise ValueError("resume state schema")
    n,d=40,len(box);t=40*int(np.floor(.3*budget/40))
    population=np.asarray(state["population"]);fitness=np.asarray(state["fitness"])
    if population.shape!=(n,d) or fitness.shape!=(n,):raise ValueError("resume population/fitness shape")
    if not np.isfinite(population).all() or not np.isfinite(fitness).all() or np.any((population<0)|(population>1)):
        raise ValueError("resume population/fitness finite/box")
    stats=state["stats"]
    if not isinstance(stats,dict) or set(stats)!=STAT_KEYS:raise ValueError("resume statistics schema")
    if any(not isinstance(v,Integral) or isinstance(v,bool) or v<0 for v in stats.values()):raise ValueError("resume integer counters")
    if stats["initial_evals"]!=40 or stats["probe_pair_evals"]!=40 or stats["probe_batches"]!=1:
        raise ValueError("resume initialization/probe identity")
    if stats["probe_evals"]!=stats["probe_pair_evals"]+stats["probe_third_evals"] or not 0<=stats["probe_third_evals"]<=20:
        raise ValueError("resume probe accounting")
    if not isinstance(state["nfev"],Integral) or not t+40<=state["nfev"]<=budget:
        raise ValueError("resume paid FE")
    if 40+stats["de_evals"]+stats["probe_evals"]!=state["nfev"]:raise ValueError("resume cost conservation")
    if not isinstance(state["generation"],Integral) or state["generation"]!=(stats["de_evals"]+39)//40:
        raise ValueError("resume generation count")
    if state["probe_done"] is not True:raise ValueError("resume must follow complete diagnostic probe")
    if stats["committed_third_candidates"]+stats["suppressed_commits"]!=stats["potential_third_improvements"]:
        raise ValueError("resume potential/commit accounting")
    if not (0<=stats["de_successes"]<=stats["de_evals"] and 0<=stats["probe_successes"]<=20):
        raise ValueError("resume acceptance accounting")
    if stats["potential_third_improvements"]>stats["probe_third_evals"] or stats["suppressed_commits"]>1:
        raise ValueError("resume single suppression")
    best=float(state["best_value"]);point=np.asarray(state["best_x"])
    if not np.isfinite(best) or best>float(fitness.min()) or point.shape!=(d,) or not np.isfinite(point).all():
        raise ValueError("resume historical best")
    if np.any((point<box[:,0])|(point>box[:,1])):raise ValueError("resume historical point box")
    he,hb=np.asarray(state["history_evals"]),np.asarray(state["history_best"])
    if he.ndim!=1 or len(he)==0 or he.shape!=hb.shape or not np.issubdtype(he.dtype,np.integer):
        raise ValueError("resume history schema")
    if not np.isfinite(hb).all() or np.any(np.diff(he)<=0) or np.any(np.diff(hb)>0) or he[-1]!=state["nfev"] or hb[-1]!=best:
        raise ValueError("resume history/clock/best")
    for key in ("rng_state","probe_rng_state"):
        try:
            if state[key]["bit_generator"]!="PCG64":raise ValueError("PCG64 required")
            test=np.random.default_rng(0);test.bit_generator.state=copy.deepcopy(state[key])
        except Exception as error:raise ValueError("resume "+key) from error


def minimize(objective,bounds,*,initial_population,max_evals,seed,probe_seed,resume_state=None,observer=None,trace_observer=None):
    box=np.asarray(bounds,dtype=float)
    if box.ndim!=2 or box.shape[1]!=2 or not np.isfinite(box).all() or np.any(box[:,1]<=box[:,0]):raise ValueError("finite box")
    if not isinstance(max_evals,Integral) or isinstance(max_evals,bool):raise ValueError("integer evaluation budget")
    budget=int(max_evals);dimension=len(box);t=40*int(np.floor(.3*budget/40))
    if t<40 or t+60>budget:raise ValueError("budget must accommodate one complete diagnostic probe")
    initial=np.asarray(initial_population,dtype=float)
    if initial.shape!=(40,dimension) or not np.isfinite(initial).all() or np.any((initial<box[:,0])|(initial>box[:,1])):
        raise ValueError("initial physical population")
    lower,upper=box.T;span=upper-lower
    rng=np.random.default_rng(seed);probe_rng=np.random.default_rng(probe_seed)
    if resume_state is None:
        state=dict(population=(initial-lower)/span,fitness=None,nfev=0,best_value=float("inf"),best_x=None,
                   history_evals=[],history_best=[],rng_state=copy.deepcopy(rng.bit_generator.state),
                   probe_rng_state=copy.deepcopy(probe_rng.bit_generator.state),generation=0,probe_done=False,
                   stats={key:(40 if key=="initial_evals" else 0) for key in STAT_KEYS})
    else:
        validate_state(resume_state,box,budget)
        state=copy.deepcopy(resume_state)
        state["history_evals"],state["history_best"]=list(state["history_evals"]),list(state["history_best"])
        rng.bit_generator.state=copy.deepcopy(state["rng_state"])
        probe_rng.bit_generator.state=copy.deepcopy(state["probe_rng_state"])

    def snapshot():
        state["rng_state"]=copy.deepcopy(rng.bit_generator.state)
        state["probe_rng_state"]=copy.deepcopy(probe_rng.bit_generator.state)
        return copy.deepcopy(state)

    def evaluate(unit):
        count=len(unit)
        if count<1 or state["nfev"]+count>budget:raise ValueError("evaluation budget")
        physical=np.clip(lower+span*unit,lower,upper)
        values=np.asarray(objective(physical),dtype=float)
        if values.shape!=(count,) or not np.isfinite(values).all():raise ValueError("finite vectorized objective values")
        state["nfev"]+=count
        winner=int(np.argmin(values))
        if values[winner]<state["best_value"]:
            state["best_value"]=float(values[winner]);state["best_x"]=physical[winner].copy()
        state["history_evals"].append(state["nfev"]);state["history_best"].append(state["best_value"])
        return values

    def emit(stage,details,event):
        if trace_observer is not None:
            trace_observer(dict(stage=stage,state=snapshot(),details=copy.deepcopy(details),event=copy.deepcopy(event)))

    def finish_event(event):
        event.update(nfev_after=state["nfev"],best_after=state["best_value"],best_gain=event["best_before"]-state["best_value"])
        if observer is not None:observer(copy.deepcopy(event))

    if resume_state is None:state["fitness"]=evaluate(state["population"])
    while state["nfev"]<budget:
        stats=state["stats"];population=state["population"];fitness=state["fitness"]
        before=state["nfev"];best_before=state["best_value"]
        if not state["probe_done"] and before==t:
            targets=probe_rng.choice(40,20,replace=False)
            centers=population[targets].copy();parents=fitness[targets].copy()
            directions=probe_rng.normal(size=(20,dimension));norms=np.linalg.norm(directions,axis=1)
            zero=norms==0;directions[zero,0]=1.;norms[zero]=1.;directions/=norms[:,None]
            radius=.1*np.sqrt(dimension);delta=radius*directions
            endpoints=reflect(np.stack((centers+delta,centers-delta),axis=1).reshape(40,dimension))
            values=evaluate(endpoints).reshape(20,2);side=np.argmin(values,axis=1)
            pair_points=endpoints.reshape(20,2,dimension)[np.arange(20),side].copy()
            pair_values=values[np.arange(20),side].copy();pair_success=pair_values<parents
            ids,third_points=proposals(centers,delta,parents,values,lower,span,"interpolate",probe_rng,upper=upper)
            extra=len(ids)
            third_values=evaluate(third_points) if extra else np.empty(0)
            better=third_values<np.minimum(parents[ids],pair_values[ids])
            selected_points,selected_values=pair_points.copy(),pair_values.copy()
            base_event=dict(kind="probe",generation=int(state["generation"]),nfev_before=before,best_before=best_before)
            details=dict(targets=targets,centers=centers,parent_values=parents,directions=directions,delta=delta,
                         endpoints=endpoints,endpoint_values=values,pair_points=pair_points,pair_values=pair_values,
                         pair_success=pair_success,third_ids=ids,third_points=third_points,third_values=third_values,
                         third_better=better,selected_points=selected_points.copy(),selected_values=selected_values.copy(),
                         success=pair_success.copy(),radius_before=np.full(20,radius),count=20,extra=extra,eligible=extra,
                         improved_by_third=int(better.sum()),best_side=side)
            if extra:emit("third_ready",details,base_event)
            selected_points[ids[better]],selected_values[ids[better]]=third_points[better],third_values[better]
            success=selected_values<parents
            population[targets[success]],fitness[targets[success]]=selected_points[success],selected_values[success]
            stats["probe_evals"]+=40+extra;stats["probe_pair_evals"]+=40;stats["probe_third_evals"]+=extra
            stats["probe_successes"]+=int(success.sum());stats["potential_third_improvements"]+=int(better.sum())
            stats["committed_third_candidates"]+=int(better.sum());stats["probe_batches"]+=1;state["probe_done"]=True
            event=dict(base_event,targets=targets.tolist(),probe_targets=20,radius=float(radius),third_eligible=extra,
                       pair_evals=40,third_evals=extra,third_improvements=int(better.sum()),pair_success=pair_success.tolist(),
                       third_ids=ids.tolist(),third_values=third_values.tolist(),third_better=better.tolist(),
                       selected_values=selected_values.tolist(),succeeded=success.tolist(),successes=int(success.sum()),evaluations=40+extra)
            finish_event(event)
            details.update(selected_points=selected_points.copy(),selected_values=selected_values.copy(),success=success.copy())
            emit("post",details,event)
        else:
            if not state["probe_done"] and before>t:raise ValueError("missed diagnostic probe boundary")
            count=min(40,budget-before,t-before if not state["probe_done"] else budget-before)
            if count<=0:raise ValueError("DE complete-generation/probe clock")
            targets=np.arange(count);parents=population.copy();parent_values=fitness[targets].copy()
            donors=np.empty((count,3),dtype=np.int64)
            for row,target in enumerate(targets):
                allowed=np.concatenate((np.arange(target),np.arange(target+1,40)))
                donors[row]=rng.choice(allowed,3,replace=False)
            mutant=parents[donors[:,0]]+.5*(parents[donors[:,1]]-parents[donors[:,2]])
            crossover=rng.random((count,dimension))<.9;jrand=rng.integers(dimension,size=count)
            crossover[np.arange(count),jrand]=True
            trial=reflect(np.where(crossover,mutant,parents[targets]))
            state["generation"]+=1;values=evaluate(trial);success=values<parent_values
            population[targets[success]],fitness[targets[success]]=trial[success],values[success]
            stats["de_evals"]+=count;stats["de_successes"]+=int(success.sum())
            event=dict(kind="de",generation=int(state["generation"]),nfev_before=before,targets=targets.tolist(),
                       donors=donors.tolist(),crossover=crossover.tolist(),jrand=jrand.tolist(),parent_values=parent_values.tolist(),
                       trial_values=values.tolist(),succeeded=success.tolist(),successes=int(success.sum()),evaluations=count,best_before=best_before)
            finish_event(event)
        assert 40+stats["de_evals"]+stats["probe_evals"]==state["nfev"]
    final=snapshot();validate_state(final,box,budget)
    return OptimizeResult(x=state["best_x"].copy(),fun=float(state["best_value"]),nfev=int(state["nfev"]),
                          history_evals=np.asarray(state["history_evals"],dtype=np.int64),history_best=np.asarray(state["history_best"]),
                          stats=copy.deepcopy(state["stats"]),final_state=final)
