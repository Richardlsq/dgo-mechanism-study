"""Recompute BX1 and matched-start PC1 ratios/intervals from saved results.
No objective calls, optimizer runs, network access, or changes to result files.
"""
from pathlib import Path
import json, csv, gzip
import numpy as np
DATA=Path(__file__).resolve().parent/'results'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def csvread(p):
    with p.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def read_runs(p):
    with gzip.open(p,'rt',encoding='utf-8') as f: return [json.loads(s) for s in f]
def interval(x): return np.quantile(x,[.025,.975]).tolist()
def dump(name, rows):
    expected=read(DATA/'supplement'/name.lower())
    assert len(rows)==len(expected)
    for a,b in zip(rows,expected):
        for key in ['ratio','shared_seed_interval','seed_specific_ratios','worst_pair']:
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-14)
    print(name+': '+str(len(rows))+' comparisons reproduced')
    for row in rows:
        name=row.get('reference',row.get('denominator'))
        print(f"  {row['dimension']}D {name}: ratio={row['ratio']:.6g}, interval={row['shared_seed_interval']}")
def savecsv(name,rows): pass

# BX1 primary ratios, same global seed indices for every function/instance.
bp=DATA/'BX1';br=csvread(bp/'endpoints.csv');bi={(int(x['dimension']),int(x['function']),int(x['instance']),int(x['seed']),x['method']):x for x in br};assert len(bi)==len(br)==1728
bs=sorted({int(x['seed']) for x in br});refs=['DE5D','PSO','Random','L','C'];old=read(bp/'results.json');bx=[];rng=np.random.default_rng(2026103001)
for d in [10,20]:
 seedidx=rng.integers(0,2,(2000,2));instidx=rng.integers(0,3,(2000,24,3))
 for m in refs:
  a=np.array([[[float(bi[d,f,i,se,'JQ100']['error']) for se in bs] for i in [1,2,3]] for f in range(1,25)])
  b=np.array([[[float(bi[d,f,i,se,m]['error']) for se in bs] for i in [1,2,3]] for f in range(1,25)])
  assert np.isfinite(a).all() and np.isfinite(b).all() and (a>=0).all() and (b>=0).all()
  logs=np.log(np.maximum(a,.01))-np.log(np.maximum(b,.01));point=float(np.exp(logs.mean()))
  ref=next(x for x in old['comparisons'] if x['dimension']==d and x['reference']==m);assert np.isclose(point,ref['primary_geometric_ratio'],rtol=1e-13)
  boot=np.exp(logs[np.arange(24)[None,:,None,None],instidx[:,:,:,None],seedidx[:,None,None,:]].mean(axis=(1,2,3)))
  bx.append(dict(dimension=d,reference=m,ratio=point,old_interval=ref['bootstrap95'],shared_seed_interval=interval(boot),seed_specific_ratios=np.exp(logs.mean(axis=(0,1))).tolist(),worst_pair=float(np.exp(logs.max()))))
dump('BX1_BLOCKS.json',bx);print('BX1 shared-seed analysis complete',flush=True)
# PC1: fixed six-family weights, paired crossed instance/seed resampling.
pp=DATA/'PC1';cfg=read(pp/'config.json');pr=read_runs(pp/'runs.jsonl.gz');pi={(r['dimension'],r['profile']['id'],r['instance_block'],r['seed'],r['method']):r for r in pr};assert len(pi)==len(pr)==1296
ps=sorted({r['seed'] for r in pr});profiles=sorted({r['profile']['id'] for r in pr});fam={p:next(r['profile']['family'] for r in pr if r['profile']['id']==p) for p in profiles};families=sorted(set(fam.values()));weights=np.array([1/(6*sum(fam[q]==fam[p] for q in profiles)) for p in profiles]);assert np.isclose(weights.sum(),1)
oldpc=read(pp/'results.json');pcs=[];pfs=[];rng=np.random.default_rng(2026103002)
for d in [10,20]:
 ss=rng.integers(0,3,(2000,3));ii=rng.integers(0,3,(2000,len(profiles),3));nested_ss=rng.integers(0,3,(2000,len(profiles),3,3))
 for a,b in [('JQ100','DE40'),('JQ100','DE5D'),('DE40','DE5D')]:
  av=np.array([[[pi[d,p,i,se,a]['fun'] for se in ps] for i in [1,2,3]] for p in profiles]);bv=np.array([[[pi[d,p,i,se,b]['fun'] for se in ps] for i in [1,2,3]] for p in profiles]);targets=np.array([pi[d,p,1,ps[0],a]['profile']['target'] for p in profiles]);logs=np.log(np.maximum(av,targets[:,None,None]))-np.log(np.maximum(bv,targets[:,None,None]));point=float(np.exp(weights@logs.mean(axis=(1,2))))
  oldr=next(x for x in oldpc['comparisons'] if x['dimension']==d and x['numerator']==a and x['denominator']==b);assert np.isclose(point,oldr['ratios']['primary']['macro_ratio'],rtol=1e-13)
  z=logs[np.arange(len(profiles))[None,:,None,None],ii[:,:,:,None],ss[:,None,None,:]].mean(axis=(2,3))@weights
  zn=logs[np.arange(len(profiles))[None,:,None,None],ii[:,:,:,None],nested_ss].mean(axis=(2,3))@weights
  pcs.append(dict(dimension=d,numerator=a,denominator=b,ratio=point,shared_seed_interval=interval(np.exp(z)),nested_interval=interval(np.exp(zn)),seed_specific_ratios=np.exp(weights@logs.mean(axis=1)).tolist(),worst_pair=float(np.exp(logs.max()))))
  for f in families:
   ix=[j for j,p in enumerate(profiles) if fam[p]==f];pfs.append(dict(dimension=d,numerator=a,denominator=b,family=f,ratio=float(np.exp(logs[ix].mean())),hits_numerator=int(np.sum(av[ix]<=targets[ix,None,None])),hits_denominator=int(np.sum(bv[ix]<=targets[ix,None,None])),runs=9*len(ix)))
dump('PC1_INTERVALS.json',pcs);savecsv('PC1_FAMILIES.csv',pfs);print('PC1 intervals and all-family summaries complete',flush=True)
