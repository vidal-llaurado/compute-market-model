"""Deterministic independent-path execution in ordered process chunks."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
import numpy as np
from .hedge_paths import FinancialTape,HedgePathResult,simulate_hedges
from .simulation import MarketTape


def subset_tape(tape,indices):
    ids=np.asarray(indices,int)
    if ids.ndim!=1 or not len(ids):raise ValueError('nonempty path indices required')
    raw={name:(value[ids].copy() if isinstance(value,np.ndarray) else value)
        for name,value in vars(tape.physical).items()}
    raw['config']=replace(tape.physical.config,paths=len(ids))
    financial={name:(value[ids].copy() if isinstance(value,np.ndarray) else value)
        for name,value in vars(tape).items() if name!='physical'}
    return FinancialTape(MarketTape(**raw),**financial).validate()


def _worker(args):
    tape,kernel,policy,kwargs=args
    return simulate_hedges(tape,kernel,policy,**kwargs)


def simulate_parallel(tape,kernel,policy,workers=1,**kwargs):
    if not isinstance(workers,int) or workers<1:raise ValueError('positive worker count required')
    if workers==1:return simulate_hedges(tape,kernel,policy,**kwargs)
    chunks=np.array_split(np.arange(tape.physical.config.paths),min(workers,tape.physical.config.paths))
    args=[(subset_tape(tape,ids),kernel,policy,kwargs) for ids in chunks]
    with ProcessPoolExecutor(max_workers=len(chunks),mp_context=mp.get_context('spawn')) as pool:
        pieces=list(pool.map(_worker,args))
    fields={}
    sumkeys={'conditional_optimizations','joint_optimizations','joint_incremental_draws','joint_incremental_repays','joint_spot_block_intervals'}
    for name,value in vars(pieces[0]).items():
        if isinstance(value,np.ndarray):fields[name]=np.concatenate([getattr(r,name) for r in pieces])
        elif name=='failure_reason':fields[name]=sum([r.failure_reason for r in pieces],[])
        elif name=='audit':
            audit={}
            for key,val in value.items():
                if key.startswith('max_'):audit[key]=max(r.audit[key] for r in pieces)
                elif key in sumkeys:audit[key]=sum(r.audit[key] for r in pieces)
                elif key=='all_paths_terminated':audit[key]=all(r.audit[key] for r in pieces)
                else:audit[key]=val
            fields[name]=audit
        else:fields[name]=value
    return HedgePathResult(**fields)
