"""Export one claim registry to the finite-state settlement and collateral engine."""
from __future__ import annotations
import numpy as np
from ..hedge_market import blank_market, NettingSet


def export_claims(model,instruments,tree,times,states):
    times=np.asarray(times,float);X=np.asarray(states,float)
    if times.shape!=(tree.size,) or X.shape!=(tree.size,model.dimension):raise ValueError('node-aligned claim states required')
    grid=np.unique(times)
    if any(not any(abs(T-grid)<1e-9) for h in instruments for T in h.event_dates):
        raise ValueError('finite graph must include every cash event')
    kinds=['futures' if not h.funded else 'otc' if getattr(h,'counterparty','cleared')!='cleared' else 'cash' for h in instruments]
    h=blank_market(tree,[z.name for z in instruments],kinds)
    groups=sorted(set(getattr(z,'counterparty','cleared') for z,k in zip(instruments,kinds) if k=='otc'))
    h.groups=[getattr(z,'counterparty','') if k=='otc' else '' for z,k in zip(instruments,kinds)]
    h.netting_sets=[]
    for name in groups:
        contracts=[z for z,k in zip(instruments,kinds) if k=='otc' and z.counterparty==name]
        recoveries={z.recovery for z in contracts}
        if len(recoveries)!=1:raise ValueError('one recovery convention per netting set required')
        h.netting_sets.append(NettingSet(name,recovery=recoveries.pop()))
    h.default=np.zeros((tree.size,len(groups)));h.initial_posted_vm=np.zeros(len(groups));h.initial_received_vm=np.zeros(len(groups))
    h.multiplier=np.array([z.multiplier for z in instruments])
    for t in grid:
        ix=np.flatnonzero(abs(times-t)<1e-10)
        for j,z in enumerate(instruments):
            if t<=z.maturity+1e-9:
                h.mark[ix,j]=z.mark(model,float(t),X[ix])
                h.distribution[ix,j]=z.settlement(model,float(t),X[ix])
                h.im_per_unit[ix,j]=z.margin(model,float(t),X[ix])
            lo,hi=z.bounds(t);h.lower[ix,j]=lo;h.upper[ix,j]=hi
            h.survives[ix,j]=float(t<z.maturity-1e-9)
    h.initial_mark=h.mark[0].copy();h.trade_limit[:]=[z.trade_limit for z in instruments]
    h.half_spread[:]=[z.spread for z in instruments]
    h.impact_depth[:]=[z.impact_depth for z in instruments];h.impact_slope[:]=[z.impact_slope for z in instruments]
    return h.validate(tree)
