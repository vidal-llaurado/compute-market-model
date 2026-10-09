"""Event-time financial accounting shared by valuation and execution."""
from __future__ import annotations
import numpy as np
from .model import CurveModel


def reference_model(model,basis):
    """Persist today's observable benchmark dislocation in conditional marks."""
    basis=np.asarray(basis,float)
    if basis.shape!=(2,) or not np.isfinite(basis).all() or np.any(basis<=0):
        raise ValueError('two positive benchmark basis multipliers required')
    if np.array_equal(basis,np.ones(2)): return model
    key=('benchmark-basis',float(basis[0]),float(basis[1]))
    if key not in model._bond_cache:
        data=model.to_dict()
        for curve in data['curves']:
            if curve['name'] in ('G1','G2'):
                j=0 if curve['name']=='G1' else 1
                curve['values']=(np.asarray(curve['values'])*basis[j]).tolist()
        model._bond_cache[key]=CurveModel.from_dict(data)
    return model._bond_cache[key]


def marks(instruments,model,t,x):
    return np.stack([np.zeros(np.asarray(x).shape[:-1]) if t>h.maturity+1e-9
                     else h.mark(model,t,x) for h in instruments],axis=-1)


def funded_prices(instruments,current_marks):
    return np.asarray(current_marks)*np.array([h.multiplier if h.funded else 0. for h in instruments])


def settle(instruments,model,previous_time,t,x,old,previous_marks,*,counterparty_alive=True):
    """Settle due cash and return surviving inventory before new decisions.

    Coupons and exercise pay once. Funded mark changes remain unrealized until
    sale; the cash ledger releases segregated collateral separately."""
    old=np.asarray(old,float).copy();previous_marks=np.asarray(previous_marks,float)
    if old.shape!=(len(instruments),) or previous_marks.shape!=old.shape or not t>previous_time:
        raise ValueError('aligned positions and increasing settlement dates required')
    current=marks(instruments,model,t,x);cash=0.;default_loss=0.
    for k,h in enumerate(instruments):
        if previous_time>=h.maturity-1e-9:
            if abs(old[k])>1e-8:raise ValueError('inventory survived beyond its maturity')
            continue
        if h.funded:
            due=float(h.settlement(model,t,x))
            if getattr(h,'counterparty','cleared')!='cleared' and not counterparty_alive:
                clean=old[k]*h.multiplier*(current[k]+due)
                recovery=clean if clean<=0 else h.recovery*clean
                cash+=recovery;default_loss+=clean-recovery;old[k]=0.
            else:cash+=old[k]*h.multiplier*due
        else:
            if t>h.maturity+1e-9:raise ValueError('event grid skipped a futures maturity')
            cash+=old[k]*h.multiplier*(current[k]-previous_marks[k])
        if t>=h.maturity-1e-9:old[k]=0.
    return {'cash':float(cash),'positions':old,'marks':current,'credit_loss':float(default_loss)}


def conditional_gains(instruments,model,t,x,next_time,next_states,current_marks=None):
    """Economic one-step gains, distinct from same-step cash settlement."""
    now=marks(instruments,model,t,x) if current_marks is None else np.asarray(current_marks,float);H=np.zeros((len(next_states),len(instruments)))
    for k,h in enumerate(instruments):
        if t>=h.maturity-1e-9:continue
        if next_time>h.maturity+1e-9:raise ValueError('conditional step skips an instrument maturity')
        nxt=h.mark(model,next_time,next_states)
        coupon=h.settlement(model,next_time,next_states) if h.funded else 0.
        H[:,k]=h.multiplier*(nxt+coupon-now[k])
    return H
