"""Curve-factor Greeks and covariance-weighted incomplete-market projection."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from .model import _psd_root


@dataclass(frozen=True)
class Instrument:
    name: str
    market: str
    delivery: float
    multiplier: float
    position_limit: float
    trade_limit: float
    impact_depth: float
    spread: float
    impact_slope: float
    direction: str='both'
    margin_sigma: float=3.
    margin_days: float=10.
    minimum_margin: float=1.
    liquidity_charge_annual: float=0.
    def __post_init__(self):
        vals=[self.delivery,self.multiplier,self.position_limit,self.trade_limit,self.impact_depth,self.spread,self.impact_slope,self.margin_sigma,self.margin_days,self.minimum_margin,self.liquidity_charge_annual]
        if not np.isfinite(vals).all() or min(vals)<0 or self.multiplier<=0 or self.direction not in ('both','long','short','none'):raise ValueError('invalid instrument economics')
    @property
    def maturity(self): return self.delivery
    @property
    def funded(self): return False
    @property
    def event_dates(self): return (self.delivery,)
    def settlement(self,model,t,x): return np.zeros(np.asarray(x).shape[:-1])
    def mark(self,model,t,x):
        if t>self.delivery+1e-10:raise ValueError('instrument expired')
        return model.rate_future(t,x,self.delivery) if self.market=='rate' else model.value(self.market,t,x,self.delivery)
    def delta(self,model,t,x):
        if self.market=='rate':return np.broadcast_to(self.multiplier*model.rate_loading*np.exp(-model.kappa*(self.delivery-t)),np.asarray(x).shape)
        return self.multiplier*model.greeks(self.market,t,x,self.delivery)[0]
    def margin(self,model,t,x):
        if t>=self.delivery-1e-10:return np.zeros(np.asarray(x).shape[:-1])
        d=self.delta(model,t,x);dt=min(self.margin_days/365,self.delivery-t);_,_,C=model.transition_moments(dt,'P')
        return np.maximum(self.minimum_margin,self.margin_sigma*np.sqrt(np.einsum('...i,ij,...j->...',d,C,d)))
    def bounds(self,t):
        if t>=self.delivery-1e-10:return 0.,0.
        return (0. if self.direction in ('long','none') else -self.position_limit,0. if self.direction in ('short','none') else self.position_limit)


def project(delta,instrument_deltas,covariance,rcond=1e-10):
    """Minimize local gain variance. q is positive long; delta is portfolio VALUE."""
    d=np.asarray(delta,float);H=np.asarray(instrument_deltas,float);C=np.asarray(covariance,float)
    if H.ndim!=2 or H.shape[1]!=len(d) or C.shape!=(len(d),len(d)) or not all(np.isfinite(x).all() for x in (d,H,C)):raise ValueError('finite aligned factor exposures required')
    root=_psd_root(C);a=H@root;b=d@root
    q=-np.linalg.pinv(a.T,rcond=rcond)@b
    res=d+H.T@q;G=H@C@H.T
    return {'positions':q,'residual_delta':res,'unhedged_variance':float(d@C@d),'residual_variance':float(max(0,res@C@res)),
            'span_rank':int(np.linalg.matrix_rank(a,tol=rcond*max(1,np.linalg.norm(a,2)))), 'orthogonality_error':float(np.max(np.abs(H@C@res),initial=0)),
            'singular_values':np.linalg.svd(a,compute_uv=False)}


def factor_attribution(delta,covariance):
    """Euler variance contributions sum to variance; correlated entries can be negative."""
    d=np.asarray(delta,float);C=np.asarray(covariance,float)
    return d*(C@d)


def default_instruments(horizon=1.):
    out=[]
    for market in ('G1','G2','power_US'):
        for frac in (.25,.5,1.):
            out.append(Instrument(f'{market} {frac*horizon:.4f}y',market,frac*horizon,
                1000. if market!='power_US' else 25.,500.,200.,70.,2.,.05,
                'short' if market!='power_US' else 'long'))
    for frac in (.5,1.):out.append(Instrument(f'rate {frac*horizon:.4f}y','rate',frac*horizon,100000.,120.,60.,20.,2.,.04,'long'))
    return out


def constrained_project(delta,H,covariance,lower,upper,margin,liquidity_budget,previous=None,trade_limit=None):
    """Local minimum-variance hedge with signed/depth bounds and cash margin.

    This is a covariance diagnostic, not the nonlinear cash-flow controller.
    Positions are scaled before optimization; the returned feasibility errors
    are measured in original currency and contract units.
    """
    from scipy.optimize import minimize,Bounds
    g=np.asarray(delta,float);H=np.asarray(H,float);C=np.asarray(covariance,float)
    lo=np.asarray(lower,float).copy();hi=np.asarray(upper,float).copy();im=np.asarray(margin,float)
    n=len(H);old=np.zeros(n) if previous is None else np.asarray(previous,float)
    if H.shape!=(n,len(g)) or C.shape!=(len(g),len(g)) or any(a.shape!=(n,) for a in (lo,hi,im,old)):
        raise ValueError('projection dimension mismatch')
    if liquidity_budget<0 or np.any(im<0) or not all(np.isfinite(a).all() for a in (g,H,C,lo,hi,im,old)):
        raise ValueError('invalid projection inputs')
    if trade_limit is not None:
        depth=np.asarray(trade_limit,float)
        if depth.shape!=(n,) or np.any(depth<0):raise ValueError('invalid turnover limits')
        lo=np.maximum(lo,old-depth);hi=np.minimum(hi,old+depth)
    if np.any(lo>hi):raise ValueError('inconsistent position/turnover bounds')
    scale=np.maximum(1,np.maximum(abs(lo),abs(hi)));Hs=H*scale[:,None]
    base=max(float(g@C@g),1.)
    def fun(z):
        r=g+Hs.T@z;return float(r@C@r/base)
    def jac(z):return 2*Hs@C@(g+Hs.T@z)/base
    budgetscale=max(liquidity_budget,1.)
    constraint={'type':'ineq','fun':lambda z:(liquidity_budget-im@abs(z*scale))/budgetscale,
                'jac':lambda z:-im*scale*np.sign(z)/budgetscale}
    z0=np.clip(np.zeros(n),lo/scale,hi/scale)
    r=minimize(fun,z0,jac=jac,bounds=Bounds(lo/scale,hi/scale),constraints=[constraint],method='SLSQP',
        options={'ftol':1e-11,'maxiter':2000})
    q=r.x*scale;res=g+H.T@q
    violation=max(0.,float(np.max(lo-q)),float(np.max(q-hi)),float(im@abs(q)-liquidity_budget))
    if not r.success or violation>1e-3:raise RuntimeError(f'constrained hedge projection failed: {r.message}, violation={violation}')
    return {'positions':q,'residual_delta':res,'unhedged_variance':float(g@C@g),'residual_variance':float(res@C@res),
            'initial_margin':float(im@abs(q)),'liquidity_budget':float(liquidity_budget),'maximum_constraint_violation':violation,
            'iterations':r.nit,'solver':'SLSQP convex quadratic objective; signed bounds, turnover and absolute margin','success':True}
