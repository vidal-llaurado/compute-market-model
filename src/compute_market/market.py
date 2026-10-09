"""Market observations, demand choice, forward identification and hedge diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear, linprog
from scipy.special import logsumexp


@dataclass
class RidgeLogPrice:
    coefficients: np.ndarray
    ridge: float
    residual_std: float

    @classmethod
    def fit(cls, design, prices, weights=None, ridge: float=1., penalize_intercept=False):
        x,y=np.asarray(design,float),np.asarray(prices,float)
        if x.ndim != 2 or x.shape[1] == 0 or y.shape != (x.shape[0],) or x.shape[0] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0) or ridge < 0:
            raise ValueError("finite design, positive prices and nonnegative ridge required")
        w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
        if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("positive finite observation weights required")
        if not penalize_intercept and not np.allclose(x[:,0],1):
            raise ValueError("first design column must be the intercept when it is unpenalized")
        penalty=np.eye(x.shape[1])*np.sqrt(ridge)
        if not penalize_intercept: penalty[0,0]=0
        a=np.vstack([x*np.sqrt(w)[:,None],penalty]); b=np.r_[np.log(y)*np.sqrt(w),np.zeros(x.shape[1])]
        coef=np.linalg.lstsq(a,b,rcond=None)[0]
        residual=np.log(y)-x@coef
        return cls(coef,ridge,float(np.sqrt(np.average(residual**2,weights=w))))

    def median_price(self, design):
        x=np.asarray(design,float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients) or not np.isfinite(x).all():
            raise ValueError("finite compatible design required")
        result=np.exp(x@self.coefficients)
        if not np.isfinite(result).all(): raise ValueError("price prediction overflow; review inputs")
        return result


def fit_delivery_curve(weights, strip_prices, adjustments=None, observation_weights=None,
                       smoothing: float=.1, floor: float=0.) -> dict:
    w,k=np.asarray(weights,float),np.asarray(strip_prices,float)
    if w.ndim != 2 or not w.shape[0] or not w.shape[1] or k.shape != (w.shape[0],) or not np.isfinite(w).all() or not np.isfinite(k).all() or np.any(w < 0) or np.any(w.sum(axis=1) <= 0) or smoothing < 0 or not np.isfinite(floor):
        raise ValueError("valid delivery weights, strip prices and regularization required")
    a=np.zeros_like(k) if adjustments is None else np.asarray(adjustments,float)
    obs=np.ones_like(k) if observation_weights is None else np.asarray(observation_weights,float)
    if a.shape != k.shape or obs.shape != k.shape or not np.isfinite(a).all() or not np.isfinite(obs).all() or np.any(obs <= 0):
        raise ValueError("finite adjustments and positive observation weights required")
    second=np.diff(np.eye(w.shape[1]),n=2,axis=0)
    matrix=np.vstack([w*np.sqrt(obs)[:,None],np.sqrt(smoothing)*second])
    target=np.r_[(k-a)*np.sqrt(obs),np.zeros(second.shape[0])]
    result=lsq_linear(matrix,target,bounds=(floor,np.inf),tol=1e-12,lsmr_tol=1e-12)
    if not result.success: raise RuntimeError(result.message)
    rank=int(np.linalg.matrix_rank(w))
    return {"forwards":result.x.tolist(),"fitted_strips":(w@result.x+a).tolist(),
            "residuals":(w@result.x+a-k).tolist(),"observation_rank":rank,
            "unidentified_dimensions":w.shape[1]-rank,
            "label":"inferred delivery marks; not executable financial forwards"}


def state_price_bounds(terminal_payoffs, instrument_prices, target_payoff,
                       discount_factor: float=1.) -> dict:
    """Finite-state no-arbitrage closure bounds with nonnegative state prices.

    terminal_payoffs has shape [scenarios, instruments], and instrument_prices
    excludes the cash numeraire. Strictly positive state prices are not enforced;
    returned extrema are closure bounds, not guaranteed executable quotes.
    """
    h,q,g=np.asarray(terminal_payoffs,float),np.asarray(instrument_prices,float),np.asarray(target_payoff,float)
    if h.ndim != 2 or q.shape != (h.shape[1],) or g.shape != (h.shape[0],) or not np.isfinite(h).all() or not np.isfinite(q).all() or not np.isfinite(g).all() or not np.isfinite(discount_factor) or discount_factor <= 0:
        raise ValueError("invalid payoff system")
    eq=np.vstack([np.ones(h.shape[0]),h.T]); rhs=np.r_[discount_factor,q]
    low=linprog(g,A_eq=eq,b_eq=rhs,bounds=(0,None),method="highs")
    high=linprog(-g,A_eq=eq,b_eq=rhs,bounds=(0,None),method="highs")
    if not low.success or not high.success:
        return {"consistent":False,"reason":"no nonnegative state-price system for supplied marks"}
    return {"consistent":True,"lower":float(low.fun),"upper":float(-high.fun),
            "lower_state_prices":low.x.tolist(),"upper_state_prices":high.x.tolist(),
            "label":"finite-state closure bounds; not a unique market price"}


def minimum_variance_hedge(losses, unit_payoffs, ridge: float=0.) -> dict:
    l,h=np.asarray(losses,float),np.asarray(unit_payoffs,float)
    if l.ndim != 1 or h.ndim != 2 or h.shape[0] != l.size or l.size < 2 or not np.isfinite(l).all() or not np.isfinite(h).all() or ridge < 0:
        raise ValueError("finite samples and nonnegative ridge required")
    lc=l-l.mean(); hc=h-h.mean(axis=0)
    covariance=hc.T@hc/(len(l)-1); cross=hc.T@lc/(len(l)-1)
    positions=np.linalg.lstsq(covariance+ridge*np.eye(h.shape[1]),cross,rcond=None)[0]
    residual=l-h@positions
    return {"long_positions":positions.tolist(),"unhedged_variance":float(np.var(l,ddof=1)),
            "residual_variance":float(np.var(residual,ddof=1)),
            "label":"diagnostic only; ignores financing, limits and tail states"}


def logit_choice(prices, performance, preference, sensitivity: float, eligible=None):
    """One workload, outside option utility zero; performance in useful units/GPU-hour."""
    p,v,d=np.asarray(prices,float),np.asarray(performance,float),np.asarray(preference,float)
    if p.ndim != 1 or v.shape != p.shape or d.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(d).all() or np.any(v <= 0) or sensitivity <= 0:
        raise ValueError("matching finite vectors, positive performance and sensitivity required")
    e=np.ones(len(p),bool) if eligible is None else np.asarray(eligible,bool)
    if e.shape != p.shape: raise ValueError("eligibility shape mismatch")
    utility=np.where(e,d-sensitivity*p/v,-np.inf)
    all_u=np.r_[0.,utility]
    probs=np.exp(all_u-logsumexp(all_u))
    return {"outside":float(probs[0]),"products":probs[1:].tolist()}
