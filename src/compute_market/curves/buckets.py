"""Calendar-delivery bucket risk and its projection onto factor space.

Piecewise-linear tents shift positive curves in log price and normal/rate
curves in absolute units. Rebuilding the initial condition preserves pricing
identities. Independent bucket shocks expose shapes outside the factor span."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Sequence
import numpy as np
from .model import CurveModel

@dataclass(frozen=True)
class CurveBucket:
    market: str
    left: float
    center: float
    right: float
    bump: float = 1e-4
    def __post_init__(self):
        if (not np.isfinite([self.left, self.center, self.right, self.bump]).all()
                or not 0 <= self.left <= self.center <= self.right
                or self.left == self.right or self.bump <= 0):
            raise ValueError('ordered bucket support and positive bump required')
    @property
    def name(self):
        return f'{self.market}:{self.center:g}y'
    def weights(self, dates):
        T = np.asarray(dates, float)
        rise = np.ones_like(T) if self.left == self.center else (T-self.left)/(self.center-self.left)
        fall = np.ones_like(T) if self.right == self.center else (self.right-T)/(self.right-self.center)
        return np.where((T >= self.left) & (T <= self.right),
                        np.maximum(0., np.minimum(1., np.minimum(rise, fall))), 0.)

def key_delivery_buckets(market, dates, bump=1e-4):
    dates = np.asarray(dates, float)
    if dates.ndim != 1 or len(dates) < 2 or np.any(np.diff(dates) <= 0):
        raise ValueError('at least two increasing delivery nodes required')
    return [CurveBucket(market, float(dates[max(0,i-1)]), float(T),
                        float(dates[min(len(dates)-1,i+1)]), bump)
            for i,T in enumerate(dates)]

def bump_curves(model, buckets: Sequence[CurveBucket], amounts) -> CurveModel:
    amounts = np.asarray(amounts, float)
    if amounts.shape != (len(buckets),) or not np.isfinite(amounts).all():
        raise ValueError('one finite shift per curve bucket required')
    data = model.to_dict()
    markets = set(b.market for b in buckets)
    for name in markets:
        specs = [(b,a) for b,a in zip(buckets,amounts) if b.market == name]
        if name == 'rate':
            knots = model.rate_dates; values = model.rate_forwards
        else:
            if name not in model.curves: raise ValueError(f'unknown bucket market {name}')
            knots = np.asarray(model.curves[name].dates); values = model.curves[name].values
        nodes = np.unique(np.r_[knots, [v for b,_ in specs for v in (b.left,b.center,b.right)]])
        if nodes[-1] > knots[-1] or nodes[0] < knots[0]:
            raise ValueError('bucket exceeds calibrated initial-curve horizon')
        shift = sum(a*b.weights(nodes) for b,a in specs)
        if name == 'rate':
            data['rate_dates'] = nodes.tolist()
            data['rate_forwards'] = (np.interp(nodes,knots,values)+shift).tolist()
        else:
            c = next(c for c in data['curves'] if c['name'] == name)
            base = model.initial(name,nodes)
            c['dates'] = nodes.tolist()
            c['values'] = (base+shift if c['kind']=='normal' else base*np.exp(shift)).tolist()
    return CurveModel.from_dict(data)

def bucket_greeks(model, value: Callable[[CurveModel], float], buckets,
                   cross_gamma=False):
    """Revalue with common random numbers supplied by the valuation callback.

    Delta is currency per unit coordinate; unit_pnl uses the declared bump.
    Halving each bump reports numerical stability independently of scenario error.
    """
    buckets = list(buckets); n = len(buckets)
    if not n: raise ValueError('nonempty bucket set required')
    base = float(value(model)); delta = np.zeros(n); gamma = np.zeros((n,n))
    refined = np.zeros(n)
    for i,b in enumerate(buckets):
        e = np.zeros(n); e[i] = b.bump
        up,dn = float(value(bump_curves(model,buckets,e))),float(value(bump_curves(model,buckets,-e)))
        delta[i] = (up-dn)/(2*b.bump)
        gamma[i,i] = (up-2*base+dn)/b.bump**2
        refined[i] = (float(value(bump_curves(model,buckets,e/2)))-
                      float(value(bump_curves(model,buckets,-e/2))))/b.bump
        if cross_gamma:
            for j in range(i):
                f = np.zeros(n); f[j]=buckets[j].bump
                gamma[i,j]=gamma[j,i]=(float(value(bump_curves(model,buckets,e+f)))
                    -float(value(bump_curves(model,buckets,e-f)))
                    -float(value(bump_curves(model,buckets,-e+f)))
                    +float(value(bump_curves(model,buckets,-e-f))))/(4*b.bump*buckets[j].bump)
    if not np.isfinite(np.r_[base,delta,refined,gamma.ravel()]).all():
        raise ValueError('nonfinite valuation in bucket repricing')
    return {'value':base,'buckets':[asdict(b) for b in buckets],
            'delta':delta,'gamma':gamma,'half_bump_delta':refined,
            'bump_consistency':refined-delta,
            'unit_pnl':delta*np.array([b.bump for b in buckets]),
            'gamma_layout':'full' if cross_gamma else 'diagonal',
            'coordinates':'log compute/funding; absolute electricity and initial forward rates'}

def bucket_factor_jacobian(model, buckets, t=0.):
    """Map factor moves to price-coordinate shifts at bucket centers.

    The bridge is exact for node-supported linear claims and approximate between
    nodes. Rate rows use instantaneous initial-rate shifts and forward loadings;
    callers report the remaining term-structure integration error."""
    rows=[]
    for b in buckets:
        if b.center < t: raise ValueError('risk node already delivered')
        rows.append(model.rate_loading*np.exp(-model.kappa*(b.center-t))
                    if b.market=='rate' else model.loading(b.market,b.center-t))
    return np.asarray(rows)

def bridge_risk(bucket_delta, jacobian, factor_delta=None, rcond=1e-10):
    d=np.asarray(bucket_delta,float); J=np.asarray(jacobian,float)
    if J.ndim!=2 or J.shape[0]!=len(d) or not np.isfinite(J).all() or not np.isfinite(d).all():
        raise ValueError('aligned finite bucket sensitivities and Jacobian required')
    represented=J.T@d
    # Projection acts on *shock coordinates*. It is not the economic hedge
    # projection, which also uses covariance and executable instruments.
    P=J@np.linalg.pinv(J,rcond=rcond)
    out={'factor_delta_from_buckets':represented,'rank':int(np.linalg.matrix_rank(J)),
         'bucket_shock_projector':P,'unspanned_bucket_delta':(np.eye(len(d))-P).T@d}
    if factor_delta is not None:
        actual=np.asarray(factor_delta,float)
        if actual.shape!=represented.shape: raise ValueError('factor delta dimension mismatch')
        out['interpolation_residual']=actual-represented
    return out
