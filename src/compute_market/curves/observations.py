"""Quote-to-curve calibration and causal latent-factor filtering."""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import numpy as np
from scipy.optimize import least_squares
from .model import CurveModel,CurveSpec,_psd_root
from .claims import continuous_strip


@dataclass(frozen=True)
class Quote:
    market: str
    delivery_start: float
    delivery_end: float
    price: float
    standard_error: float
    convention: str='futures' # futures, forward, normalized_physical
    rights_adjustment: float | None=None
    source: str='synthetic executable quote'
    def __post_init__(self):
        if not np.isfinite([self.delivery_start,self.delivery_end,self.price,self.standard_error]).all() or self.delivery_start<0 or self.delivery_end<self.delivery_start or self.standard_error<=0:raise ValueError('invalid quote dates or uncertainty')
        if self.convention not in ('futures','forward','normalized_physical'):raise ValueError('unknown price convention')
        if self.convention=='normalized_physical' and (self.rights_adjustment is None or not np.isfinite(self.rights_adjustment)):raise ValueError('physical quote requires an explicit rights/payment normalization')


def observe(model:CurveModel,quote:Quote,t,x):
    if quote.delivery_start<t-1e-10:raise ValueError('quote delivery precedes observation')
    a=quote.delivery_start;b=quote.delivery_end
    f=model.forward if quote.convention in ('forward','normalized_physical') else model.value
    if abs(a-b)<1e-12:y=f(quote.market,t,x,a)
    elif quote.convention=='futures':y=continuous_strip(model,quote.market,t,x,a,b)
    else:
        # Weighted terminally settled strip; distinct from a single payment at strip end.
        z,w=np.polynomial.legendre.leggauss(24);ds=(a+b)/2+(b-a)*z/2
        numer=np.sum([ww*model.discount(t,x,T)*f(quote.market,t,x,T) for T,ww in zip(ds,w)],axis=0)
        denom=np.sum([ww*model.discount(t,x,T) for T,ww in zip(ds,w)],axis=0);y=numer/denom
    return y+(quote.rights_adjustment or 0.)


def observation_jacobian(model,quote,t,x,bump=1e-5):
    if quote.convention=='futures':
        a=quote.delivery_start;b=quote.delivery_end
        if a==b:return model.greeks(quote.market,t,x,a)[0]
        z,w=np.polynomial.legendre.leggauss(32);knots=[a]+[v for v in model.curves[quote.market].dates if a<v<b]+[b];out=0.
        for lo,hi in zip(knots[:-1],knots[1:]):
            ds=(lo+hi)/2+(hi-lo)*z/2
            out=out+np.sum([ww*model.greeks(quote.market,t,x,T)[0]*(hi-lo)/(2*(b-a)) for T,ww in zip(ds,w)],axis=0)
        return out
    E=np.eye(model.dimension)*bump
    return np.array([(observe(model,quote,t,x+e)-observe(model,quote,t,x-e))/(2*bump) for e in E])


@dataclass
class FactorFilter:
    time: float
    mean: np.ndarray
    covariance: np.ndarray
    observations: int=0
    def __post_init__(self):
        self.mean=np.asarray(self.mean,float);self.covariance=np.asarray(self.covariance,float)
        if self.mean.ndim!=1 or self.covariance.shape!=(len(self.mean),len(self.mean)) or not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all() or self.time<0:raise ValueError('finite filter state and covariance required')
        _psd_root(self.covariance)
    def predict(self,model,time):
        if time<self.time:raise ValueError('filter cannot look backward')
        A,b,C=model.transition_moments(time-self.time,'P')
        self.mean=A*self.mean+b;self.covariance=A[:,None]*self.covariance*A[None,:]+C;self.time=float(time)
        return self
    def update(self,model,quotes):
        if not quotes:return {'count':0,'observed_rank':0,'innovations':[]}
        pred=np.array([observe(model,q,self.time,self.mean) for q in quotes]);y=np.array([q.price for q in quotes]);H=np.array([observation_jacobian(model,q,self.time,self.mean) for q in quotes])
        R=np.diag([q.standard_error**2 for q in quotes]);P=self.covariance
        S=H@P@H.T+R;K=np.linalg.solve(S,H@P).T
        self.mean=self.mean+K@(y-pred);A=np.eye(len(self.mean))-K@H
        self.covariance=A@P@A.T+K@R@K.T;self.covariance=(self.covariance+self.covariance.T)/2
        self.observations+=len(quotes)
        return {'count':len(quotes),'observed_rank':int(np.linalg.matrix_rank(H/np.sqrt(np.diag(R))[:,None])), 'innovations':(y-pred).tolist(),'normalized_innovation_squared':float((y-pred)@np.linalg.solve(S,y-pred))}
    def to_dict(self):return {'time':self.time,'mean':self.mean.tolist(),'covariance':self.covariance.tolist(),'observations':self.observations}
    @classmethod
    def from_dict(cls,d):return cls(float(d['time']),np.asarray(d['mean'],float),np.asarray(d['covariance'],float),int(d.get('observations',0)))


def fit_initial_curve(model,market,quotes,penalty=.01):
    """Fit level knots to point/strip observations while preserving one delivery measure.

    Q drift compensation is unchanged by this initial-condition fit. Other market
    curves are held fixed; factor loadings are estimated in a separate problem.
    """
    if not quotes or any(q.market!=market for q in quotes) or penalty<0:raise ValueError('one-market quote panel and nonnegative penalty required')
    spec=model.curves[market];positive=spec.kind!='normal';initial=np.log(spec.values) if positive else np.asarray(spec.values)
    source=model.to_dict();ix=list(model.curves).index(market)
    def candidate(theta):
        d={**source,'curves':[dict(c) for c in source['curves']]};d['curves'][ix]['values']=(np.exp(theta) if positive else theta).tolist()
        return CurveModel.from_dict(d)
    def residual(theta):
        m=candidate(theta);r=np.array([(float(observe(m,q,0,np.zeros(m.dimension)))-q.price)/q.standard_error for q in quotes])
        return np.r_[r,np.sqrt(penalty)*np.diff(theta,2)]
    bounds=(-12.,12.) if positive else(-1e5,1e5)
    result=least_squares(residual,initial,bounds=bounds,xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=500)
    if not result.success:raise RuntimeError(result.message)
    calibrated=candidate(result.x);J=result.jac[:len(quotes)];sv=np.linalg.svd(J,compute_uv=False)
    return calibrated,{'weighted_quote_rmse':float(np.sqrt(np.mean(residual(result.x)[:len(quotes)]**2))), 'data_rank':int(np.linalg.matrix_rank(J)), 'knots':len(initial),'singular_values':sv.tolist(),'regularization':penalty,'quotes':[asdict(q) for q in quotes]}


def fit_physical_drift(model,tape):
    """Estimate P drift with decay, diffusion and the P jump law fixed.

    The synthetic study observes factors. Inference from filtered factors must
    also account for posterior uncertainty."""
    if tape.measure!='P':raise ValueError('physical history required')
    scores=[];weights=[]
    for n in range(1,len(tape.times)):
        dt=tape.times[n]-tape.times[n-1];A,b,C=model.transition_moments(dt,'P')
        B=np.where(model.kappa>1e-12,-np.expm1(-model.kappa*dt)/np.where(model.kappa>1e-12,model.kappa,1),dt)
        scores.append((tape.factors[:,n]-A*tape.factors[:,n-1])/B-model.jump_intensity_p*model.jump_mean_p)
        weights.append(dt)
    a=np.stack(scores,axis=1);w=np.asarray(weights);per_path=(a*w[None,:,None]).sum(axis=1)/w.sum();eta=per_path.mean(axis=0)
    d=model.to_dict();d['physical_drift']=eta.tolist();fitted=CurveModel.from_dict(d)
    return fitted,{'drift':eta.tolist(),'standard_error':(per_path.std(axis=0,ddof=1)/len(per_path)**.5).tolist(),'paths':len(per_path),'transitions':int(len(per_path)*len(weights)),'source_model_sha256':model.fingerprint,'fitted_model_sha256':fitted.fingerprint}
