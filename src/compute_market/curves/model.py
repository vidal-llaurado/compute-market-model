"""Finite-factor Musiela curves with an affine OU--Levy realization.

All dates are ACT/365 year fractions from the calibration origin. Financial
commodity curves are continuously settled futures references. Terminally paid
forwards use the separate collateralized discounting operator.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import numpy as np
from numpy.polynomial.legendre import leggauss


def _finite(x, name):
    a=np.asarray(x, dtype=float)
    if not np.isfinite(a).all(): raise ValueError(f'{name} must be finite')
    return a


def decay_integral(k, t):
    """Integral of exp(-k s), including the zero mean-reversion limit."""
    k=np.asarray(k, float);t=np.asarray(t, float)
    return np.divide(-np.expm1(-k*t), k, out=np.broadcast_to(t,np.broadcast_shapes(k.shape,t.shape)).copy(), where=np.abs(k)>1e-12)


def _psd_root(a):
    a=(a+a.T)/2
    v,u=np.linalg.eigh(a)
    if v.min() < -1e-10*max(1.,float(abs(v).max())): raise ValueError('covariance is not positive semidefinite')
    return u*np.sqrt(np.maximum(v,0.))


@dataclass(frozen=True)
class CurveSpec:
    name: str
    kind: str                    # log, normal, or funding (nontraded commercial spread)
    dates: tuple[float,...]
    values: tuple[float,...]
    loading: tuple[float,...]
    unit: str
    observable: str='reference'  # benchmark, reference, or commercial

    def __post_init__(self):
        d=_finite(self.dates,'curve dates');v=_finite(self.values,'curve values');_finite(self.loading,'loading')
        if self.kind not in ('log','normal','funding'): raise ValueError('curve kind must be log, normal or funding')
        if d.ndim!=1 or len(d)<2 or v.shape!=d.shape or d[0]!=0 or np.any(np.diff(d)<=0): raise ValueError('curve dates must start at zero and increase')
        if self.kind!='normal' and np.any(v<=0): raise ValueError('log and funding levels must be positive')
        if self.observable not in ('benchmark','reference','commercial'): raise ValueError('unknown observation class')


class CurveModel:
    """Affine factor state with quadrature-evaluated cumulants.

    X follows dX=(-kappa X+eta_M)dt+L dW_M+dJ_M, with eta_Q=0.
    J has compound-Poisson arrivals and multivariate Gaussian marks. P and Q
    share diffusion and loading shapes; drift and jump laws may differ."""
    def __init__(self, factors, kappa, diffusion, curves, *, physical_drift=None,
                 jump_intensity_q=0.,jump_intensity_p=0.,jump_mean_q=None,
                 jump_mean_p=None,jump_cov_q=None,jump_cov_p=None,
                 rate_loading=None,rate_dates=(0.,10.),rate_forwards=(.04,.04),quadrature=24):
        self.factors=tuple(factors);self.kappa=_finite(kappa,'kappa');self.diffusion=_finite(diffusion,'diffusion')
        k=len(self.factors)
        if k==0 or len(set(self.factors))!=k or self.kappa.shape!=(k,) or np.any(self.kappa<0) or self.diffusion.shape!=(k,k): raise ValueError('inconsistent factor dimensions')
        self.cov=self.diffusion@self.diffusion.T
        self.physical_drift=np.zeros(k) if physical_drift is None else _finite(physical_drift,'physical drift')
        self.jump_intensity_q=float(jump_intensity_q);self.jump_intensity_p=float(jump_intensity_p)
        if not np.isfinite([jump_intensity_q,jump_intensity_p]).all() or min(jump_intensity_q,jump_intensity_p)<0: raise ValueError('nonnegative jump intensities required')
        self.jump_mean_q=np.zeros(k) if jump_mean_q is None else _finite(jump_mean_q,'jump mean Q')
        self.jump_mean_p=self.jump_mean_q.copy() if jump_mean_p is None else _finite(jump_mean_p,'jump mean P')
        self.jump_cov_q=np.zeros((k,k)) if jump_cov_q is None else _finite(jump_cov_q,'jump covariance Q')
        self.jump_cov_p=self.jump_cov_q.copy() if jump_cov_p is None else _finite(jump_cov_p,'jump covariance P')
        self.rate_loading=np.zeros(k) if rate_loading is None else _finite(rate_loading,'rate loading')
        for a in (self.physical_drift,self.jump_mean_q,self.jump_mean_p,self.rate_loading):
            if a.shape!=(k,): raise ValueError('factor vector has wrong dimension')
        for a in (self.jump_cov_q,self.jump_cov_p):
            if a.shape!=(k,k) or not np.allclose(a,a.T): raise ValueError('jump covariance must be symmetric')
            _psd_root(a)
        self.rate_dates=_finite(rate_dates,'rate dates');self.rate_forwards=_finite(rate_forwards,'rate forwards')
        if self.rate_dates.ndim!=1 or self.rate_dates.shape!=self.rate_forwards.shape or self.rate_dates[0]!=0 or np.any(np.diff(self.rate_dates)<=0): raise ValueError('invalid initial rate curve')
        self.curves={c.name:c for c in curves}
        if len(self.curves)!=len(curves) or not curves or any(len(c.loading)!=k for c in curves): raise ValueError('unique, factor-aligned curves required')
        self.quadrature=int(quadrature)
        if self.quadrature<8: raise ValueError('at least eight quadrature nodes required')
        self._nodes,self._weights=leggauss(self.quadrature)
        # Equivalent P/Q specifications must share diffusion and jump support.
        residual=self.physical_drift-self.diffusion@np.linalg.pinv(self.diffusion)@self.physical_drift
        if np.linalg.norm(residual)>1e-9: raise ValueError('P drift is outside the diffusion span')
        if bool(self.jump_intensity_p)!=bool(self.jump_intensity_q): raise ValueError('P and Q must share jump-event support')
        if self.jump_intensity_q:
            PP=self.jump_cov_p@np.linalg.pinv(self.jump_cov_p);PQ=self.jump_cov_q@np.linalg.pinv(self.jump_cov_q)
            dm=self.jump_mean_p-self.jump_mean_q
            if not np.allclose(PP,PQ,atol=1e-8) or np.linalg.norm(dm-PQ@dm)>1e-8:raise ValueError('P and Q normal jump marks need equivalent support')
        self._transition_cache={};self._comp_cache={};self._bond_cache={}

    @property
    def dimension(self): return len(self.factors)

    def to_dict(self):
        keys=('kappa','diffusion','physical_drift','jump_intensity_q','jump_intensity_p','jump_mean_q','jump_mean_p','jump_cov_q','jump_cov_p','rate_loading','rate_dates','rate_forwards','quadrature')
        return {'factors':list(self.factors),'curves':[{**asdict(c),'dates':list(map(float,c.dates)),'values':list(map(float,c.values)),'loading':list(map(float,c.loading))} for c in self.curves.values()],**{k:(getattr(self,k).tolist() if isinstance(getattr(self,k),np.ndarray) else getattr(self,k)) for k in keys}}

    @classmethod
    def from_dict(cls,data):
        d=dict(data);d['curves']=[CurveSpec(**c) for c in d['curves']];return cls(**d)

    @property
    def fingerprint(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _grid(self,a,b):
        if not np.isfinite([a,b]).all() or a < -1e-12 or b<a-1e-12: raise ValueError('ordered nonnegative dates required')
        return .5*(a+b)+.5*(b-a)*self._nodes,.5*(b-a)*self._weights

    def initial(self,name,T):
        c=self.curves[name];T=_finite(T,'delivery')
        if np.any(T<0) or np.any(T>c.dates[-1]+1e-10): raise ValueError(f'delivery outside calibrated {name} horizon')
        a=np.asarray(c.values)
        return np.interp(T,c.dates,a) if c.kind=='normal' else np.exp(np.interp(T,c.dates,np.log(a)))

    def loading(self,name,x):
        x=_finite(x,'time to delivery')
        if np.any(x< -1e-10): raise ValueError('delivery has expired')
        return np.asarray(self.curves[name].loading)*np.exp(-np.maximum(x,0)[...,None]*self.kappa)

    def cumulant(self,z,measure='Q'):
        """Unit-time log MGF of Brownian plus raw compound-Poisson increments."""
        if measure not in ('P','Q'): raise ValueError('measure must be P or Q')
        z=np.asarray(z);m=getattr(self,'jump_mean_'+measure.lower());C=getattr(self,'jump_cov_'+measure.lower());nu=getattr(self,'jump_intensity_'+measure.lower())
        ans=.5*np.einsum('...i,ij,...j->...',z,self.cov,z)
        if measure=='P': ans=ans+z@self.physical_drift
        if nu: ans=ans+nu*np.expm1(z@m+.5*np.einsum('...i,ij,...j->...',z,C,z))
        return ans

    def compensator(self,name,t,T):
        """Integral from zero to t of the Q exponent at delivery lag T-s."""
        key=(name,float(t),float(T))
        if key not in self._comp_cache:
            if T<t-1e-10: raise ValueError('delivery has expired')
            u,w=self._grid(T-t,T);l=self.loading(name,u)
            c=self.curves[name]
            if c.kind=='funding':a=0.
            elif c.kind=='normal':a=float(w@(l@(self.jump_intensity_q*self.jump_mean_q)))
            else:a=float(w@self.cumulant(l))
            self._comp_cache[key]=a
        return self._comp_cache[key]

    def value(self,name,t,x,T):
        """One dated price for any batch of factor states; multiple T uses stack."""
        x=_finite(x,'state')
        if x.shape[-1]!=self.dimension or t<0: raise ValueError('state dimension or date mismatch')
        c=self.curves[name];v=x@self.loading(name,T-t)-self.compensator(name,t,T)
        level=self.initial(name,T)
        if c.kind=='normal': return level+v
        with np.errstate(over='raise',invalid='raise'):
            try:return level*np.exp(v)
            except FloatingPointError as e:raise ValueError('curve state exceeds finite exponential domain') from e

    def surface(self,name,t,x,delivery_dates):
        return np.stack([self.value(name,t,x,float(T)) for T in delivery_dates],axis=-1)

    def greeks(self,name,t,x,T):
        l=self.loading(name,T-t);v=self.value(name,t,x,T)
        if self.curves[name].kind=='normal':
            shape=np.shape(v);return np.broadcast_to(l,shape+(self.dimension,)),np.zeros(shape+(self.dimension,self.dimension))
        return v[...,None]*l,v[...,None,None]*np.outer(l,l)

    def rate_B(self,tau):return self.rate_loading*decay_integral(self.kappa,np.asarray(tau)[...,None])

    def rate_shift(self,t):
        t=_finite(t,'rate time')
        if np.any(t<0) or np.any(t>self.rate_dates[-1]):raise ValueError('rate date outside calibration horizon')
        return np.interp(t,self.rate_dates,self.rate_forwards)+self.cumulant(-self.rate_B(t))

    def shift_integral(self,t,T):
        key=(float(t),float(T),'shift')
        if key not in self._bond_cache:
            # Split at rate-curve knots so piecewise-linear initial forwards integrate exactly.
            dates=[t]+[d for d in self.rate_dates if t<d<T]+[T];v=0.
            for a,b in zip(dates[:-1],dates[1:]):
                u,w=self._grid(a,b);v+=float(w@self.rate_shift(u))
            self._bond_cache[key]=v
        return self._bond_cache[key]

    def short_rate(self,t,x):return self.rate_shift(t)+np.asarray(x)@self.rate_loading

    def discount(self,t,x,T):
        key=(float(t),float(T),'bond')
        if key not in self._bond_cache:
            v,w=self._grid(0,T-t)
            self._bond_cache[key]=-self.shift_integral(t,T)+float(w@self.cumulant(-self.rate_B(v)))
        return np.exp(self._bond_cache[key]-np.asarray(x)@self.rate_B(T-t))

    def forward(self,name,t,x,T):
        """Fixed strike for a terminally settled clean forward, including convexity."""
        c=self.curves[name]
        if c.kind=='funding':raise ValueError('commercial funding curve is not a traded underlying')
        v,w=self._grid(0,T-t);l=self.loading(name,v);B=self.rate_B(v)
        if c.kind=='log':
            correction=float(w@(self.cumulant(l-B)-self.cumulant(-B)-self.cumulant(l)))
            return self.value(name,t,x,T)*np.exp(correction)
        # The linear payoff changes measure by exponential tilting of the increments.
        shift=self.cov@(-B).T
        if self.jump_intensity_q:
            m=self.jump_mean_q;C=self.jump_cov_q
            a=np.exp(-B@m+.5*np.einsum('...i,ij,...j->...',B,C,B))
            shift+=self.jump_intensity_q*((m[None,:]-B@C)*a[:,None]-m).T
        return self.value(name,t,x,T)+float(w@np.sum(l*shift.T,axis=1))

    def rate_future(self,t,x,T):
        l=self.rate_loading*np.exp(-self.kappa*(T-t))
        mean=self.jump_intensity_q*self.jump_mean_q*decay_integral(self.kappa,T-t)
        return self.rate_shift(T)+np.asarray(x)@l+self.rate_loading@mean

    def transition_moments(self,dt,measure='P'):
        if measure not in ('P','Q') or dt<0: raise ValueError('nonnegative step and P/Q required')
        A=np.exp(-self.kappa*dt);B=decay_integral(self.kappa,dt)
        m=getattr(self,'jump_mean_'+measure.lower());C=getattr(self,'jump_cov_'+measure.lower());nu=getattr(self,'jump_intensity_'+measure.lower())
        eta=self.physical_drift if measure=='P' else np.zeros(self.dimension)
        diffusion=self.cov+nu*(C+np.outer(m,m))
        v,w=self._grid(0,dt);a=np.exp(-v[:,None]*self.kappa)
        cov=np.einsum('n,ni,ij,nj->ij',w,a,diffusion,a)
        return A,B*(eta+nu*m),cov

    def _segment(self,dt):
        if dt not in self._transition_cache:
            v,w=self._grid(0,dt);a=np.exp(-v[:,None]*self.kappa);b=decay_integral(self.kappa,v[:,None])
            block=np.concatenate([a,b],axis=1);full=np.tile(self.cov,(2,2))
            C=np.einsum('n,ni,ij,nj->ij',w,block,full,block)
            self._transition_cache[dt]=_psd_root(C)
        return self._transition_cache[dt]

    def step(self,x,dt,rng,measure='P'):
        """Joint exact-law OU endpoint and integrated factors, including jump ages."""
        if measure not in ('P','Q') or not np.isfinite(dt) or dt<0:raise ValueError('nonnegative step and P/Q required')
        x=_finite(x,'factor state');scalar=x.ndim==1;x=np.atleast_2d(x)
        if x.shape[1]!=self.dimension:raise ValueError('factor dimension mismatch')
        k=self.dimension;n=len(x);A=np.exp(-self.kappa*dt);B=decay_integral(self.kappa,dt)
        eta=self.physical_drift if measure=='P' else np.zeros(k)
        H=np.divide(dt-B,self.kappa,out=np.full(k,.5*dt**2),where=self.kappa>1e-12)
        z=rng.normal(size=(n,2*k))@self._segment(float(dt)).T
        nxt=x*A+B*eta+z[:,:k];integ=x*B+H*eta+z[:,k:]
        nu=getattr(self,'jump_intensity_'+measure.lower())
        if nu and dt:
            counts=rng.poisson(nu*dt,n);index=np.repeat(np.arange(n),counts);count=len(index)
            if count:
                age=rng.uniform(0,dt,count);m=getattr(self,'jump_mean_'+measure.lower());C=getattr(self,'jump_cov_'+measure.lower())
                marks=m+rng.normal(size=(count,k))@_psd_root(C).T
                np.add.at(nxt,index,marks*np.exp(-age[:,None]*self.kappa))
                np.add.at(integ,index,marks*decay_integral(self.kappa,age[:,None]))
        return (nxt[0],integ[0]) if scalar else (nxt,integ)

    def simulate(self,times,paths,seed=0,measure='P',initial=None):
        times=_finite(times,'times')
        if times.ndim!=1 or len(times)<1 or times[0]<0 or np.any(np.diff(times)<=0) or not isinstance(paths,(int,np.integer)) or paths<1:raise ValueError('positive path count and increasing dates required')
        rng=np.random.default_rng(seed);X=np.zeros((paths,len(times),self.dimension));R=np.zeros((paths,len(times)))
        if initial is not None:X[:,0]=np.broadcast_to(initial,X[:,0].shape)
        for n in range(1,len(times)):
            X[:,n],integ=self.step(X[:,n-1],float(times[n]-times[n-1]),rng,measure)
            R[:,n]=R[:,n-1]+integ@self.rate_loading+self.shift_integral(float(times[n-1]),float(times[n]))
        return CurveTape(times,X,R,measure,seed,self.fingerprint)


@dataclass
class CurveTape:
    times: np.ndarray
    factors: np.ndarray
    integrated_rates: np.ndarray
    measure: str
    seed: int
    model_sha256: str
    def save(self,path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,times=self.times,factors=self.factors,integrated_rates=self.integrated_rates,
                            metadata=np.array(json.dumps({'measure':self.measure,'seed':self.seed,'model_sha256':self.model_sha256})))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            return cls(f['times'],f['factors'],f['integrated_rates'],**json.loads(str(f['metadata'])))


# The concrete reference realization; MarketModel is the architectural interface.
AffineCurveModel = CurveModel
