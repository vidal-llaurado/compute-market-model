"""Delivery operators, clean derivatives and nonanticipative exercise on curves."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from .model import CurveModel


@dataclass(frozen=True)
class DeliveryStrip:
    market: str
    dates: tuple[float,...]
    quantities: tuple[float,...] # GPU-hours or MWh, not a concurrency constraint
    def __post_init__(self):
        d=np.asarray(self.dates,float);q=np.asarray(self.quantities,float)
        if d.ndim!=1 or q.shape!=d.shape or not len(d) or not np.isfinite(d).all() or not np.isfinite(q).all() or np.any(d<0) or np.any(q<0) or q.sum()<=0:raise ValueError('positive-volume delivery strip required')

    def mark(self,model,t,x,discounted=False):
        terms=[q*(model.discount(t,x,T)*model.forward(self.market,t,x,T) if discounted else model.value(self.market,t,x,T)) for q,T in zip(self.quantities,self.dates)]
        return np.sum(terms,axis=0)

    def average(self,model,t,x):return self.mark(model,t,x)/sum(self.quantities)

    def greeks(self,model,t,x):
        d=[];g=[]
        for q,T in zip(self.quantities,self.dates):
            a,b=model.greeks(self.market,t,x,T);d.append(q*a);g.append(q*b)
        return np.sum(d,axis=0),np.sum(g,axis=0)


def continuous_strip(model,name,t,x,start,end,nodes=48):
    """Arithmetic, not geometric, average of the instantaneous-delivery curve."""
    if not t<=start<end:raise ValueError('delivery interval must be in the future')
    z,w=np.polynomial.legendre.leggauss(nodes)
    knots=[start]+[d for d in model.curves[name].dates if start<d<end]+[end]
    ans=0.
    for a,b in zip(knots[:-1],knots[1:]):
        T=(a+b)/2+(b-a)*z/2
        ans=ans+(model.surface(name,t,x,T)@w)*(b-a)/(2*(end-start))
    return ans


def forecast_spot(model,name,t,x,T):
    """Physical conditional expectation of the standardized spot at T."""
    c=model.curves[name];v,w=model._grid(0,T-t);l=model.loading(name,v)
    if c.kind=='funding':raise ValueError('funding is a commercial curve, not a spot reference')
    f=model.value(name,t,x,T)
    if c.kind=='log':return f*np.exp(float(w@(model.cumulant(l,'P')-model.cumulant(l,'Q'))))
    drift=model.physical_drift+model.jump_intensity_p*model.jump_mean_p-model.jump_intensity_q*model.jump_mean_q
    return f+float(w@(l@drift))


def european_option(model:CurveModel,market,t,x,expiry,delivery,strike,*,put=True,paths=32768,seed=44):
    """Cash-funded European option on a dated futures reference.

    Diffusion case: exact Gaussian forward-measure formula with rate convexity.
    Jump case: joint endpoint/integrated-rate Monte Carlo under the selected Q.
    Returned MC standard error measures simulation error, not calibration error.
    """
    if not 0<=t<=expiry<=delivery or strike<=0 or paths<2:raise ValueError('invalid option dates, strike or sample')
    if model.curves[market].kind!='log':raise ValueError('option helper expects a positive futures reference')
    x=np.asarray(x,float)
    if x.shape!=(model.dimension,):raise ValueError('one current factor state required')
    if expiry==t:
        f=float(model.value(market,t,x,delivery));return {'price':max((strike-f) if put else (f-strike),0.),'standard_error':0.,'method':'intrinsic'}
    if model.jump_intensity_q==0:
        tau=expiry-t;v,w=model._grid(0,tau);l=model.loading(market,delivery-expiry+v);B=model.rate_B(v)
        var=float(w@np.einsum('ni,ij,nj->n',l,model.cov,l))
        conv=-float(w@np.einsum('ni,ij,nj->n',l,model.cov,B))
        mean=float(model.value(market,t,x,delivery))*np.exp(conv);D=float(model.discount(t,x,expiry))
        if var<1e-15:p=D*max((strike-mean) if put else(mean-strike),0.)
        else:
            vol=np.sqrt(var);d1=np.log(mean/strike)/vol+vol/2;d2=d1-vol
            p=D*(strike*ndtr(-d2)-mean*ndtr(-d1)) if put else D*(mean*ndtr(d1)-strike*ndtr(d2))
        return {'price':float(p),'standard_error':0.,'method':'Gaussian affine forward measure'}
    X,I=model.step(np.broadcast_to(x,(paths,len(x))),expiry-t,np.random.default_rng(seed),'Q')
    D=np.exp(-model.shift_integral(t,expiry)-I@model.rate_loading)
    F=model.value(market,expiry,X,delivery);payoff=np.maximum(strike-F,0) if put else np.maximum(F-strike,0)
    v=D*payoff
    return {'price':float(v.mean()),'standard_error':float(v.std(ddof=1)/np.sqrt(paths)), 'method':'affine jump Monte Carlo','paths':paths,'seed':seed}


def finite_difference_greeks(function,x,bump=1e-4):
    x=np.asarray(x,float);k=len(x);base=float(function(x));d=np.zeros(k);g=np.zeros((k,k));e=np.eye(k)*bump
    for i in range(k):
        up=float(function(x+e[i]));dn=float(function(x-e[i]));d[i]=(up-dn)/(2*bump);g[i,i]=(up-2*base+dn)/bump**2
        for j in range(i):
            g[i,j]=g[j,i]=(float(function(x+e[i]+e[j]))-float(function(x+e[i]-e[j]))-float(function(x-e[i]+e[j]))+float(function(x-e[i]-e[j])))/(4*bump**2)
    return {'value':base,'delta':d,'gamma':g,'bump':bump}


def _features(x,scale):
    z=np.asarray(x)/scale
    return np.column_stack([np.ones(len(z)),z,z*z])


@dataclass
class SwingPolicy:
    coefficients: np.ndarray # [date,remaining rights,factor features]
    scales: np.ndarray
    max_rights: int
    dates: np.ndarray
    strike: float
    market: str
    ridge: float


def train_swing(model,market,times,strike,max_rights=2,paths=8192,seed=321,ridge=1e-5):
    """Fitted backward stopping rule; each nomination consumes one interval right.

    At most one right per date, zero minimum exercise, no future information.
    Continuation regressions are fitted only on the supplied training tape.
    """
    times=np.asarray(times,float)
    if times[0]!=0 or max_rights<1 or max_rights>len(times)-1 or paths<16 or strike<=0:raise ValueError('invalid swing training specification')
    tape=model.simulate(times,paths,seed,'Q');N=len(times);k=model.dimension
    coef=np.zeros((N,max_rights+1,1+2*k));scales=np.maximum(tape.factors.std(axis=0),.02)
    V=np.zeros((paths,max_rights+1))
    for n in range(N-1,0,-1):
        F=model.value(market,times[n],tape.factors[:,n],times[n]);pay=np.maximum(F-strike,0)
        Z=_features(tape.factors[:,n],scales[n]);A=Z.T@Z+ridge*np.eye(Z.shape[1]);A[0,0]-=ridge
        if n<N-1:
            D=np.exp(-(tape.integrated_rates[:,n+1]-tape.integrated_rates[:,n]));future=V*D[:,None]
        else:future=np.zeros_like(V)
        coef[n]=np.linalg.solve(A,Z.T@future).T
        estimate=Z@coef[n].T
        new=np.zeros_like(V)
        for r in range(1,max_rights+1):
            exercise=pay+estimate[:,r-1]>estimate[:,r]
            new[:,r]=np.where(exercise,pay+future[:,r-1],future[:,r])
        V=new
    return SwingPolicy(coef,scales,max_rights,times,float(strike),market,float(ridge))


def evaluate_swing(model,policy,paths=8192,seed=432):
    tape=model.simulate(policy.dates,paths,seed,'Q');remaining=np.full(paths,policy.max_rights,int);pv=np.zeros(paths);exercise=np.zeros((paths,len(policy.dates)),bool)
    for n in range(1,len(policy.dates)):
        F=model.value(policy.market,policy.dates[n],tape.factors[:,n],policy.dates[n]);pay=np.maximum(F-policy.strike,0)
        est=_features(tape.factors[:,n],policy.scales[n])@policy.coefficients[n].T
        ix=np.arange(paths);act=(remaining>0)&(pay+est[ix,np.maximum(remaining-1,0)]>est[ix,remaining])
        pv+=np.exp(-tape.integrated_rates[:,n])*pay*act;remaining-=act;exercise[:,n]=act
    return {'value':float(pv.mean()),'standard_error':float(pv.std(ddof=1)/np.sqrt(paths)), 'paths':paths,'seed':seed,'mean_exercises':float(exercise.sum(axis=1).mean()),'maximum_exercises':int(exercise.sum(axis=1).max()),'payoffs':pv}
