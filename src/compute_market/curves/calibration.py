"""Joint calibration, latent-history estimation and path bootstrap.

Declared coordinates and anchors separate initial curves, P dynamics, Q prices
and measurement noise. Identification uses the data Jacobian before priors."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib, json
import numpy as np
from scipy.optimize import least_squares, minimize
from .model import CurveModel, CurveTape
from .observations import Quote, observe
from .option_pricing import option_price

@dataclass(frozen=True)
class Parameter:
    name: str
    path: tuple
    lower: float
    upper: float
    scale: float = 1.
    prior_mean: float | None = None
    prior_standard_error: float | None = None
    def __post_init__(self):
        if not self.path or not np.isfinite([self.lower,self.upper,self.scale]).all() or self.lower>=self.upper or self.scale<=0:
            raise ValueError('parameter path, finite bounds and positive scale required')
        if (self.prior_mean is None)!=(self.prior_standard_error is None):raise ValueError('prior needs mean and standard error')
        if self.prior_standard_error is not None and self.prior_standard_error<=0:raise ValueError('positive prior standard error required')

@dataclass(frozen=True)
class MarketDatum:
    market: str
    time: float
    delivery: float
    value: float
    standard_error: float
    kind: str = 'futures'
    delivery_end: float | None = None
    expiry: float | None = None
    strike: float | None = None
    put: bool = True
    state: tuple[float,...] = ()
    source: str = 'synthetic'
    def __post_init__(self):
        if (not np.isfinite([self.time,self.delivery,self.value,self.standard_error]).all()
                or not 0<=self.time<=self.delivery or self.standard_error<=0):raise ValueError('invalid market observation')
        if self.kind not in ('futures','forward','discount','rate','option'):raise ValueError('unknown observation operator')
        if self.kind=='option' and (self.expiry is None or self.strike is None or not self.time<=self.expiry<=self.delivery or self.strike<=0):raise ValueError('complete option specification required')
    def predict(self,model,state=None):
        x=np.asarray(self.state if state is None else state,float)
        if not len(x):x=np.zeros(model.dimension)
        if x.shape!=(model.dimension,):raise ValueError('observation factor dimension mismatch')
        if self.kind=='option':return float(option_price(model,self.market,self.time,x,self.expiry,self.delivery,self.strike,self.put))
        if self.kind=='discount':return float(model.discount(self.time,x,self.delivery))
        if self.kind=='rate':return float(model.rate_future(self.time,x,self.delivery))
        return float(observe(model,Quote(self.market,self.delivery,self.delivery if self.delivery_end is None else self.delivery_end,self.value,self.standard_error,self.kind,source=self.source),self.time,x))

class Parameterization:
    def __init__(self,model,parameters,anchors):
        self.template=model.to_dict();self.parameters=tuple(parameters);self.anchors=tuple(anchors)
        if not self.parameters or not self.anchors or len(set(p.name for p in parameters))!=len(parameters):
            raise ValueError('unique free coordinates and explicit gauge/identification anchors required')
        if len(set(tuple(p.path) for p in parameters))!=len(parameters):raise ValueError('duplicate free parameter coordinate')
        # Jump covariance parameters are Cholesky coordinates to preserve PSD.
        for side in ('p','q'):
            C=np.asarray(self.template['jump_cov_'+side]);self.template['jump_cholesky_'+side]=np.linalg.cholesky(C+np.eye(len(C))*1e-14).tolist()
    def vector(self):
        out=[]
        for p in self.parameters:
            obj=self.template
            for key in p.path:obj=obj[key]
            out.append(float(obj))
        return np.asarray(out)
    def model(self,values):
        values=np.asarray(values,float)
        if values.shape!=(len(self.parameters),) or not np.isfinite(values).all():raise ValueError('finite parameter vector required')
        data=deepcopy(self.template)
        for p,val in zip(self.parameters,values):
            if val<p.lower-1e-10 or val>p.upper+1e-10:raise ValueError('parameter outside declared bounds')
            obj=data
            for key in p.path[:-1]:obj=obj[key]
            obj[p.path[-1]]=float(val)
        for side in ('p','q'):
            key='jump_cholesky_'+side;L=np.asarray(data.pop(key))
            if any(p.path[0]==key for p in self.parameters):data['jump_cov_'+side]=(np.tril(L)@np.tril(L).T).tolist()
        return CurveModel.from_dict(data)

@dataclass
class CalibrationData:
    quotes: list[MarketDatum]
    transitions: CurveTape | None = None
    frequencies: np.ndarray | None = None
    transition_weight: float = 1.
    def __post_init__(self):
        if not self.quotes and self.transitions is None:raise ValueError('calibration observations required')
        if self.transition_weight<0:raise ValueError('nonnegative observation-block weight required')
        if self.transitions is not None:
            if self.transitions.measure!='P' or len(self.transitions.times)<2:raise ValueError('physical transition history required')
            d=self.transitions.factors.shape[-1]
            if self.frequencies is None:
                self.frequencies=np.vstack([np.eye(d)*s for s in (1.,3.,6.)])
            self.frequencies=np.asarray(self.frequencies,float)
            if self.frequencies.ndim!=2 or self.frequencies.shape[1]!=d:raise ValueError('ECF frequency dimensions differ')
    @property
    def fingerprint(self):
        h=hashlib.sha256(json.dumps([asdict(q) for q in self.quotes],sort_keys=True).encode())
        if self.transitions is not None:
            for a in (self.transitions.times,self.transitions.factors,self.frequencies):h.update(np.asarray(a).tobytes())
        h.update(str(self.transition_weight).encode());return h.hexdigest()
    def residuals(self,model):
        r=[(q.predict(model)-q.value)/q.standard_error for q in self.quotes]
        if self.transitions is not None:
            X=self.transitions.factors;times=self.transitions.times;freq=self.frequencies
            for dt in np.unique(np.round(np.diff(times),12)):
                ix=np.flatnonzero(np.isclose(np.diff(times),dt,rtol=1e-10,atol=1e-12))
                before=X[:,ix].reshape(-1,model.dimension);after=X[:,ix+1].reshape(-1,model.dimension)
                A,b,C=model.transition_moments(float(dt),'P');innovation=after-before*A
                v,w=model._grid(0,float(dt));probe=1j*freq[:,None,:]*np.exp(-v[None,:,None]*model.kappa)
                theory=np.exp(model.cumulant(probe,'P')@w)
                empirical=np.exp(1j*(innovation@freq.T)).mean(axis=0)
                N=len(before);se=np.maximum(np.sqrt(np.maximum(1-abs(theory)**2,1e-8)/N),1e-5)
                diff=(empirical-theory)/se
                # Conditional centering moments identify persistence, which an
                # unconditional innovation marginal alone may not identify.
                old=(before-before.mean(axis=0))/np.maximum(before.std(axis=0),.05)
                new=(innovation-b)/np.maximum(np.sqrt(np.diag(C)),1e-4)
                cross=old.T@new/np.sqrt(N)
                r.extend(np.sqrt(self.transition_weight)*np.r_[diff.real,diff.imag,cross.ravel()])
        out=np.asarray(r,float)
        if not np.isfinite(out).all():raise ValueError('nonfinite calibration residual')
        return out

@dataclass
class CalibrationResult:
    model: CurveModel
    report: dict
    def save(self,path):
        from pathlib import Path
        Path(path).write_text(json.dumps({'model':self.model.to_dict(),'report':self.report},indent=2))

def calibrate(model,parameters,data:CalibrationData,*,anchors,initial=None,max_evaluations=300):
    mapping=Parameterization(model,parameters,anchors);v=mapping.vector() if initial is None else np.asarray(initial,float)
    lo=np.array([p.lower for p in parameters]);hi=np.array([p.upper for p in parameters]);scale=np.array([p.scale for p in parameters])
    if np.any(v<lo) or np.any(v>hi):raise ValueError('starting parameter lies outside declared bounds')
    base_res=data.residuals(mapping.model(v));nr=len(base_res)
    prior=[i for i,p in enumerate(parameters) if p.prior_mean is not None]
    def residual(theta):
        candidate=mapping.model(theta)
        r=data.residuals(candidate)
        if prior:r=np.r_[r,[(theta[i]-parameters[i].prior_mean)/parameters[i].prior_standard_error for i in prior]]
        return r
    opt=least_squares(residual,v,bounds=(lo,hi),x_scale=scale,ftol=1e-9,gtol=1e-8,xtol=1e-9,max_nfev=max_evaluations)
    fitted=mapping.model(opt.x);J=opt.jac[:nr]*scale[None,:];sv=np.linalg.svd(J,compute_uv=False)
    threshold=max(J.shape)*np.finfo(float).eps*(sv[0] if len(sv) else 1.)
    rank=int(np.sum(sv>max(threshold,1e-5*(sv[0] if len(sv) else 1.))))
    report={'success':bool(opt.success),'message':opt.message,'objective':float(2*opt.cost),
            'data_objective':float(np.sum(data.residuals(fitted)**2)),
            'free_parameters':[asdict(p) for p in parameters],'estimates':opt.x.tolist(),
            'anchors':list(anchors),'data_rank':rank,'parameter_count':len(parameters),
            'locally_identified':rank==len(parameters),'scaled_data_singular_values':sv.tolist(),
            'active_bounds':opt.active_mask.tolist(),'evaluations':int(opt.nfev),
            'data_sha256':data.fingerprint,'initial_model_sha256':model.fingerprint,'fitted_model_sha256':fitted.fingerprint,
            'method':'weighted market repricing and conditional empirical-characteristic-function moments',
            'uncertainty':'resample complete independent paths; priors do not add data identification'}
    return CalibrationResult(fitted,report)

def latent_loglikelihood(model,history,initial_mean=None,initial_covariance=None):
    """Gaussian EKF quasi-likelihood using exact jump-inclusive P moments.

    Observation linearization and Gaussian innovations approximate the latent
    jump likelihood while preserving its first two transition moments."""
    d=model.dimension;mean=np.zeros(d) if initial_mean is None else np.asarray(initial_mean,float).copy()
    P=np.eye(d)*1e-6 if initial_covariance is None else np.asarray(initial_covariance,float).copy()
    last=0.;nll=0.;innovations=[];states=[]
    for t,quotes in history:
        if t<last or not quotes:raise ValueError('ordered nonempty observation batches required')
        A,b,C=model.transition_moments(t-last,'P');mean=A*mean+b;P=A[:,None]*P*A[None,:]+C
        y=np.array([q.value for q in quotes]);pred=np.array([q.predict(model,mean) for q in quotes])
        h=1e-5;eye=np.eye(d)*h
        H=np.array([[(q.predict(model,mean+e)-q.predict(model,mean-e))/(2*h) for e in eye] for q in quotes])
        R=np.diag([q.standard_error**2 for q in quotes]);S=H@P@H.T+R
        sign,logdet=np.linalg.slogdet(S)
        if sign<=0:raise ValueError('innovation covariance is not positive definite')
        error=y-pred;K=np.linalg.solve(S,H@P).T
        nll+=.5*(len(y)*np.log(2*np.pi)+logdet+error@np.linalg.solve(S,error))
        mean+=K@error;M=np.eye(d)-K@H;P=M@P@M.T+K@R@K.T;P=(P+P.T)/2
        innovations.append(float(error@np.linalg.solve(S,error)));states.append(mean.copy());last=t
    return float(nll),{'filtered_states':np.asarray(states),'normalized_innovation_squared':np.asarray(innovations),'final_covariance':P}

def calibrate_latent(model,parameters,history,*,anchors,max_iterations=200):
    mapping=Parameterization(model,parameters,anchors)
    scale=np.array([p.scale for p in parameters]);start=mapping.vector()/scale
    def objective(z):
        theta=z*scale;nll,_=latent_loglikelihood(mapping.model(theta),history)
        for i,p in enumerate(parameters):
            if p.prior_mean is not None:nll+=.5*((theta[i]-p.prior_mean)/p.prior_standard_error)**2
        return nll
    opt=minimize(objective,start,method='L-BFGS-B',bounds=[(p.lower/s,p.upper/s) for p,s in zip(parameters,scale)],options={'maxiter':max_iterations,'ftol':1e-10})
    fitted=mapping.model(opt.x*scale);nll,diag=latent_loglikelihood(fitted,history)
    return CalibrationResult(fitted,{'success':bool(opt.success),'message':str(opt.message),'method':'EKF Gaussian quasi-likelihood',
        'negative_loglikelihood':nll,'estimates':(opt.x*scale).tolist(),'free_parameters':[asdict(p) for p in parameters],
        'anchors':list(anchors),'iterations':int(opt.nit),'normalized_innovation_squared':diag['normalized_innovation_squared'].tolist(),
        'fitted_model_sha256':fitted.fingerprint,'identification':'profile likelihood and held-out prices must be assessed for the chosen observation design'})

def bootstrap_calibration(model,parameters,data,*,anchors,replicates=20,seed=719,max_evaluations=150):
    """Path-cluster bootstrap plus independent quote measurement resampling."""
    if replicates<2:raise ValueError('at least two uncertainty replicates required')
    rng=np.random.default_rng(seed);estimates=[];models=[];status=[]
    for _ in range(replicates):
        quotes=[]
        for q in data.quotes:
            d=asdict(q);d['value']=q.value+rng.normal()*q.standard_error;quotes.append(MarketDatum(**d))
        tape=None
        if data.transitions is not None:
            old=data.transitions;ix=rng.integers(0,len(old.factors),len(old.factors))
            tape=CurveTape(old.times,old.factors[ix],old.integrated_rates[ix],old.measure,seed,old.model_sha256)
        result=calibrate(model,parameters,CalibrationData(quotes,tape,data.frequencies,data.transition_weight),anchors=anchors,max_evaluations=max_evaluations)
        status.append(result.report['success'])
        if result.report['success']:estimates.append(result.report['estimates']);models.append(result.model)
    if len(estimates)<2:raise RuntimeError('insufficient converged bootstrap calibrations')
    a=np.asarray(estimates)
    return models,{'replicates':replicates,'converged':len(a),'seed':seed,'estimates':a,
                   'quantiles_025_50_975':np.quantile(a,[.025,.5,.975],axis=0),
                   'method':'whole-path cluster resampling; independent quote noise at stated standard errors'}

def predictive_decomposition(models,market,time,state,delivery,paths=1024,seed=720):
    """Law of total variance across parameter draws and process paths."""
    if len(models)<2:raise ValueError('two or more calibrated parameter draws required')
    means=[];variances=[]
    for model in models:
        tape=model.simulate([time,delivery],paths,seed,'P',state)
        y=model.value(market,delivery,tape.factors[:,-1],delivery)
        means.append(y.mean());variances.append(y.var(ddof=1))
    return {'mean':float(np.mean(means)),'process_variance':float(np.mean(variances)),
            'parameter_variance':float(np.var(means,ddof=1)),
            'total_variance':float(np.mean(variances)+np.var(means,ddof=1)),
            'draws':len(models),'paths_per_draw':paths,'seed':seed}

def latent_profile(model,parameter,history,values,*,anchors):
    """Profile one coordinate with all nuisance parameters fixed.

    Return the objective curve without assigning finite-sample confidence levels."""
    grid=np.asarray(values,float)
    if grid.ndim!=1 or len(grid)<3 or np.any(np.diff(grid)<=0):raise ValueError('three or more increasing profile coordinates required')
    mapping=Parameterization(model,[parameter],anchors)
    objective=np.array([latent_loglikelihood(mapping.model([v]),history)[0] for v in grid])
    return {'parameter':parameter.name,'values':grid,'negative_quasi_loglikelihood':objective,
      'twice_objective_above_grid_minimum':2*(objective-objective.min()),
      'nuisance_parameters':'fixed at the supplied model; not reoptimized',
      'inference':'a sensitivity profile; coverage requires an observation-design-specific bootstrap'}


def fit_input_file(path,output=None,max_evaluations=300):
    """Fit a declared JSON calibration case, resolving its history relative to it."""
    from pathlib import Path
    path=Path(path);d=json.loads(path.read_text());model=CurveModel.from_dict(d['model'])
    parameters=[Parameter(**p) for p in d['parameters']]
    tape=CurveTape.load(path.parent/d['transition_file']) if d.get('transition_file') else None
    data=CalibrationData([MarketDatum(**q) for q in d.get('quotes',[])],tape,
       None if d.get('frequency_grid') is None else np.asarray(d['frequency_grid']),d.get('transition_weight',1.))
    r=calibrate(model,parameters,data,anchors=d['anchors'],initial=d.get('initial'),max_evaluations=max_evaluations)
    if output is not None:r.save(output)
    return r
