"""Physical book, hardware appraisal and financing as functionals of shared curves."""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import json
import hashlib
import numpy as np
from .model import CurveModel,CurveTape
from .claims import forecast_spot
from .risk import Instrument,default_instruments
from ..simulation import SimulationConfig

POOL_MARKETS=('G1_US','G1_SG','G2_US','G2_SG')
POWER_MARKETS=('power_US','power_SG','power_US','power_SG')


def example_fleet(paths=1024,steps=8,seed=731):
    return SimulationConfig(paths=paths,steps=steps,hours_per_step=8760/steps,seed=seed,
         local_prices=(2.5,2.75,3.75,4.125),firm_rates=(2.55,2.8,3.8,4.15),
         fixed_overhead_per_step=480000/steps,term_amortization=180000/steps,
         term_maturity_step=3*steps,revolver_maturity_step=2*steps,receivable_advance=0.,
         outage_annual_rate=.35,outage_repair_annual_rate=8.,customer_default_annual_hazard=.02,
         initial_cash=500000.,reserve=70000.,initial_revolver=1500000.,initial_term_debt=700000.)


@dataclass
class CurveCase:
    name: str
    model: CurveModel
    fleet: SimulationConfig
    instruments: list[Instrument]
    existing_book: dict
    offered: dict | None=None
    assumptions: dict | None=None
    def __post_init__(self):
        if self.existing_book is None or 'firm_blocks' not in self.existing_book or 'firm_rates' not in self.existing_book:raise ValueError('existing book must be explicit')
        if not np.allclose(self.existing_book['firm_blocks'],self.fleet.firm_blocks) or not np.allclose(self.existing_book['firm_rates'],self.fleet.firm_rates):raise ValueError('book and cash ledger disagree')
        horizon=self.fleet.steps*self.fleet.year_fraction
        grid=np.arange(self.fleet.steps+1)*self.fleet.year_fraction
        if any(T>horizon+1e-10 or np.min(abs(grid-T))>1e-9 for h in self.instruments for T in h.event_dates):raise ValueError('hedge settlements must lie on the event grid')
        if len(set(h.name for h in self.instruments))!=len(self.instruments):raise ValueError('instrument names must be unique')
    def to_dict(self):return {'schema':'compute-market/curve-case','name':self.name,'model':self.model.to_dict(),'fleet':self.fleet.to_dict(),'instruments':[asdict(h) for h in self.instruments],'existing_book':self.existing_book,'offered':self.offered,'assumptions':self.assumptions or {}}
    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(self.to_dict(),indent=2))
    @classmethod
    def from_dict(cls,d):
        d=dict(d)
        if d.pop('schema',None)!='compute-market/curve-case':raise ValueError('expected a complete curve case')
        d['model']=CurveModel.from_dict(d['model']);d['fleet']=SimulationConfig(**d['fleet']);d['instruments']=[__import__('compute_market.curves.instruments',fromlist=['instrument_from_dict']).instrument_from_dict(h) for h in d['instruments']];return cls(**d)
    @classmethod
    def load(cls,path):return cls.from_dict(json.loads(Path(path).read_text()))


def example_case(paths=1024,steps=8,seed=731):
    from .defaults import example_model
    f=example_fleet(paths,steps,seed)
    return CurveCase('financed compute merchant',example_model(),f,default_instruments(f.steps*f.year_fraction),
                     {'firm_blocks':list(f.firm_blocks),'firm_rates':list(f.firm_rates)},
                     {'market':'G1_SG','additional_blocks':1.,'running_rate':0.,'payment':'upfront','start':0.,'end':1.},
                     {'data':'synthetic','clean_measure':'specified affine Q','operating_cash':'non-interest-bearing','hardware_economic_life_years':3.})


def spot_state(model,t,x,c):
    x=np.asarray(x,float)
    local=np.stack([model.value(n,t,x,t) for n in POOL_MARKETS],axis=-1)
    power=np.stack([model.value(n,t,x,t) for n in POWER_MARKETS],axis=-1)
    cost=np.asarray(c.nonpower_unit_cost)+np.asarray(c.mwh_per_gpu_hour)*power
    return local,power,cost


def _appraisal(model,t,x,c):
    """Forward-margin appraisal to a fixed equipment economic life, not an option price."""
    x=np.asarray(x,float);life=3.;steps=12
    if t>=life:return np.zeros(x.shape[:-1]+(4,))
    dates=np.linspace(t,life,steps+1);ans=np.zeros(x.shape[:-1]+(4,))
    for a,b in zip(dates[:-1],dates[1:]):
        T=(a+b)/2;D=model.discount(t,x,T);hours=(b-a)*8760
        for j,(compute,power) in enumerate(zip(POOL_MARKETS,POWER_MARKETS)):
            margin=model.forward(compute,t,x,T)-c.nonpower_unit_cost[j]-c.mwh_per_gpu_hour[j]*model.forward(power,t,x,T)
            ans[...,j]+=D*np.maximum(margin,0)*hours*c.gpus_per_block[j]*c.expected_spot_utilization
    residual=np.asarray(c.hardware_values)*.08
    ans+=model.discount(t,x,life)[...,None]*residual
    return ans


def hardware_marks(model,t,x,c):
    key=('appraisal-scale',tuple(c.hardware_values),tuple(c.nonpower_unit_cost),tuple(c.mwh_per_gpu_hour),tuple(c.gpus_per_block),c.expected_spot_utilization)
    if key not in model._bond_cache:model._bond_cache[key]=np.asarray(c.hardware_values)/_appraisal(model,0,np.zeros(model.dimension),c)
    # Resale basis captures appraisal risk outside the rental-margin mark.
    liquidity=np.exp(-.35*np.asarray(x)[...,7]+.10*np.asarray(x)[...,8])
    return _appraisal(model,t,x,c)*model._bond_cache[key]*liquidity[...,None]


def continuation_value(model,t,x,holdings,debt,term,end,c,customer_alive=True):
    """Mark a future operating plan in terminal operating-account dollars.

    P means price physical cash flows; Q marks price financial claims. The current
    dispatch is a separate LP action. Terminal hardware covers only post-horizon
    life, so rental cash flows are counted once."""
    x=np.atleast_2d(x);h=np.asarray(holdings);firm=np.asarray(c.firm_blocks)
    value=np.zeros(len(x));dt=c.year_fraction
    dates=np.arange(t+dt/2,end-1e-10,dt)
    for T in dates:
        local=np.stack([forecast_spot(model,n,t,x,T) for n in POOL_MARKETS],axis=-1)
        power=np.stack([forecast_spot(model,n,t,x,T) for n in POWER_MARKETS],axis=-1)
        cost=np.asarray(c.nonpower_unit_cost)+np.asarray(c.mwh_per_gpu_hour)*power
        owned=np.minimum(h,firm);replacement=np.maximum(firm-h,0)
        spare=np.maximum(h-firm,0)*c.expected_spot_utilization
        gh=np.asarray(c.gpus_per_block)*dt*8760
        cash=(spare*np.maximum(local-cost,0)-owned*cost-replacement*c.replacement_multiplier*local)@gh
        cash+=float((firm*np.asarray(c.firm_rates))@gh)*customer_alive-c.fixed_overhead_per_step
        future_rate=model.rate_future(t,x,T)
        spread=model.value('funding',t,x,T)
        cash-=dt*(debt*(future_rate+spread)+term*c.fixed_term_coupon)
        value+=cash
    value+=hardware_marks(model,end,x*np.exp(-model.kappa*(end-t)),c)@h*(c.liquidation_fraction-c.liquidation_impact)-debt-term
    return value


@dataclass
class EconomyTape:
    curve: CurveTape
    hardware: np.ndarray
    eligible: np.ndarray
    availability: np.ndarray
    demand: np.ndarray
    replacement: np.ndarray
    freeze: np.ndarray
    customer_alive: np.ndarray
    margin_multiplier: np.ndarray
    spread_multiplier: np.ndarray
    depth_multiplier: np.ndarray
    benchmark_basis: np.ndarray
    local_multiplier: np.ndarray
    name: str='nominal'
    counterparty_alive: np.ndarray | None=None
    def save(self,path):
        arrays={k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        arrays.update(times=self.curve.times,factors=self.curve.factors,integrated_rates=self.curve.integrated_rates)
        arrays['metadata']=np.array(json.dumps({'name':self.name,'measure':self.curve.measure,'seed':self.curve.seed,'model_sha256':self.curve.model_sha256}))
        np.savez_compressed(path,**arrays)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            d=json.loads(str(f['metadata']));name=d.pop('name');curve=CurveTape(f['times'],f['factors'],f['integrated_rates'],**d)
            return cls(curve,**{k:f[k] for k in f.files if k not in ('times','factors','integrated_rates','metadata')},name=name)
    def copy(self):
        return EconomyTape(CurveTape(self.curve.times.copy(),self.curve.factors.copy(),self.curve.integrated_rates.copy(),self.curve.measure,self.curve.seed,self.curve.model_sha256),**{k:v.copy() for k,v in vars(self).items() if isinstance(v,np.ndarray)},name=self.name)


def economy_tape(case,paths=None,seed=None):
    c=case.fleet;n=paths or c.paths;seed=c.seed if seed is None else seed;T=c.steps
    dates=np.arange(T+1)*c.year_fraction;curve=case.model.simulate(dates,n,seed,'P');x=curve.factors
    rng=np.random.default_rng(np.random.SeedSequence([seed,781]))
    hardware=np.stack([hardware_marks(case.model,t,x[:,i],c) for i,t in enumerate(dates)],axis=1)
    eligible=hardware[:,np.maximum(0,np.arange(T+1)-c.eligible_mark_lag)].copy()
    shape=(n,T+1,4);avail=np.ones(shape);demand=np.zeros(shape);repl=np.broadcast_to(c.replacement_blocks,shape).copy();frozen=np.zeros((n,T+1),bool);alive=np.ones((n,T+1),bool)
    down=np.zeros((n,2),bool);severity=np.zeros((n,2));dt=c.year_fraction
    for u in range(T+1):
        if u:
            hazard=c.outage_annual_rate*np.exp(.25*x[:,u,5,None]);total=hazard+c.outage_repair_annual_rate
            pr=hazard/total+(down-hazard/total)*np.exp(-total*dt)
            new=rng.random((n,2))<pr;sev=rng.beta(2,3,(n,2));severity=np.where(new&~down,sev,severity);down=new
            hp=c.customer_default_annual_hazard*np.exp(x[:,u,7]-.3*x[:,u,0]);alive[:,u]=alive[:,u-1]&(rng.random(n)>-np.expm1(-hp*dt))
        avail[:,u]=1-down[:,[0,1,0,1]]*severity[:,[0,1,0,1]];repl[:,u]*=np.maximum(.1,1-.7*down[:,[0,1,0,1]])
        demand[:,u]=np.asarray(c.blocks)*c.expected_spot_utilization*np.exp(.35*x[:,u,0,None]+.15*rng.normal(size=(n,4))-.5*.15**2)
        frozen[:,u]=rng.random(n)<-np.expm1(-.05*np.exp(x[:,u,7])*dt)
    margin=1+.3*np.maximum(x[:,:,7],0)
    return EconomyTape(curve,hardware,eligible,avail,demand,repl,frozen,alive,margin,np.ones((n,T+1)),np.ones((n,T+1)),np.ones((n,T+1,2)),np.ones(shape))


def stress_economy(case,tape,name):
    out=tape.copy();u=max(1,case.fleet.steps//2);x=out.curve.factors
    shifts={'level_collapse':np.array([-.4,0,0,0,0,0,0,0,0]),'long_end_repricing':np.array([0,0,-.6,0,0,0,0,.3,0]),
            'generation_transition':np.array([-.1,0,0,.9,0,0,0,0,0]),'regional_scarcity':np.array([0,.15,0,0,.9,0,0,.2,0]),
            'power_and_rates':np.array([0,0,0,0,0,1.,.5,.3,0]),'liquidity_spiral':np.array([-.2,0,-.2,0,0,.2,.2,1.,0])}
    if name in shifts:
        shock=shifts[name]
        for j in range(u,len(out.curve.times)):
            age=out.curve.times[j]-out.curve.times[u];x[:,j]+=np.exp(-case.model.kappa*age)*shock
            out.hardware[:,j]=hardware_marks(case.model,out.curve.times[j],x[:,j],case.fleet)
        lag=case.fleet.eligible_mark_lag;out.eligible[:,u:]=out.hardware[:,np.maximum(0,np.arange(u,len(out.curve.times))-lag)]
    if name=='basis_break':out.local_multiplier[:,u:]*=.65
    if name=='benchmark_dislocation':out.benchmark_basis[:,u:]*=1.40
    if name in ('margin_call','liquidity_spiral'):out.margin_multiplier[:,u:]*=4
    if name in ('execution_freeze','liquidity_spiral'):
        out.depth_multiplier[:,u:]=0.;out.spread_multiplier[:,u:]*=10
    if name in ('lender_haircut','liquidity_spiral'):
        out.eligible[:,u:]*=.55;out.freeze[:,u:]=True
    if name=='counterparty_default':
        if out.counterparty_alive is None:out.counterparty_alive=np.ones(out.curve.factors.shape[:2],bool)
        out.counterparty_alive[:,u:]=False
    if name=='outage':out.availability[:,u:,[0,1]]*=.25;out.replacement[:,u:,[0,1]]*=.25
    allowed=set(shifts)|{'basis_break','benchmark_dislocation','margin_call','execution_freeze','lender_haircut','outage','counterparty_default'}
    if name not in allowed:raise ValueError('unknown curve stress')
    out.name=name
    # Realized integrated rates are part of the state and are recomputed under a
    # deterministic injected rate-factor shock; no scenario occurrence probability.
    if name in shifts:
        for j in range(u+1,len(out.curve.times)):
            age0=out.curve.times[j-1]-out.curve.times[u];age1=out.curve.times[j]-out.curve.times[u]
            extra=float(case.model.rate_loading@(shifts[name]*np.exp(-case.model.kappa*age0)*__import__('compute_market.curves.model',fromlist=['decay_integral']).decay_integral(case.model.kappa,age1-age0)))
            out.curve.integrated_rates[:,j:]+=extra
    return out


def project_case(paths=1024,steps=8,seed=731):
    """Complete reference case: futures, funded claims and observable liquidity."""
    from .instruments import diversified_instruments
    from .liquidity import LiquiditySpec
    case=example_case(paths,steps,seed)
    case.instruments=diversified_instruments(case.model,steps*case.fleet.year_fraction)
    case.assumptions.update({'liquidity_spec':asdict(LiquiditySpec()),
        'counterparty_hazard_annual':.02,'margin_collateral':'segregated and fully returned at modeled closeout',
        'economic_asset_value':'optimal switching policy, separate from lender marks',
        'financial_inventory':'ex-dividend funded claims and continuously settled futures'})
    case.__post_init__()
    return case


def project_economy(case,paths,seed):
    from .liquidity import LiquiditySpec,apply_liquidity
    tape=economy_tape(case,paths,seed)
    spec=LiquiditySpec(**case.assumptions['liquidity_spec'])
    tape,diagnostics=apply_liquidity(case,tape,spec,seed+318)
    alive=np.ones((paths,len(tape.curve.times)),bool);rng=np.random.default_rng(seed+414)
    for u,dt in enumerate(np.diff(tape.curve.times),1):
        hazard=case.assumptions.get('counterparty_hazard_annual',.02)*np.exp(.5*tape.curve.factors[:,u,7]+.5*diagnostics['state'][:,u])
        alive[:,u]=alive[:,u-1]&(rng.random(paths)>-np.expm1(-hazard*dt))
    tape.counterparty_alive=alive
    return tape,diagnostics
