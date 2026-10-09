"""Fleet value under switching and retirement.

Train a causal policy, evaluate it on independent paths, and compare it with a
perfect-information upper bound on the same grid. Lender appraisal and sale
proceeds have separate valuation rules."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np
from .model import CurveModel, CurveTape
from .continuation import features

@dataclass(frozen=True)
class AssetSpec:
    compute_market: str
    power_market: str
    gpus: float
    energy_mwh_per_gpu_hour: float
    nonpower_per_gpu_hour: float
    standby_annual_cost: float
    startup_cost: float
    shutdown_cost: float
    terminal_residual: float
    resale_initial: float
    resale_decay: float
    resale_loading: tuple[float,...]
    min_run_steps: int = 1
    utilisation: float = .8
    def __post_init__(self):
        numbers=[self.gpus,self.energy_mwh_per_gpu_hour,self.nonpower_per_gpu_hour,self.standby_annual_cost,
                 self.startup_cost,self.shutdown_cost,self.terminal_residual,self.resale_initial,self.resale_decay,self.utilisation]
        if not np.isfinite(numbers).all() or min(numbers)<0 or self.gpus<=0 or self.utilisation>1:
            raise ValueError('nonnegative asset economics and bounded utilization required')
        if not isinstance(self.min_run_steps,int) or self.min_run_steps<1:
            raise ValueError('minimum operating duration must be positive integral steps')
        if not np.isfinite(self.resale_loading).all():raise ValueError('finite resale loadings required')
    @property
    def states(self):return self.min_run_steps+1
    def resale(self,t,x):
        if np.asarray(x).shape[-1]!=len(self.resale_loading):raise ValueError('resale factor dimension mismatch')
        return self.resale_initial*np.exp(-self.resale_decay*t+np.asarray(x)@np.asarray(self.resale_loading))
    def running_cash(self,model,t,x,dt):
        margin=model.value(self.compute_market,t,x,t)-self.nonpower_per_gpu_hour-self.energy_mwh_per_gpu_hour*model.value(self.power_market,t,x,t)
        return self.gpus*8760*dt*self.utilisation*margin
    def actions(self,state):
        # 0 = idle; 1 = on and free to stop; s>=2 = s-1 future on steps owed.
        if state<0 or state>=self.states:raise ValueError('invalid hardware operating state')
        return ('run',) if state>=2 else ('idle','run','retire')
    def next_state(self,state,action):
        if action not in self.actions(state):raise ValueError('action violates minimum run constraint')
        if action=='retire':return -1
        if action=='idle':return 0
        return self.min_run_steps if state==0 else max(1,state-1)
    def reward(self,model,t,x,dt,state,action):
        if action=='retire':return self.resale(t,x)-(self.shutdown_cost if state else 0.)
        if action=='idle':return np.full(np.asarray(x).shape[:-1],-self.standby_annual_cost*dt-(self.shutdown_cost if state else 0.))
        return self.running_cash(model,t,x,dt)-(self.startup_cost if state==0 else 0.)

@dataclass
class FleetPolicy:
    asset: AssetSpec
    times: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray
    model_sha256: str
    measure: str
    training_seed: int
    training_paths: int
    ridge: float
    def choose(self,model,n,x,states):
        x=np.atleast_2d(x);states=np.broadcast_to(np.asarray(states,int),(len(x),))
        Z=features(x,self.scales[n]);E=Z@self.coefficients[n].T
        actions=np.full(len(x),-1,int);nextstates=np.full(len(x),-1,int);rewards=np.zeros(len(x))
        names=('idle','run','retire');t=self.times[n];dt=self.times[n+1]-t
        for state in np.unique(states):
            if state<0:continue
            ix=np.flatnonzero(states==state);candidates=[];pay=[];ns=[];ids=[]
            for a in self.asset.actions(int(state)):
                if state==0 and a=='run' and len(self.times)-1-n<self.asset.min_run_steps:continue
                r=self.asset.reward(model,t,x[ix],dt,int(state),a);q=self.asset.next_state(int(state),a)
                candidates.append(r+(E[ix,q] if q>=0 else 0.));pay.append(r);ns.append(q);ids.append(names.index(a))
            best=np.argmax(np.stack(candidates,axis=1),axis=1)
            for a in range(len(ids)):
                selected=ix[best==a];actions[selected]=ids[a];nextstates[selected]=ns[a];rewards[selected]=pay[a][best==a]
        return actions,nextstates,rewards
    def save(self,path):
        meta={'asset':asdict(self.asset),'model_sha256':self.model_sha256,'measure':self.measure,
              'training_seed':self.training_seed,'training_paths':self.training_paths,'ridge':self.ridge}
        np.savez_compressed(path,times=self.times,coefficients=self.coefficients,scales=self.scales,metadata=np.array(json.dumps(meta)))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));m['asset']=AssetSpec(**m['asset'])
            return cls(times=z['times'],coefficients=z['coefficients'],scales=z['scales'],**m)

def train_fleet_policy(model,asset,times,paths=8192,seed=701,measure='Q',ridge=1e-4):
    if paths<32 or measure not in ('P','Q') or ridge<0:raise ValueError('valid measure and 32+ training paths required')
    times=np.asarray(times,float)
    if len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):raise ValueError('increasing asset dates from zero required')
    if asset.min_run_steps>len(times)-1:raise ValueError('operating lock exceeds valuation grid')
    tape=model.simulate(times,paths,seed,measure);X=tape.factors;N=len(times);d=model.dimension
    if len(asset.resale_loading)!=d:raise ValueError('resale factor dimensions differ')
    scales=np.maximum(X.std(axis=0),.025);nf=1+d+d*(d+1)//2
    coef=np.zeros((N-1,asset.states,nf));future=np.full((paths,asset.states),asset.terminal_residual,dtype=float)
    for n in range(N-2,-1,-1):
        dt=times[n+1]-times[n];D=np.exp(-(tape.integrated_rates[:,n+1]-tape.integrated_rates[:,n]))
        target=D[:,None]*future;Z=features(X[:,n],scales[n]);pen=np.eye(nf)*ridge;pen[0,0]=0.
        coef[n]=np.linalg.solve(Z.T@Z+pen+np.eye(nf)*1e-12,Z.T@target).T
        estimate=Z@coef[n].T;now=np.zeros_like(future)
        for state in range(asset.states):
            scores=[];actual=[]
            for action in asset.actions(state):
                q=asset.next_state(state,action)
                # Do not start an operation that cannot honor its minimum run.
                if state==0 and action=='run' and N-1-n<asset.min_run_steps:continue
                r=asset.reward(model,times[n],X[:,n],dt,state,action)
                scores.append(r+(estimate[:,q] if q>=0 else 0.))
                actual.append(r+(target[:,q] if q>=0 else 0.))
            choice=np.argmax(np.stack(scores,axis=1),axis=1)
            now[:,state]=np.take_along_axis(np.stack(actual,axis=1),choice[:,None],axis=1)[:,0]
        future=now
    return FleetPolicy(asset,times,coef,scales,model.fingerprint,measure,seed,paths,ridge)

def evaluate_fleet_policy(model,policy,paths=8192,seed=702,tape=None,initial_state=0):
    tape=model.simulate(policy.times,paths,seed,policy.measure) if tape is None else tape
    if not np.array_equal(tape.times,policy.times) or tape.measure!=policy.measure:raise ValueError('matching valuation measure and event grid required')
    X=tape.factors;n=len(X);N=len(policy.times);state=np.full(n,initial_state,int);pv=np.zeros(n);actions=np.full((n,N-1),-1,int)
    for u in range(N-1):
        a,nextstate,reward=policy.choose(model,u,X[:,u],state)
        pv+=np.exp(-tape.integrated_rates[:,u])*reward;actions[:,u]=a;state=nextstate
    pv+=np.exp(-tape.integrated_rates[:,-1])*policy.asset.terminal_residual*(state>=0)
    upper=perfect_information_value(model,policy.asset,tape,initial_state)
    violation=float(np.max(pv-upper,initial=0))
    if violation>1e-5:raise RuntimeError('feasible operating policy exceeded perfect-information bound')
    gap=upper-pv
    return {'value':float(pv.mean()),'standard_error':float(pv.std(ddof=1)/np.sqrt(n)),
            'perfect_information_upper_mean':float(upper.mean()),'upper_standard_error':float(upper.std(ddof=1)/np.sqrt(n)),
            'information_gap':float(gap.mean()),'paired_gap_standard_error':float(gap.std(ddof=1)/np.sqrt(n)),
            'maximum_bound_violation':violation,'paths':n,'measure':policy.measure,'training_paths':policy.training_paths,
            'training_seed':policy.training_seed,'evaluation_seed':tape.seed,'cashflows':pv,'upper_cashflows':upper,'actions':actions,
            'mean_run_intervals':float(np.mean(np.sum(actions==1,axis=1))),'retirement_fraction':float(np.mean(np.any(actions==2,axis=1)))}

def perfect_information_value(model,asset,tape,initial_state=0):
    n,N,_=tape.factors.shape;V=np.full((n,asset.states),asset.terminal_residual,dtype=float)
    for u in range(N-2,-1,-1):
        D=np.exp(-(tape.integrated_rates[:,u+1]-tape.integrated_rates[:,u]));W=np.zeros_like(V);dt=tape.times[u+1]-tape.times[u]
        for state in range(asset.states):
            choices=[]
            for action in asset.actions(state):
                if state==0 and action=='run' and N-1-u<asset.min_run_steps:continue
                ns=asset.next_state(state,action)
                r=asset.reward(model,tape.times[u],tape.factors[:,u],dt,state,action)
                choices.append(r+(D*V[:,ns] if ns>=0 else 0.))
            W[:,state]=np.max(choices,axis=0)
        V=W
    return V[:,initial_state]

def liquidation_proceeds(mark,quantity,market_depth,haircut=.2,impact=.1):
    """Executable proceeds; a haircut and quantity-dependent discount, never PV rent."""
    a=np.asarray(mark,float);q=np.asarray(quantity,float)
    if not np.isfinite(a).all() or not np.isfinite(q).all() or np.any(a<0) or np.any(q<0) or np.any(np.asarray(market_depth)<=0) or not 0<=haircut<1 or impact<0:
        raise ValueError('invalid liquidation inputs')
    return a*q*np.maximum(1-haircut-impact*q/market_depth,0.)

def hardware_value_set(case,economic_result,t=0.,state=None):
    from .portfolio import hardware_marks
    x=np.zeros(case.model.dimension) if state is None else np.asarray(state,float)
    marks=hardware_marks(case.model,t,x,case.fleet);q=np.asarray(case.fleet.blocks,float)
    liquidation=liquidation_proceeds(marks,q,np.maximum(q,1.),1-case.fleet.liquidation_fraction,case.fleet.liquidation_impact)
    return {'economic_policy_value_per_asset':economic_result['value'],
            'economic_standard_error':economic_result['standard_error'],
            'lender_marks_per_block':marks,'lender_collateral_total':float(marks@q),
            'executable_liquidation_per_pool':liquidation,'liquidation_total':float(liquidation.sum()),
            'economic_scope':'single unencumbered asset and selected pricing law; no debt or firm customer commitments',
            'financed_book_value':'separate controlled value; lender appraisal is not replaced by operating-option value'}
