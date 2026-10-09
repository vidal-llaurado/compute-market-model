"""Cash-ledger simulation of a synthetic financed fleet.

A monthly decision kernel uses continuous pool allocations and policy-independent
tapes. Policies observe the current state. Failure means the specified financing
rule fails; legal default and the exact finite-tree policy are separate questions."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Any
import hashlib
import json
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class SimulationConfig:
    paths: int = 2000
    steps: int = 24
    hours_per_step: float = 720.
    seed: int = 1729
    blocks: tuple[float, ...] = (8., 6., 5., 4.)
    gpus_per_block: tuple[float, ...] = (8., 8., 8., 8.)
    local_prices: tuple[float, ...] = (2., 2.2, 3., 3.2)
    hardware_values: tuple[float, ...] = (160000., 160000., 240000., 240000.)
    firm_blocks: tuple[float, ...] = (2., 1., 1., 1.)
    firm_rates: tuple[float, ...] = (2.1, 2.25, 3.05, 3.2)
    nonpower_unit_cost: tuple[float, ...] = (.32, .35, .46, .49)
    mwh_per_gpu_hour: tuple[float, ...] = (.0012, .0012, .0015, .0015)
    initial_cash: float = 400000.
    reserve: float = 50000.
    initial_revolver: float = 1600000.
    initial_term_debt: float = 900000.
    revolver_commitment: float = 2600000.
    advance_rate: float = .60
    receivable_advance: float = .50
    revolver_spread: float = .055
    initial_reference_rate: float = .04
    fixed_term_coupon: float = .075
    term_amortization: float = 18000.
    term_maturity_step: int = 24
    revolver_maturity_step: int = 18
    refinance_probability: float = .90
    max_draw_per_step: float = 400000.
    debt_fee: float = .001
    prepayment_fraction: float = .5
    customer_default_annual_hazard: float = .03
    counterparty_default_annual_hazard: float = .01
    counterparty_recovery: float = .40
    eligible_mark_lag: int = 1
    liquidation_fraction: float = .90
    liquidation_impact: float = .12
    maximum_sale_fraction: float = .70
    replacement_multiplier: float = 1.50
    replacement_blocks: tuple[float, ...] = (2., 2., 2., 2.)
    shortfall_penalty_per_gpu_hour: float = 6.
    fixed_overhead_per_step: float = 65000.
    asset_decline_annual: float = .14
    rental_decline_annual: float = .08
    benchmark_decline_annual: float = .08
    factor_vol_scale: float = 1.
    common_jump_annual_rate: float = .35
    outage_annual_rate: float = .8
    outage_repair_annual_rate: float = 12.
    expected_spot_utilization: float = .80
    hedge_margin_fraction: float = .18
    hedge_spread_fraction: float = .003
    hedge_forward_carry_annual: float = 0.
    power_forward_carry_annual: float = 0.

    def __post_init__(self):
        if not isinstance(self.paths,int) or self.paths<1 or not isinstance(self.steps,int) or self.steps<1:
            raise ValueError('positive integer paths and steps required')
        if not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('seed must be a nonnegative integer')
        j=len(self.blocks)
        for name in ('blocks','gpus_per_block','local_prices','hardware_values','firm_blocks','firm_rates',
                     'nonpower_unit_cost','mwh_per_gpu_hour','replacement_blocks'):
            a=np.asarray(getattr(self,name),float)
            if a.shape!=(j,) or not np.isfinite(a).all() or np.any(a<0):
                raise ValueError(f'{name}: equal finite nonnegative pool vectors required')
        if j != 4 or np.any(np.asarray(self.gpus_per_block)<=0) or np.any(np.asarray(self.local_prices)<=0):
            raise ValueError('this synthetic factor map has exactly four pools with positive block sizes/prices')
        for name,value in asdict(self).items():
            if isinstance(value,(float,int)) and not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('initial_cash','reserve','initial_revolver','initial_term_debt','revolver_commitment',
                     'revolver_spread','initial_reference_rate','fixed_term_coupon','term_amortization',
                     'max_draw_per_step','debt_fee','customer_default_annual_hazard','counterparty_default_annual_hazard',
                     'liquidation_impact','shortfall_penalty_per_gpu_hour','fixed_overhead_per_step',
                     'asset_decline_annual','rental_decline_annual','benchmark_decline_annual','factor_vol_scale','common_jump_annual_rate',
                     'outage_annual_rate','outage_repair_annual_rate','hedge_margin_fraction','hedge_spread_fraction'):
            if getattr(self,name)<0: raise ValueError(f'{name} must be nonnegative')
        for name in ('advance_rate','receivable_advance','refinance_probability','prepayment_fraction',
                     'counterparty_recovery','maximum_sale_fraction','expected_spot_utilization'):
            if not 0<=getattr(self,name)<=1: raise ValueError(f'{name} must be in [0,1]')
        if not 0<=self.debt_fee<1:
            raise ValueError('debt transaction fee must be below one dollar per dollar')
        if self.hours_per_step<=0 or not 0<self.liquidation_fraction<=1 or self.replacement_multiplier<=0:
            raise ValueError('invalid hours, liquidation fraction or replacement multiplier')
        if self.liquidation_fraction-2*self.liquidation_impact<=0:
            raise ValueError('net sale proceeds must be increasing over the full sale range')
        if not isinstance(self.eligible_mark_lag,int) or self.eligible_mark_lag<0:
            raise ValueError('eligible mark lag must be a nonnegative integer')
        for name in ('term_maturity_step','revolver_maturity_step'):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')

    @property
    def year_fraction(self): return self.hours_per_step/(24*365.)

    def to_dict(self): return asdict(self)


@dataclass
class MarketTape:
    local: np.ndarray                    # [path, boundary, pool], observed local price
    index: np.ndarray                    # [path,boundary], synthetic financial benchmark
    power: np.ndarray                    # [path,boundary], USD/MWh
    hardware: np.ndarray                 # [path,boundary,pool], ordinary mark
    eligible: np.ndarray                 # [path,boundary,pool], lagged lender mark
    availability: np.ndarray             # [path,boundary,pool]
    demand_blocks: np.ndarray            # [path,boundary,pool], pooled block-hours / interval hours
    replacement: np.ndarray              # [path,boundary,pool], verified replacement blocks
    reference_rate: np.ndarray           # [path,boundary]
    freeze: np.ndarray                   # [path,boundary], draw conditions fail
    customer_alive: np.ndarray           # [path,boundary], absorbing common customer credit event
    counterparty_alive: np.ndarray       # [path,boundary], absorbing hedge counterparty event
    refinance_u: np.ndarray              # [path,boundary], opening financing offer observable when due
    margin_multiplier: np.ndarray       # [path,boundary]
    log_likelihood: np.ndarray            # [path], P/Q for optional Gaussian tilting
    config: SimulationConfig
    metadata: dict[str,Any]

    def copy(self):
        return MarketTape(**{name:(value.copy() if isinstance(value,np.ndarray) else value)
                             for name,value in vars(self).items()})

    def validate(self):
        n,t,j=self.config.paths,self.config.steps+1,4
        for name in ('local','hardware','eligible','availability','demand_blocks','replacement'):
            a=getattr(self,name)
            if a.shape!=(n,t,j) or not np.isfinite(a).all() or np.any(a<0):
                raise ValueError(f'invalid tape {name}')
        for name in ('index','power','reference_rate','refinance_u','margin_multiplier'):
            a=getattr(self,name)
            if a.shape!=(n,t) or not np.isfinite(a).all() or np.any(a<0):
                raise ValueError(f'invalid tape {name}')
        for name in ('freeze','customer_alive','counterparty_alive'):
            if getattr(self,name).shape!=(n,t): raise ValueError(f'invalid tape {name}')
        if np.any(self.availability>1) or self.log_likelihood.shape!=(n,) or not np.isfinite(self.log_likelihood).all():
            raise ValueError('invalid availability or likelihood')
        if np.any(self.local<=0) or np.any(self.index<=0) or np.any(self.power<=0):
            raise ValueError('strictly positive financial marks required')
        return self


def _rng(seed,stream,step):
    return np.random.default_rng(np.random.SeedSequence([seed,stream,step]))


def generate_tape(c: SimulationConfig, gaussian_tilt: float=0.) -> MarketTape:
    """Sample the discrete kernel with exact regime-conditional OU segments.

    Six Gaussian drivers share marked Poisson shocks. gaussian_tilt shifts the
    first independent normal and records its exact likelihood ratio. Reuse the
    complete tape for policy and stress comparisons."""
    if not np.isfinite(gaussian_tilt): raise ValueError('finite importance tilt required')
    n,T,j=c.paths,c.steps,4;dt=c.year_fraction
    shape=(n,T+1,j);scalar=(n,T+1)
    local=np.zeros(shape);hardware=np.zeros(shape);avail=np.ones(shape)
    demand=np.zeros(shape);replacement=np.broadcast_to(c.replacement_blocks,shape).copy()
    index=np.zeros(scalar);power=np.zeros(scalar);rates=np.zeros(scalar)
    freeze=np.zeros(scalar,bool);ca=np.ones(scalar,bool);ha=np.ones(scalar,bool)
    margin=np.ones(scalar);refu=np.zeros(scalar)
    x=np.zeros((n,6));regime=np.zeros(n,dtype=int);logw=np.zeros(n)
    outage_state=np.zeros((n,2),bool);outage_severity=np.zeros((n,2))
    # Common demand, power, hardware, credit, A-basis, B-basis.
    correlation=np.array([[1,.15,.40,-.35,.25,.25],[.15,1,.05,.15,.05,.05],
                          [.40,.05,1,-.20,.10,.10],[-.35,.15,-.20,1,0,0],
                          [.25,.05,.10,0,1,.25],[.25,.05,.10,0,.25,1.]])
    kappa=np.array([1.2,2.,.7,1.8,2.5,2.5]);sig=np.array([.35,.40,.25,.65,.25,.25])*c.factor_vol_scale
    cov=correlation*sig[:,None]*sig[None,:]*(-np.expm1(-(kappa[:,None]+kappa[None,:])*dt))/(kappa[:,None]+kappa[None,:])
    L=np.linalg.cholesky(cov) if np.any(cov) else np.zeros((6,6))
    means=np.array([[0,0,0,0,0,0],[-.40,.10,-.30,.90,-.10,-.10],[.40,.25,.12,.25,.15,.10]])
    # Rates converted to interval probabilities, so changing dt has an explicit kernel.
    transition=np.array([[1,0,0],[0,1,0],[0,0,1]],float)
    exit_rates=[.6,1.8,2.0]
    next_weights=[[0,.45,.55],[.85,0,.15],[.85,.15,0]]
    for r in range(3):
        leave=-np.expm1(-exit_rates[r]*dt)
        transition[r]=(1-leave)*np.eye(3)[r]+leave*np.array(next_weights[r])
    for t in range(T+1):
        if t:
            u=_rng(c.seed,0,t).random(n)
            regime=(u[:,None]>np.cumsum(transition[regime],axis=1)).sum(axis=1)
            z=_rng(c.seed,1,t).normal(size=(n,6));z[:,0]+=gaussian_tilt
            logw+=-gaussian_tilt*z[:,0]+.5*gaussian_tilt**2
            x=means[regime]+np.exp(-kappa*dt)*(x-means[regime])+z@L.T
            counts=_rng(c.seed,2,t).poisson(c.common_jump_annual_rate*dt,size=n)
            marks=_rng(c.seed,3,t).normal(size=(n,2))
            severity=counts*.22+np.sqrt(counts)*.08*marks[:,0]
            # Signed common adverse mark. Negative realization is allowed, not silently clipped.
            x+=severity[:,None]*np.array([-1.,.65,-1.25,1.6,-.35,-.35])
        basis=x[:,[4,5,4,5]]
        technology=np.array([1.0,1.0,.6,.6])
        index[:,t]=c.local_prices[0]*np.exp(x[:,0]-c.benchmark_decline_annual*t*dt)
        local[:,t]=np.array(c.local_prices)*np.exp(x[:,0,None]+basis-c.rental_decline_annual*t*dt*technology)
        power[:,t]=70*np.exp(x[:,1])
        hardware[:,t]=np.array(c.hardware_values)*np.exp(.45*x[:,0,None]+x[:,2,None]-c.asset_decline_annual*t*dt*technology)
        rates[:,t]=c.initial_reference_rate*np.exp(.2*x[:,3])
        refu[:,t]=_rng(c.seed,4,t).random(n)
        freeze[:,t]=_rng(c.seed,5,t).random(n)<(-np.expm1(-.35*dt*np.exp(x[:,3])))
        # Persistent site-level outage state with exact two-state CTMC skeleton
        # for rates held over this interval. Repairs do not disappear with a finer grid.
        if t:
            fail=c.outage_annual_rate*np.exp(.3*x[:,1,None])
            total=fail+c.outage_repair_annual_rate
            equilibrium=np.divide(fail,total,out=np.zeros_like(fail),where=total>0)
            decay=np.exp(-total*dt)
            probability_down=equilibrium+(outage_state-equilibrium)*decay
            newstate=_rng(c.seed,6,t).random((n,2))<probability_down
            newseverity=_rng(c.seed,7,t).beta(2.,4.,size=(n,2))
            outage_severity=np.where(newstate & ~outage_state,newseverity,outage_severity)
            outage_state=newstate
        avail[:,t]=1-outage_state[:,[0,1,0,1]]*outage_severity[:,[0,1,0,1]]
        replacement[:,t]*=np.maximum(.05,1-.8*outage_state[:,[0,1,0,1]])
        # Request counts scale with elapsed hours; gamma marks have fixed expected
        # useful demand, so annual arrival/work intensity is not changed by dt.
        reference_arrivals=80*c.hours_per_step/720.
        counts=_rng(c.seed,8,t).poisson(reference_arrivals*np.exp(x[:,0,None]+.25*basis),size=(n,j))
        hours=_rng(c.seed,9,t).gamma(2*counts,1/(2*reference_arrivals),size=(n,j))
        demand[:,t]=np.array(c.blocks)*c.expected_spot_utilization*hours
        if t:
            hazard=c.customer_default_annual_hazard*np.exp(.8*x[:,3]-.4*x[:,0])
            ca[:,t]=ca[:,t-1]&(_rng(c.seed,10,t).random(n)>-np.expm1(-hazard*dt))
            hazard=c.counterparty_default_annual_hazard*np.exp(.7*x[:,3])
            ha[:,t]=ha[:,t-1]&(_rng(c.seed,11,t).random(n)>-np.expm1(-hazard*dt))
        margin[:,t]=1+.45*np.abs(x[:,0])+.2*np.maximum(x[:,3],0)
    lag=np.maximum(0,np.arange(T+1)-c.eligible_mark_lag)
    eligible=hardware[:,lag].copy()
    payload=json.dumps(c.to_dict(),sort_keys=True).encode()
    tape=MarketTape(local,index,power,hardware,eligible,avail,demand,replacement,rates,freeze,ca,ha,
                    refu,margin,logw,c,{'synthetic':True,'kernel':'interval-regime exact-OU + boundary compound Poisson',
                    'config_sha256':hashlib.sha256(payload).hexdigest(),'gaussian_tilt':gaussian_tilt,
                    'scope':'pooled continuous-block, current-observation simulator; not a latent-state optimal policy'})
    return tape.validate()


@dataclass(frozen=True)
class HedgePolicy:
    name: str='unhedged'
    compute_ratio: float=0.
    power_ratio: float=0.
    adaptive_volume: bool=True
    margin_cash_fraction: float=.50
    def __post_init__(self):
        if not all(np.isfinite([self.compute_ratio,self.power_ratio,self.margin_cash_fraction])):
            raise ValueError('finite hedge policy parameters required')
        if not 0<=self.compute_ratio<=1 or not 0<=self.power_ratio<=1 or not 0<=self.margin_cash_fraction<=1:
            raise ValueError('ratios must be in [0,1]')


@dataclass
class SimulationResult:
    policy: str
    terminal_wealth: np.ndarray
    financing_failure: np.ndarray
    delivery_breach: np.ndarray
    failure_step: np.ndarray
    minimum_cash_headroom: np.ndarray
    maximum_borrowing_gap: np.ndarray
    total_asset_sales: np.ndarray
    total_shortfall_gpu_hours: np.ndarray
    hedge_cash_pnl: np.ndarray
    creditor_loss: np.ndarray
    equity_recovery: np.ndarray
    traces: dict[str,np.ndarray]
    audit: dict[str,Any]
    metadata: dict[str,Any]


def sale_proceeds(marked_value, sale_fraction, realization, impact):
    return marked_value*(realization*sale_fraction-impact*sale_fraction**2)


def _feasible_root(feasibility, lo, hi):
    """Return a cash-feasible endpoint of a bracketed sale root."""
    root = brentq(feasibility, lo, hi, xtol=np.nextafter(0., 1.))
    if feasibility(root) >= 0:
        return float(root)
    # A fraction tolerance can cross a binding dollar constraint. Retain the
    # feasible bracket endpoint until the two fractions are adjacent floats.
    lo = root
    while np.nextafter(lo, hi) < hi:
        mid = lo + (hi-lo)/2
        if feasibility(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return float(hi)


def first_feasible_sale(feasibility, maximum: float, grid_points: int=65):
    """First sampled feasible bracket + root; does not assume global monotonicity.

    For the implemented proportional-sale repair, headroom is concave piecewise
    linear/quadratic while delivery feasibility imposes an upper bound. A dense
    scan is still numerical; returns the grid and reports nonexistence on grid.
    """
    if not 0<=maximum<=1 or grid_points<3: raise ValueError('invalid sale search')
    grid=np.linspace(0,maximum,grid_points); values=np.array([feasibility(float(x)) for x in grid])
    hits=np.flatnonzero(values>=0)
    if not len(hits): return None, float(values.max())
    i=int(hits[0])
    if i==0:return 0.,float(values[0])
    if values[i-1]>=0:return float(grid[i-1]),float(values[i-1])
    root=_feasible_root(feasibility,float(grid[i-1]),float(grid[i]))
    return float(root),float(feasibility(root))


def _repair_one(c, holdings, cash, debt, hardware, eligible, availability, replacement,
                local, owned_cost, commitment, draw_frozen, receivable_support, margin):
    """Find the minimum proportional sale under the financing heuristic.

    The one-dimensional cure uses current cash only. Unreplaced delivery shortfall
    is charged and recorded."""
    firm=np.asarray(c.firm_blocks);gh=np.asarray(c.gpus_per_block)*c.hours_per_step
    marked=float(holdings@hardware)
    def evaluate(x,full=False):
        remaining=holdings*(1-x)
        owned=np.minimum(firm,remaining*availability)
        external=np.minimum(firm-owned,replacement)
        short=np.maximum(0,firm-owned-external)
        cost=float(gh@(owned*owned_cost+external*c.replacement_multiplier*local+short*c.shortfall_penalty_per_gpu_hour))
        cost+=c.fixed_overhead_per_step
        proceeds=sale_proceeds(marked,x,c.liquidation_fraction,c.liquidation_impact)
        bb=float(c.advance_rate*(remaining@eligible)+receivable_support)
        ceiling=min(commitment,bb)
        repay=max(0.,debt-ceiling)
        drawmax=0. if draw_frozen else min(c.max_draw_per_step,max(0.,ceiling-debt))
        liquidity=cash+proceeds-repay*(1+c.debt_fee)+drawmax*(1-c.debt_fee)-margin-cost-c.reserve
        if full:
            draw=min(drawmax,max(0.,(c.reserve+margin+cost-(cash+proceeds-repay*(1+c.debt_fee)))/(1-c.debt_fee)))
            return dict(holdings=remaining,owned=owned,external=external,short=short,cost=cost,
                        proceeds=proceeds,repay=repay,draw=draw,bb=bb,headroom=liquidity)
        return liquidity
    # Partition at every physical and financing kink. On each interval headroom
    # is a concave quadratic (sale impact); globally it need not be concave.
    if evaluate(0.)>=0:return evaluate(0.,True)
    knots=[0.,c.maximum_sale_fraction]
    cap=holdings*availability
    for j in range(len(cap)):
        if cap[j]>0:
            knots.extend([1-firm[j]/cap[j],1-(firm[j]-replacement[j])/cap[j]])
    bbassets=float(c.advance_rate*(holdings@eligible))
    if bbassets>0:
        for level in (commitment,debt,debt+c.max_draw_per_step):
            knots.append(1-(level-receivable_support)/bbassets)
    knots=sorted(set(max(0.,min(c.maximum_sale_fraction,float(x))) for x in knots))
    quadratic=-marked*c.liquidation_impact
    for lo,hi in zip(knots[:-1],knots[1:]):
        if hi-lo<1e-12:continue
        flo,fhi=evaluate(lo),evaluate(hi)
        linear=(fhi-flo)/(hi-lo)-quadratic*(hi+lo)
        peak=float(np.clip(-linear/(2*quadratic),lo,hi)) if quadratic<0 else (hi if fhi>flo else lo)
        if flo>=0:return evaluate(lo,True)
        if evaluate(peak)>=0:
            root=_feasible_root(evaluate,lo,peak)
            return evaluate(root,True)
    return None


def simulate(tape: MarketTape, policy: HedgePolicy=HedgePolicy(), trace_paths: int=8) -> SimulationResult:
    """Replay a causal policy and resolve financing failures immediately.

    Failure rates condition on the model and policy. Resolution marks remaining
    service obligations with the stated closeout proxy; it does not assume delivery
    continues after failure."""
    tape.validate();c=tape.config;n,T=c.paths,c.steps;j=4;dt=c.year_fraction
    if trace_paths<0: raise ValueError('trace_paths must be nonnegative')
    trace_n=min(n,trace_paths)
    blocks=np.broadcast_to(c.blocks,(n,j)).copy();cash=np.full(n,c.initial_cash,dtype=float)
    debt=np.full(n,c.initial_revolver,dtype=float);term=np.full(n,c.initial_term_debt,dtype=float)
    active=np.ones(n,bool);failed=np.zeros(n,bool);breach=np.zeros(n,bool);fs=np.full(n,-1,int)
    wealth=np.zeros(n);mincash=np.full(n,np.inf);maxgap=np.zeros(n);totalsales=np.zeros(n)
    shortgh=np.zeros(n);hedgepnl=np.zeros(n);creditor=np.zeros(n);equity=np.zeros(n)
    oldh=np.zeros((n,2));oldstrike=np.zeros((n,2));posted=np.zeros(n)
    commitment=np.full(n,c.revolver_commitment,dtype=float)
    gh=np.array(c.gpus_per_block)*c.hours_per_step;firm=np.array(c.firm_blocks)
    trace={key:np.full((trace_n,T+1),np.nan) for key in ('cash','revolver','term_debt','blocks','margin',
           'compute_hedge','power_hedge','funding_gap','local_price','index','asset_mark')}
    maxcasherror=0.;maxinventoryerror=0.;maxdebterror=0.;maxdeliveryerror=0.
    for t in range(T+1):
        for i in np.flatnonzero(active):
            beforecash=cash[i];beforeblocks=blocks[i].copy();beforedebt=debt[i]
            cashflows=0.
            if t:
                marks=np.array([tape.index[i,t],tape.power[i,t]])
                settle=float(oldh[i]@(marks-oldstrike[i]))
                if not tape.counterparty_alive[i,t] and settle>0:
                    settle*=c.counterparty_recovery
                hedgepnl[i]+=settle;cashflows+=settle
                cashflows+=posted[i];posted[i]=0.
                interest=debt[i]*(c.revolver_spread+tape.reference_rate[i,t-1])*dt+term[i]*c.fixed_term_coupon*dt
                amort=term[i] if t==c.term_maturity_step else min(term[i],c.term_amortization)
                term[i]-=amort;cashflows-=interest+amort
                if t==c.revolver_maturity_step:
                    # Observe the refinancing offer at maturity.
                    risk_adjusted_probability=c.refinance_probability/(1+.25*(tape.margin_multiplier[i,t]-1))
                    if tape.refinance_u[i,t]>risk_adjusted_probability:commitment[i]=0.
            cash[i]+=cashflows
            if t==T:
                # Terminal exit is a cash liquidation, including impact, not a magical
                # noncash asset mark. All remaining debt is extinguished or recognized as loss.
                proceeds=sale_proceeds(float(blocks[i]@tape.hardware[i,t]),1.,
                                       c.liquidation_fraction,c.liquidation_impact)
                cash[i]+=proceeds-debt[i]-term[i]
                cashflows+=proceeds-debt[i]-term[i]
                blocks[i]*=0.;debt[i]=0.;term[i]=0.;oldh[i]*=0.
                wealth[i]=cash[i]
                if wealth[i]<-1e-7:failed[i]=True;fs[i]=t
                creditor[i]=max(0.,-wealth[i]);equity[i]=max(0.,wealth[i]);active[i]=False
            else:
                fixed=float(gh@(firm*np.array(c.firm_rates)))*float(tape.customer_alive[i,t])
                opening=fixed*c.prepayment_fraction;closing=fixed-opening
                cash[i]+=opening;cashflows+=opening
                ownedcost=np.array(c.nonpower_unit_cost)+np.array(c.mwh_per_gpu_hour)*tape.power[i,t]
                planned_blocks=blocks[i] if policy.adaptive_volume else np.array(c.blocks)
                floating_hours=float(gh@np.maximum(0,planned_blocks-firm))*c.expected_spot_utilization
                power_mwh=float(gh@(np.array(c.mwh_per_gpu_hour)*(firm+c.expected_spot_utilization*np.maximum(0,planned_blocks-firm))))
                h=np.array([-policy.compute_ratio*floating_hours,policy.power_ratio*power_mwh])
                # Once the only synthetic hedge counterparty defaults, no new contract can be opened.
                if not tape.counterparty_alive[i,t]:h*=0
                marks=np.array([tape.index[i,t],tape.power[i,t]])
                margin_per=np.abs(h)@marks*c.hedge_margin_fraction*tape.margin_multiplier[i,t]
                budget=policy.margin_cash_fraction*max(0.,cash[i]-c.reserve)
                if margin_per>budget and margin_per>0:h*=budget/margin_per
                newmargin=float(np.abs(h)@marks*c.hedge_margin_fraction*tape.margin_multiplier[i,t])
                spread=float(np.abs(h)@marks*c.hedge_spread_fraction)
                cash[i]-=spread;cashflows-=spread;hedgepnl[i]-=spread
                support=c.receivable_advance*closing
                maxgap[i]=max(maxgap[i],max(0.,debt[i]-min(commitment[i],c.advance_rate*float(blocks[i]@tape.eligible[i,t])+support)))
                repaired=_repair_one(c,blocks[i],cash[i],debt[i],tape.hardware[i,t],tape.eligible[i,t],
                                    tape.availability[i,t],tape.replacement[i,t],tape.local[i,t],ownedcost,
                                    commitment[i],bool(tape.freeze[i,t]),support,newmargin)
                if repaired is None:
                    failed[i]=True;fs[i]=t;active[i]=False
                    totalsales[i]+=float(blocks[i].sum())
                    # Refund opening payment for unserved interval; close out all remaining firm service.
                    cash[i]+=-opening+spread;cashflows+=-opening+spread;hedgepnl[i]+=spread
                    liquidation=sale_proceeds(float(blocks[i]@tape.hardware[i,t]),1.,c.liquidation_fraction,c.liquidation_impact)
                    closeout=(T-t)*float(gh@(firm*np.maximum(0.,c.replacement_multiplier*tape.local[i,t]-np.array(c.firm_rates)*float(tape.customer_alive[i,t]))))
                    wealth[i]=cash[i]+liquidation-debt[i]-term[i]-closeout
                    creditor[i]=max(0.,-wealth[i]);equity[i]=max(0.,wealth[i])
                    mincash[i]=min(mincash[i],cash[i]-newmargin-c.reserve)
                    breach[i]=bool(np.any(firm>0)) or breach[i]
                    shortgh[i]+=(T-t)*float(gh@firm)
                else:
                    r=repaired;blocks[i]=r['holdings'];sold=beforeblocks-blocks[i]
                    totalsales[i]+=float(sold.sum());debt[i]+=r['draw']-r['repay']
                    financing=r['proceeds']+r['draw']-r['repay']-c.debt_fee*(r['draw']+r['repay'])
                    cash[i]+=financing-newmargin;cashflows+=financing-newmargin;posted[i]=newmargin
                    capacity=np.maximum(0,blocks[i]*tape.availability[i,t]-r['owned'])
                    spot=np.minimum(capacity,tape.demand_blocks[i,t])
                    # Current positive-margin spot only; scale to available prefunding.
                    spot=np.where(tape.local[i,t]>ownedcost,spot,0.)
                    spotcost=float(gh@(spot*ownedcost))
                    available=max(0.,cash[i]-r['cost']-c.reserve)
                    if spotcost>available and spotcost>0:spot*=available/spotcost
                    spotcost=float(gh@(spot*ownedcost));revenue=float(gh@(spot*tape.local[i,t]))
                    mincash[i]=min(mincash[i],cash[i]-r['cost']-spotcost-c.reserve)
                    cash[i]+=-r['cost']-spotcost+revenue+closing
                    cashflows+=-r['cost']-spotcost+revenue+closing
                    amount=float(gh@r['short']);shortgh[i]+=amount;breach[i]|=amount>1e-7
                    maxinventoryerror=max(maxinventoryerror,float(np.max(np.abs(blocks[i]+sold-beforeblocks))))
                    maxdebterror=max(maxdebterror,abs(debt[i]-beforedebt-r['draw']+r['repay']))
                    maxdeliveryerror=max(maxdeliveryerror,float(np.max(np.abs(r['owned']+r['external']+r['short']-firm))))
                    oldh[i]=h
                    oldstrike[i]=marks*np.exp(dt*np.array([c.hedge_forward_carry_annual,c.power_forward_carry_annual]))
            maxcasherror=max(maxcasherror,abs(cash[i]-beforecash-cashflows))
            if i<trace_n:
                for key,val in {'cash':cash[i],'revolver':debt[i],'term_debt':term[i],
                                'blocks':blocks[i].sum(),'margin':posted[i],
                                'compute_hedge':oldh[i,0],'power_hedge':oldh[i,1],
                                'funding_gap':maxgap[i],'local_price':tape.local[i,t,0],
                                'index':tape.index[i,t],'asset_mark':float(blocks[i]@tape.hardware[i,t])}.items():
                    trace[key][i,t]=val
    audit={'max_cash_conservation_error':maxcasherror,'max_inventory_error':maxinventoryerror,
           'max_debt_balance_error':maxdebterror,'max_delivery_equation_error':maxdeliveryerror,
           'nonnegative_remaining_blocks':bool(np.all(blocks>=-1e-8)),
           'all_paths_terminated':bool(not active.any()),
           'failure_semantics':'failure of causal prefunding/collateral/reserve rule; not legal default',
           'terminal_semantics':'cash liquidation less outstanding debt; immediate marked closeout on failure'}
    if max(maxcasherror,maxinventoryerror,maxdebterror,maxdeliveryerror)>1e-5:
        raise RuntimeError(f'simulation conservation audit failed: {audit}')
    return SimulationResult(policy.name,wealth,failed,breach,fs,mincash,maxgap,totalsales,shortgh,hedgepnl,
                            creditor,equity,trace,audit,dict(tape.metadata))
