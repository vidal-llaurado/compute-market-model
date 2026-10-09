"""Named shocks, policy revaluation and reverse stress along declared rays.

Results condition on each intervention; the interventions have no assigned
occurrence probabilities. Ray searches locate crossings, not a global optimum.
Finite-support ambiguity cannot cover states absent from the support."""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import numpy as np
from .simulation import MarketTape,SimulationConfig,HedgePolicy,simulate
from .statistics import simulation_summary


@dataclass(frozen=True)
class StressScenario:
    name: str
    narrative: str
    start: int=1
    duration: int|None=None
    local_multiplier: tuple[float,...]=(1.,1.,1.,1.)
    benchmark_multiplier: float=1.
    hardware_multiplier: tuple[float,...]=(1.,1.,1.,1.)
    eligible_haircut: float=0.
    power_multiplier: float=1.
    demand_multiplier: float=1.
    availability_multiplier: tuple[float,...]=(1.,1.,1.,1.)
    replacement_multiplier: float=1.
    rate_shift: float=0.
    margin_multiplier: float=1.
    draw_freeze: bool=False
    customer_failure: bool=False
    hedge_counterparty_failure: bool=False
    refinancing_closed: bool=False
    structural_liquidation_fraction: float|None=None

    def __post_init__(self):
        if not isinstance(self.start,int) or self.start<0 or (self.duration is not None and (not isinstance(self.duration,int) or self.duration<=0)):
            raise ValueError('nonnegative start, optional positive duration required')
        for name in ('local_multiplier','hardware_multiplier','availability_multiplier'):
            a=np.asarray(getattr(self,name),float)
            if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0): raise ValueError(f'invalid {name}')
        if np.any(np.array(self.availability_multiplier)>1):raise ValueError('availability multiplier exceeds one')
        for name in ('benchmark_multiplier','power_multiplier','demand_multiplier','replacement_multiplier','margin_multiplier'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0: raise ValueError(f'invalid {name}')
        if self.benchmark_multiplier<=0 or self.power_multiplier<=0 or np.any(np.array(self.local_multiplier)<=0):
            raise ValueError('financial marks must stay positive')
        if not 0<=self.eligible_haircut<=1 or not np.isfinite(self.rate_shift):raise ValueError('invalid lender/rate shock')
        if self.structural_liquidation_fraction is not None and self.start!=0:
            raise ValueError('structural liquidation convention changes require start=0')

    def to_dict(self):return asdict(self)


def apply_stress(tape:MarketTape,stress:StressScenario)->MarketTape:
    out=tape.copy();c=tape.config
    if stress.start>c.steps:raise ValueError('stress starts after horizon')
    end=c.steps+1 if stress.duration is None else min(c.steps+1,stress.start+stress.duration)
    sl=slice(stress.start,end)
    out.local[:,sl]*=np.array(stress.local_multiplier)
    out.index[:,sl]*=stress.benchmark_multiplier
    out.hardware[:,sl]*=np.array(stress.hardware_multiplier)
    # Market asset shock reaches lender marks after their stated lag. The explicit
    # eligible haircut is an immediate separate lender eligibility/advance shock.
    shifted=slice(min(c.steps+1,stress.start+c.eligible_mark_lag),min(c.steps+1,end+c.eligible_mark_lag))
    out.eligible[:,shifted]*=np.array(stress.hardware_multiplier)
    out.eligible[:,sl]*=1-stress.eligible_haircut
    out.power[:,sl]*=stress.power_multiplier
    out.demand_blocks[:,sl]*=stress.demand_multiplier
    out.availability[:,sl]*=np.array(stress.availability_multiplier)
    out.replacement[:,sl]*=stress.replacement_multiplier
    out.reference_rate[:,sl]+=stress.rate_shift
    out.margin_multiplier[:,sl]*=stress.margin_multiplier
    if stress.draw_freeze:out.freeze[:,sl]=True
    # Credit defaults are absorbing even when the exogenous shock window expires.
    if stress.customer_failure:out.customer_alive[:,stress.start:]=False
    if stress.hedge_counterparty_failure:out.counterparty_alive[:,stress.start:]=False
    if stress.refinancing_closed:out.refinance_u[:,sl]=1.
    if stress.structural_liquidation_fraction is not None:
        out.config=replace(c,liquidation_fraction=stress.structural_liquidation_fraction)
    out.metadata={**tape.metadata,'stress':stress.to_dict(),
                  'probability_note':'conditional on named scenario; scenario likelihood is not assigned'}
    return out.validate()


def standard_stresses(steps=24):
    end=min(18,steps)
    return [
        StressScenario('oversupply','Broad rental and demand decline; financial index falls with rentals.',
                       local_multiplier=(.55,)*4,benchmark_multiplier=.55,demand_multiplier=.65,hardware_multiplier=(.75,)*4),
        StressScenario('generation_transition','Older generation prices and assets fall more than newer generation.',
                       local_multiplier=(.4,.4,.85,.85),benchmark_multiplier=.4,
                       hardware_multiplier=(.45,.45,.85,.85),demand_multiplier=.8),
        StressScenario('local_index_basis','Regional rental revenues fall while financial benchmark is unchanged.',
                       local_multiplier=(.45,.6,.7,.7),demand_multiplier=.75),
        StressScenario('power_price_spike','Electricity price triples without matching compute-price gains.',power_multiplier=3.),
        StressScenario('regional_outage','Region A outage with scarce replacement blocks.',duration=min(3,steps),
                       availability_multiplier=(.15,1.,.15,1.),replacement_multiplier=.15),
        StressScenario('customer_and_receivables','Major customer stops paying; associated receivable support disappears.',
                       customer_failure=True),
        StressScenario('collateral_and_draw_freeze','Lender collateral markdown, equipment decline and no new draws.',
                       hardware_multiplier=(.65,)*4,eligible_haircut=.25,draw_freeze=True),
        StressScenario('refinancing_closed','No refinancing offer at contractual revolver maturity.',start=end,
                       refinancing_closed=True,draw_freeze=True),
        StressScenario('margin_squeeze','Collateral demand triples; index rises against a short revenue hedge.',
                       benchmark_multiplier=1.7,margin_multiplier=3.,draw_freeze=True),
        StressScenario('hedge_counterparty_failure','Positive hedge payouts impaired; no subsequent trades with failed counterparty.',
                       local_multiplier=(.65,)*4,benchmark_multiplier=.65,hedge_counterparty_failure=True),
        StressScenario('funding_rate_shock','Reference rates rise 600 basis points; fixed term coupon does not reset.',rate_shift=.06),
        StressScenario('liquidation_illiquidity','Structural lower realization prices across the whole horizon.',start=0,
                       structural_liquidation_fraction=.65),
        StressScenario('compound_wrong_way','Joint local revenue, collateral, customer, replacement and funding shock.',
                       local_multiplier=(.4,.5,.6,.6),benchmark_multiplier=.8,hardware_multiplier=(.5,.5,.65,.65),
                       eligible_haircut=.2,demand_multiplier=.5,power_multiplier=1.8,
                       availability_multiplier=(.6,.8,.6,.8),replacement_multiplier=.25,
                       margin_multiplier=2.5,draw_freeze=True,customer_failure=True),
    ]


def severity_scenario(severity:float,start=1):
    if not 0<=severity<=1:raise ValueError('ray severity must be in [0,1]')
    return StressScenario('reverse_joint_ray','Prespecified joint adverse ray, not a probabilistic event.',start=start,
        local_multiplier=(1-.7*severity,)*4,benchmark_multiplier=1-.3*severity,
        hardware_multiplier=(1-.65*severity,)*4,eligible_haircut=.3*severity,
        demand_multiplier=1-.65*severity,power_multiplier=1+1.5*severity,
        margin_multiplier=1+3*severity,replacement_multiplier=1-.7*severity,
        availability_multiplier=(1-.5*severity,)*4)


def reverse_stress(tape,policy,target=.10,grid_points=11,refinements=5):
    """Locate the first failure-rate crossing on a ray using common paths.

    Evaluate the full coarse grid to expose nonmonotonicity, then refine its first
    crossing. Judge near-threshold estimates with their binomial sampling intervals."""
    if not 0<target<1 or grid_points<3 or refinements<0:raise ValueError('invalid reverse stress controls')
    from .statistics import wilson_interval
    rows=[]
    def run(s):
        r=simulate(apply_stress(tape,severity_scenario(float(s))),policy,trace_paths=0)
        ci=wilson_interval(r.financing_failure)
        row={'severity':float(s),**ci};rows.append(row);return row
    grid=[run(s) for s in np.linspace(0,1,grid_points)]
    hits=[i for i,row in enumerate(grid) if row['probability']>=target]
    interval=None
    if hits:
        i=hits[0]
        if i==0:interval=[0.,0.]
        else:
            lo,hi=grid[i-1]['severity'],grid[i]['severity']
            for _ in range(refinements):
                mid=(lo+hi)/2;row=run(mid)
                if row['probability']>=target:hi=mid
                else:lo=mid
            interval=[lo,hi]
    return {'target_failure_probability':target,'crossing_bracket':interval,
            'grid_nonmonotone':any(grid[i+1]['probability']<grid[i]['probability'] for i in range(len(grid)-1)),
            'evaluations':sorted(rows,key=lambda x:x['severity']),
            'interpretation':'first sampled crossing along specified ray; not global nearest-failure distance',
            'scenario_probability':None}


def optimizer_stress_suite(portfolio,book):
    """Reoptimize shocked finite programs and replay the nominal policy.

    Infeasibility rules out policies within the supplied program. Feasibility says
    nothing about unseen paths. Frozen-policy violations locate broken constraints."""
    from .optimizer import solve
    from .validation import independent_policy_audit
    cases={
        'base':portfolio,
        'rental_down_40pct':portfolio.copy(spot_price=portfolio.spot_price*.6,spot_demand=portfolio.spot_demand*.7),
        'collateral_down_35pct':portfolio.copy(asset_eligible_value=portfolio.asset_eligible_value*.65),
        'replacement_unavailable':portfolio.copy(replacement_limit=np.zeros_like(portfolio.replacement_limit),availability=portfolio.availability*.5),
        'no_draws':portfolio.copy(draw_limit=np.zeros_like(portfolio.draw_limit)),
        'margin_tripled':portfolio.copy(hedge_margin=portfolio.hedge_margin*3),
        'TV_radius_10pct':portfolio.copy(ambiguity_radius=np.full(portfolio.tree.size,.1)),
        'compound':portfolio.copy(asset_eligible_value=portfolio.asset_eligible_value*.45,
            asset_sale_price=portfolio.asset_sale_price*.5,terminal_asset_value=portfolio.terminal_asset_value*.5,
            spot_price=portfolio.spot_price*.5,draw_limit=np.zeros_like(portfolio.draw_limit),
            availability=portfolio.availability*.65,replacement_limit=portfolio.replacement_limit*.25)
    }
    nominal=solve(portfolio,book);out=[]
    for name,p in cases.items():
        result=solve(p,book)
        frozen=independent_policy_audit(p,book,nominal.policy) if nominal.status=='optimal' else None
        fixed=solve(p,book,fixed_hedge=nominal.policy['hedge_net']) if nominal.status=='optimal' else None
        out.append({'scenario':name,'status':result.status,'reoptimized_value':result.value,
                    'value_change':None if result.value is None else result.value-nominal.value,
                    'audit':result.audit,'frozen_policy_equation_diagnostics':frozen,
                    'fixed_initial_hedge_status':None if fixed is None else fixed.status,
                    'fixed_initial_hedge_value':None if fixed is None else fixed.value,
                    'interpretation':'full reoptimization within represented finite observable tree'})
    return out
