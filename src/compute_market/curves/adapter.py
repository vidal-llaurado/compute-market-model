"""Sample the curve law into a finite control program.

Branching controls numerical resolution; the law and delivery horizon stay fixed."""
from __future__ import annotations
import numpy as np
from .portfolio import CurveCase,POOL_MARKETS,spot_state,hardware_marks
from ..tree import regular_tree
from ..examples import blank_portfolio
from ..schema import Contract
from ..contracts import firm_strip
from ..hedge_market import blank_market
from ..case import ValuationCase


def finite_case(case:CurveCase,branches=16,periods=2,seed=123,risk_weight=.35,alpha=.9,initial_state=None):
    if not isinstance(branches,int) or branches<2 or not isinstance(periods,int) or periods<1:raise ValueError('branching >=2 and positive periods required')
    if sum(branches**k for k in range(periods+1))>40000:raise ValueError('finite validation node budget exceeded')
    m=case.model;c=case.fleet;horizon=c.steps*c.year_fraction;dt=horizon/periods
    tree=regular_tree(periods,[1/branches]*branches);N=tree.size;stage=np.asarray(tree.stage);times=stage*dt
    X=np.zeros((N,m.dimension));rng=np.random.default_rng(seed)
    if initial_state is not None:X[0]=np.asarray(initial_state,float)
    for parent,ch in enumerate(tree.children):
        if ch:X[ch]=m.step(np.broadcast_to(X[parent],(len(ch),m.dimension)),dt,rng,'P')[0]
    local=np.zeros((N,4));cost=np.zeros_like(local);hw=np.zeros_like(local);rates=np.zeros(N)
    for s in range(periods+1):
        ix=np.flatnonzero(stage==s);t=s*dt;local[ix],_,cost[ix]=spot_state(m,t,X[ix],c);hw[ix]=hardware_marks(m,t,X[ix],c)
        rates[ix]=m.short_rate(t,X[ix])+m.value('funding',t,X[ix],t)
    if np.any(cost<0):raise ValueError('finite delivery backend requires nonnegative all-in prefunded unit costs')
    eligible=hw.copy()
    for n in range(1,N):eligible[n]=hw[tree.parent[n]]
    hours=np.where(stage>0,dt*8760,0.);demand=np.asarray(c.blocks)*c.expected_spot_utilization*np.exp(.3*X[:,0,None]);demand[0]=0
    alive=np.ones((N,4));a=c.term_amortization*(c.steps/periods);remaining=np.maximum(c.initial_term_debt-a*stage,0.)
    previous=np.maximum(c.initial_term_debt-a*np.maximum(stage-1,0),0.)
    amort=np.where(stage>0,np.minimum(previous,a),0.);coupon=np.where(stage>0,previous*c.fixed_term_coupon*dt,0.)
    p=blank_portfolio(tree,4).copy(pool_names=list(POOL_MARKETS),gpus_per_block=c.gpus_per_block,initial_blocks=c.blocks,
      hours=hours,operating_year_fraction=np.where(stage>0,dt,0.),spot_price=local,spot_demand=demand,
      owned_unit_cost=cost,replacement_price=c.replacement_multiplier*local,replacement_limit=np.broadcast_to(c.replacement_blocks,(N,4)),availability=alive,
      asset_sale_price=c.liquidation_fraction*hw,asset_eligible_value=eligible,terminal_asset_value=(c.liquidation_fraction-c.liquidation_impact)*hw,
      sale_limit=np.zeros((N,4)),initial_cash=c.initial_cash,minimum_cash=np.full(N,c.reserve),opening_cashflow=-amort-coupon,
      closing_cashflow=np.where(stage>0,-c.fixed_overhead_per_step*c.steps/periods,0.),terminal_other_liability=remaining,
      facility_names=['hardware revolver'],initial_debt=[c.initial_revolver],commitment=np.full((N,1),c.revolver_commitment),
      draw_limit=np.full((N,1),c.max_draw_per_step),repayment_limit=np.full((N,1),c.revolver_commitment),facility_rate=rates[:,None],
      advance_rate=np.full((1,4),c.advance_rate),pledge_fraction=np.ones((1,4)),receivable_support=np.zeros((N,1)),debt_transaction_fee=[c.debt_fee],
      risk_weight=risk_weight,alpha=alpha)
    grid=np.unique(times)
    inst=[a for a in case.instruments if np.isclose(a.maturity,horizon)
          and all(np.min(abs(grid-T))<1e-9 for T in a.event_dates)]
    if not inst:raise ValueError('a horizon hedge with cash events on the finite grid is required')
    from .registry import export_claims
    h=export_claims(m,inst,tree,times,X)
    # This pricing fixture uses a single linear execution tier; nonlinear cash
    # marks and the OTC collateral identities still retain their own structure.
    h.impact_depth[:]=h.trade_limit;h.impact_slope[:]=0.
    book=Contract.empty(p,'existing physical book');book.required_blocks[stage>0]=c.firm_blocks
    payments=hours*(np.asarray(c.gpus_per_block)*np.asarray(c.firm_blocks)*np.asarray(c.firm_rates)).sum()
    book.opening_receipts=payments*c.prepayment_fraction;book.closing_receipts=payments*(1-c.prepayment_fraction)
    offer=case.offered or {'market':'G1_SG','additional_blocks':1.}
    offered=firm_strip(p,'offered capacity',offer['market'],float(offer['additional_blocks']),1,periods)
    return ValuationCase(case.name,p,h.validate(tree),book,offered,
       {'source':'stochastic forward curves','curve_model_sha256':m.fingerprint,'curve_case_sha256':case.fingerprint,
        'sample_measure':'P','seed':seed,'branches':branches,'periods':periods,'horizon_years':horizon,
        'factor_states':X.tolist(),'dates':times.tolist(),'purpose':'finite-control pricing and independent accounting validation',
        'physical_scope':'deterministic available pools, continuous blocks, no hardware sale in this fixture',
        'instrument_names':[z.name for z in inst],
        'excluded_cash_dates':[z.name for z in case.instruments if z not in inst],
        'mandate':'joint total-business risk value with signed instrument bounds'})


def quote_from_curves(case,branches=16,periods=2,seed=123,**kwargs):
    finite=finite_case(case,branches,periods,seed)
    result=finite.quote(**kwargs);result['curve_case_sha256']=case.fingerprint;result['curve_model_sha256']=case.model.fingerprint
    result['numerical_representation']={'method':'sampled finite control program','branches':branches,'periods':periods,'horizon':case.fleet.steps*case.fleet.year_fraction,'nodes':finite.portfolio.tree.size}
    return result


def capacity_shadow(case,market,bumps=(.1,.25),branches=8,periods=2,seed=123):
    """Private per-block inventory shadow value; not an observed forward quote."""
    finite=finite_case(case,branches,periods,seed);base=finite.solve().value;j=finite.portfolio.pool_names.index(market);out=[]
    for bump in bumps:
        if bump<=0:raise ValueError('positive marginal inventory bump required')
        blocks=finite.portfolio.initial_blocks.copy();blocks[j]+=bump
        candidate=ValuationCase(finite.name,finite.portfolio.copy(initial_blocks=blocks),finite.market,finite.book,finite.offered,finite.assumptions)
        value=candidate.solve().value;out.append({'additional_blocks':bump,'marginal_private_value_per_block':(value-base)/bump})
    return {'market':market,'base_value':base,'estimates':out,'units':'USD per physical block over remaining lifetime and obligations'}
