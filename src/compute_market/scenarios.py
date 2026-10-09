"""Fixed-horizon sampled scenario trees for conditional-support refinement.

The branch count changes numerical resolution, never contractual delivery dates.
The example distribution is a specified correlated lognormal/jump factor model.
"""
from __future__ import annotations
import numpy as np
from .tree import regular_tree
from .examples import blank_portfolio
from .schema import Contract
from .contracts import firm_strip
from .hedge_market import blank_market
from .hedge_examples import conditional_marks
from .case import ValuationCase


def sampled_case(branches=16,periods=2,seed=1201,alpha=.9):
    if not isinstance(branches,int) or branches<2 or not isinstance(periods,int) or periods<1:
        raise ValueError('at least two branches and a positive fixed delivery horizon required')
    if sum(branches**i for i in range(periods+1))>100000:raise ValueError('scenario budget exceeds 100000 nodes')
    tree=regular_tree(periods,[1/branches]*branches);n=tree.size;j=2
    rng=np.random.default_rng(seed);x=np.zeros((n,4));outage=np.ones((n,j))
    chol=np.linalg.cholesky([[1,.25,-.1,.25],[.25,1,.05,.2],[-.1,.05,1,.15],[.25,.2,.15,1]])
    for parent,children in enumerate(tree.children):
        if not children:continue
        shock=rng.normal(size=(branches,4))@chol.T
        jump=rng.random(branches)<.04
        x[children]=.65*x[parent]+shock*np.array([.19,.14,.23,.08])
        x[children,0]-=.4*jump;x[children,2]+=.35*jump
        outage[children,1]=np.where(rng.random(branches)<.035,.5,1.)
    stage=np.asarray(tree.stage);hours=np.where(stage>0,720.,0.);dt=hours/(24*365)
    index=2*np.exp(x[:,0]-.025*stage);power=70*np.exp(x[:,2])
    local=np.column_stack([index,index*1.1*np.exp(x[:,1])])
    asset=np.column_stack([160000*np.exp(.3*x[:,0]+x[:,3]-.015*stage)]*j)
    demand=np.column_stack([5*np.exp(.3*x[:,0]),4*np.exp(.3*x[:,0]+.15*x[:,1])]);demand[0]=0
    p=blank_portfolio(tree,j).copy(pool_names=['G1/A','G1/B'],gpus_per_block=[8,8],initial_blocks=[6,5],
        hours=hours,operating_year_fraction=dt,spot_price=local,spot_demand=demand,availability=outage,
        owned_unit_cost=np.column_stack([.32+.0012*power,.35+.0012*power]),
        replacement_price=1.5*local,replacement_limit=np.full((n,j),2.),
        asset_sale_price=.88*asset,asset_eligible_value=asset,terminal_asset_value=.88*asset,
        sale_limit=np.zeros((n,j)),initial_cash=250000.,minimum_cash=np.full(n,30000.),
        closing_cashflow=np.where(stage>0,-35000.,0.),facility_names=['asset-revolver'],initial_debt=[650000.],
        commitment=np.full((n,1),1000000.),draw_limit=np.full((n,1),200000.),repayment_limit=np.full((n,1),1000000.),
        facility_rate=np.full((n,1),.095),advance_rate=np.full((1,j),.6),pledge_fraction=np.ones((1,j)),
        receivable_support=np.zeros((n,1)),debt_transaction_fee=[.001],risk_weight=.35,alpha=alpha)
    m=blank_market(tree,['compute-future','power-future'])
    m.multiplier=np.array([5760.,60.]);m.mark[:,0]=conditional_marks(tree,index,periods)
    m.mark[:,1]=conditional_marks(tree,power,periods);m.initial_mark=m.mark[0].copy()
    m.lower[:]=[-10.,0.];m.upper[:]=[0.,10.];m.trade_limit[:]=20.
    m.im_per_unit[:]=[1500.,600.];m.half_spread[:]=[8.,8.];m.impact_depth[:]=20.;m.impact_slope[:]=0.
    m.survives[tree.leaves]=0.
    book=firm_strip(p,'existing-service','G1/A',1.,1,periods)
    book.opening_receipts=p.hours*8*2.1*book.required_blocks[:,0]
    offered=firm_strip(p,'offered-service','G1/B',1.,1,periods)
    return ValuationCase('sampled compute producer',p,m.validate(tree),book,offered,
        {'data':'synthetic','branches':branches,'delivery_intervals':periods,'hours_per_interval':720,
         'seed':seed,'price_measure':'supplied scenario probabilities price benchmark futures',
         'experiment':'fixed-horizon conditional scenario refinement'})
