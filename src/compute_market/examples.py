"""Synthetic fixtures; parameter values are illustrative."""
from __future__ import annotations
import numpy as np
from .schema import Portfolio, Contract
from .tree import ScenarioTree, regular_tree
from .debt import TermLoan
from .contracts import firm_strip


def blank_portfolio(tree: ScenarioTree | None=None, pools: int=1) -> Portfolio:
    tree=tree or ScenarioTree([-1],[1.],[0]); n=tree.size;j=pools
    z=lambda *s:np.zeros(s)
    return Portfolio(
        tree=tree,pool_names=[f"pool-{i}" for i in range(j)],gpus_per_block=np.ones(j),
        initial_blocks=z(j),hours=z(n),year_fraction=z(n),availability=np.ones((n,j)),
        spot_price=z(n,j),spot_demand=z(n,j),owned_unit_cost=z(n,j),
        replacement_price=z(n,j),replacement_limit=z(n,j),asset_sale_price=z(n,j),
        asset_eligible_value=z(n,j),terminal_asset_value=z(n,j),sale_limit=z(n,j),
        initial_cash=0.,cash_growth=np.ones(n),discount=np.ones(n),opening_cashflow=z(n),
        closing_cashflow=z(n),minimum_cash=z(n),terminal_other_liability=z(n),
        facility_names=[],initial_debt=z(0),commitment=z(n,0),draw_limit=z(n,0),
        repayment_limit=z(n,0),facility_rate=z(n,0),advance_rate=z(0,j),
        pledge_fraction=z(0,j),receivable_support=z(n,0),debt_transaction_fee=z(0),
        hedge_names=[],hedge_settlement=z(n,0),hedge_initial_price=z(0),
        hedge_half_spread=z(0),hedge_gross_limit=z(0),hedge_margin=z(n,0),
        hedge_terminal_value=z(n,0))


def synthetic_case(depth: int=3, integer_blocks: bool=False):
    """Two generations, two regions, three branches, debt and two hedges.

    G1/G2, quotes and probabilities are synthetic labels and assumptions."""
    if depth < 1: raise ValueError("depth must be positive")
    tree=regular_tree(depth,[.25,.5,.25]); n=tree.size; j=4
    p=blank_portfolio(tree,j).to_dict()
    initial=np.array([8.,6.,5.,4.]); hardware=np.array([40000.,40000.,60000.,60000.])
    price0=np.array([2.0,2.2,3.0,3.2]); rates=np.array([.42,.48,.6,.65])
    price=np.zeros((n,j)); asset=np.zeros((n,j)); demand=np.zeros((n,j)); avail=np.ones((n,j))
    power=np.zeros(n); factor=np.ones(n); electricity=np.ones(n)
    price[0]=price0;asset[0]=hardware;power[0]=70.
    for node in range(1,n):
        parent=tree.parent[node]; branch=tree.children[parent].index(node)
        shock=[.65,1.,1.35][branch]
        factor[node]=.5*factor[parent]+.5*shock
        electricity[node]=.5*electricity[parent]+.5*[.85,1.,1.6][branch]
        price[node]=price0*factor[node]*(1-.025*tree.stage[node])
        asset[node]=hardware*(.65+.35*factor[node])*(.96**tree.stage[node])
        demand[node]=np.floor(initial*[.6,.9,1.25][branch])
        power[node]=70*electricity[node]
        if branch == 2: avail[node,[1,3]]=.75
    hours=np.where(np.array(tree.stage)>0,720.,0.)
    dt=hours/(24*365)
    operating=rates[None,:]+((power-70.)*.0012)[:,None]
    # Synthetic monthly swaps; long receives index increase at each opening date.
    settlement=np.column_stack([(price[:,0]-price0[0])*5760,(power-70)*60])
    settlement[0]=0
    margin=np.tile([3000.,1800.],(n,1))
    margin[1:]*=np.where((factor[1:]<.9)[:,None],1.5,1.)
    p.update(pool_names=["G1/region-A","G1/region-B","G2/region-A","G2/region-B"],
             gpus_per_block=np.full(j,8.),initial_blocks=initial,hours=hours,year_fraction=np.zeros(n),
             operating_year_fraction=dt,
             availability=avail,spot_price=price,spot_demand=demand,owned_unit_cost=operating,
             replacement_price=1.5*price,replacement_limit=np.full((n,j),2.),
             asset_sale_price=.88*asset,asset_eligible_value=asset,terminal_asset_value=.88*asset,
             sale_limit=np.where((np.arange(n)>0)[:,None],np.full((n,j),2.),0.),
             initial_cash=100000.,opening_cashflow=np.zeros(n),closing_cashflow=np.zeros(n),
             minimum_cash=np.full(n,15000.),facility_names=["hardware-revolver"],
             initial_debt=np.array([380000.]),commitment=np.full((n,1),600000.),
             draw_limit=np.full((n,1),200000.),repayment_limit=np.full((n,1),600000.),
             facility_rate=np.full((n,1),.095),advance_rate=np.full((1,j),.6),
             pledge_fraction=np.ones((1,j)),receivable_support=np.zeros((n,1)),
             debt_transaction_fee=np.array([.001]),hedge_names=["compute-swap","power-swap"],
             hedge_settlement=settlement,hedge_initial_price=np.zeros(2),
             hedge_half_spread=np.array([30.,20.]),hedge_gross_limit=np.array([8.,8.]),
             hedge_margin=margin,hedge_terminal_value=np.zeros((n,2)),
             risk_weight=.3,alpha=.9,integer_blocks=integer_blocks)
    p=Portfolio(**p)
    term=TermLoan("fixed-amortizer",200000.,.075,{depth:50000.})
    service,terminal=term.compile(tree,dt)
    p=p.copy(closing_cashflow=-service,terminal_other_liability=terminal)
    book=firm_strip(p,"existing-commitment",p.pool_names[0],1.,1,depth)
    book.opening_receipts=p.hours*p.gpus_per_block[0]*book.required_blocks[:,0]*2.1
    offered=firm_strip(p,"new-firm-G2",p.pool_names[2],2.,1,depth)
    return p,book,offered
