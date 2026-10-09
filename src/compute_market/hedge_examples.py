"""Synthetic complete physical/financial cases; each instrument has a fixed vintage."""
import numpy as np
from .examples import synthetic_case,blank_portfolio
from .tree import ScenarioTree
from .hedge_market import blank_market,NettingSet
from .dynamic_hedging import without_static_hedges


def conditional_marks(tree,terminal,expiry,discount=1.):
    """Q-prices under explicit supplied tree probabilities; not a carry extrapolation."""
    terminal=np.asarray(terminal,float);out=np.zeros(tree.size)
    for n in reversed(range(tree.size)):
        if tree.stage[n]==expiry:out[n]=terminal[n]
        elif tree.stage[n]<expiry:
            out[n]=discount*sum(tree.probability[c]*out[c] for c in tree.children[n])
    for n in range(1,tree.size):
        if tree.stage[n]>expiry:out[n]=out[tree.parent[n]]
    return out


def dynamic_case(depth=3):
    p,b,c=synthetic_case(depth);p=without_static_hedges(p);tree=p.tree;n=tree.size
    stage=np.asarray(tree.stage)
    names=['G1-compute-near-future','G1-compute-terminal-future','G2-compute-terminal-future',
           'regional-power-future','short-rate-payer-claim','FX-future','compute-put','hardware-floor']
    m=blank_market(tree,names,['futures']*4+['otc','futures','cash','cash'])
    m.groups=['','','','','rates-CSA','','',''];m.netting_sets=[NettingSet('rates-CSA',2000,2000,.4)]
    m.default=np.zeros((n,1));m.initial_posted_vm=np.zeros(1);m.initial_received_vm=np.zeros(1)
    m.lower=np.tile([-15.,-15.,-12.,0.,0.,0.,0.,0.],(n,1));m.upper=np.tile([0.,0.,0.,12.,12.,10.,12.,5.],(n,1))
    m.trade_limit=np.full((n,8),20.);m.half_spread=np.tile([8,8,10,8,10,8,5,5],(n,1))
    m.impact_depth=np.full((n,8),4.);m.impact_slope=np.tile([2,2,2,2,2,2,1,1],(n,1))
    m.multiplier=np.array([5760,5760,5760,60,1.,10000,5760,1.])
    m.im_per_unit=np.tile([1200,1800,2200,600,300,500,0,0.],(n,1))
    # Benchmark and local region basis are separate series.
    g1=.6*p.spot_price[:,0]+.4*p.spot_price[:,1]-.08
    g2=.55*p.spot_price[:,2]+.45*p.spot_price[:,3]-.10
    power=70+(p.owned_unit_cost[:,0]-.42)/.0012
    rate=.04+.008*(g1/g1[0]-1)
    fx=1.30+.045*(power/70-1)-.02*(g1/g1[0]-1)
    p=p.copy(facility_rate=(rate+.055)[:,None])
    expiry=[1,depth,depth,depth,depth,depth,depth,depth]
    series=[g1,g1,g2,power,None,fx,None,None]
    for k in (0,1,2,3,5):m.mark[:,k]=conditional_marks(tree,series[k],expiry[k])
    # Fixed-strike payer claim originated at zero PV. Purchased assignments trade at current PV.
    rate_forward=conditional_marks(tree,rate,depth)[0]
    rate_payoff=2e5*(rate-rate_forward)
    m.mark[:,4]=conditional_marks(tree,rate_payoff,depth,.997)
    put=np.maximum(g1[0]-g1,0.)
    floor=np.maximum(40000*.91-p.asset_eligible_value[:,0],0.)
    m.mark[:,6]=conditional_marks(tree,put,depth,.997)
    m.mark[:,7]=conditional_marks(tree,floor,depth,.997)
    for node in range(n):
        for k in range(8):
            if stage[node]>=expiry[k]:
                m.lower[node,k]=m.upper[node,k]=0.;m.survives[node,k]=0.
                if m.kinds[k]!='futures':
                    if stage[node]==expiry[k]:m.distribution[node,k]=m.mark[node,k]
                    m.mark[node,k]=0.
            if stage[node]>0 and g1[node]<g1[0]*.9:m.im_per_unit[node]*=1.5
    m.initial_mark=m.mark[0].copy()
    return p,m.validate(tree),b,c


def information_case():
    tree=ScenarioTree([-1,0,0,1,1,2,2],[1,.5,.5,.5,.5,.5,.5],[0,1,1,2,2,2,2])
    p=blank_portfolio(tree).copy(initial_cash=300,opening_cashflow=[0,0,0,-100,100,-200,200],risk_weight=1,alpha=.5)
    m=blank_market(tree,['benchmark']);m.mark=np.array([[100],[100],[100],[110],[90],[110],[90]],float)
    m.initial_mark=np.array([100.]);m.lower[:]=0;m.upper[:]=25;m.trade_limit[:]=50
    return p,m.validate(tree)
