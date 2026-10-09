import numpy as np
import pytest
from compute_market.ambiguity import robust_risk,robust_nested_value
from compute_market.risk import risk
from compute_market.examples import blank_portfolio,synthetic_case
from compute_market.tree import regular_tree
from compute_market.optimizer import solve

@pytest.mark.parametrize('alpha',[0.,.5,.95])
@pytest.mark.parametrize('weight',[0.,.3,1.])
def test_zero_ambiguity_recovers_nominal(alpha,weight):
    x=[-7.,2.,11.];p=[.25,.5,.25]
    assert robust_risk(x,p,alpha,weight,0)['risk']==pytest.approx(risk(x,p,alpha,weight))

@pytest.mark.parametrize('radius',[.01,.1,.4,1.])
def test_mean_tv_two_state_exact(radius):
    answer=robust_risk([0.,10.],[.7,.3],.9,0.,radius)
    assert answer['risk']==pytest.approx(min(1.,.3+radius)*10)
    assert answer['tv_distance'] <= radius+1e-8

def test_support_monotonic_cash_translation():
    previous=-np.inf
    for r in [0,.03,.1,.3,1]:
        v=robust_risk([1,3,20],[.7,.25,.05],.8,.4,r)['risk']
        assert v >= previous-1e-9
        assert robust_risk([8,10,27],[.7,.25,.05],.8,.4,r)['risk']==pytest.approx(v+7)
        previous=v

def test_dual_inside_optimizer_matches_independent_support():
    tree=regular_tree(1,[.7,.3]);p=blank_portfolio(tree)
    p=p.copy(initial_cash=100.,closing_cashflow=[0,0,-10],ambiguity_radius=[.2,0,0],risk_weight=.4,alpha=.5)
    fit=solve(p)
    wanted=-robust_risk([-100,-90],[.7,.3],.5,.4,.2)['risk']
    assert fit.value==pytest.approx(wanted)
    assert fit.audit['risk_recursion_error']<1e-7

@pytest.mark.parametrize('integer',[False,True])
def test_real_book_robust_value_not_above_nominal(integer):
    p,b,_=synthetic_case(depth=1,integer_blocks=integer)
    a=solve(p,b);r=solve(p.copy(ambiguity_radius=np.full(p.tree.size,.1)),b)
    assert r.status=='optimal'
    assert r.value<=a.value+1e-5
    assert r.audit['risk_recursion_error']<1e-4

@pytest.mark.parametrize('radius',[-.1,1.1,float('nan')])
def test_invalid_radius(radius):
    with pytest.raises(ValueError): robust_risk([0,1],[.5,.5],radius=radius)


def test_fixed_root_hedge_restriction_is_exact():
    p,b,_=synthetic_case(depth=1)
    unrestricted=solve(p,b)
    zero=solve(p,b,fixed_hedge=np.zeros(len(p.hedge_names)))
    assert zero.policy['hedge_net']==pytest.approx([0,0])
    assert zero.value<=unrestricted.value+1e-6
    with pytest.raises(ValueError):solve(p,b,fixed_hedge=[100,0])

@pytest.mark.parametrize('weight',[0,.3,1])
def test_full_tv_radius_equals_worst_support_loss(weight):
    x=[-2,3,8,4];p=[.4,.2,.1,.3]
    assert robust_risk(x,p,.8,weight,1.)['risk']==pytest.approx(8)
