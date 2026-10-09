import numpy as np
import pytest
from compute_market.risk import expected_shortfall,risk,certainty_equivalent,nested_value
from compute_market.tree import ScenarioTree,regular_tree
from compute_market.examples import blank_portfolio
from compute_market.schema import Portfolio,Contract
from compute_market.optimizer import solve
from compute_market.contracts import firm_strip,combine,deterministic_gpu_hours
from compute_market.pricing import reservation_price,quote_menu


def one_period(**changes):
    tree=ScenarioTree([-1,0],[1.,1.],[0,1])
    p=blank_portfolio(tree).copy(initial_cash=100.,initial_blocks=[1.],hours=[0.,1.],
        spot_price=[[0.],[10.]],spot_demand=[[0.],[1.]],owned_unit_cost=[[0.],[2.]])
    return p.copy(**changes) if changes else p

@pytest.mark.parametrize('alpha,expected',[(0,1.),(.5,2.),(.75,4.),(.9,10.),(.99,10.)])
def test_es_fractional_atom(alpha,expected):
    assert expected_shortfall([0,10],[.9,.1],alpha)==pytest.approx(expected)

@pytest.mark.parametrize('alpha',[-.1,1.,np.nan])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError): expected_shortfall([1],[1],alpha)

@pytest.mark.parametrize('p',[[.5,.4],[-.1,1.1],[np.nan,1]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError): risk([1,2],p)


def test_risk_cash_translation():
    assert risk(np.array([1.,7.])+13,[.7,.3])==pytest.approx(risk([1.,7.],[.7,.3])+13)
    assert certainty_equivalent([4,4],[.1,.9])==pytest.approx(4)


def test_nested_not_flat():
    t=regular_tree(2,[.5,.5]); v=np.zeros(t.size);v[t.leaves]=[0,0,0,100]
    nested=nested_value(t,v,.5,.5)[0]
    flat=certainty_equivalent(v[t.leaves],[.25]*4,.5,.5)
    assert nested != pytest.approx(flat)


def test_tree_probabilities_paths():
    t=regular_tree(3,[.2,.8])
    assert t.unconditional[t.leaves].sum()==pytest.approx(1)
    assert len(t.path(t.leaves[0]))==4

@pytest.mark.parametrize('args',[
    ([-1,1],[1,1],[0,1]),([-1,0],[1,.9],[0,1]),([-1,0],[1,1],[0,2]),
    ([-1,0],[1,0],[0,1])])
def test_tree_invalid(args):
    with pytest.raises(ValueError): ScenarioTree(*args)


def test_roundtrip_empty_dimensions(tmp_path):
    p=blank_portfolio(); path=tmp_path/'p.json';p.save(path)
    q=Portfolio.load(path)
    assert q.advance_rate.shape==(0,1)
    assert q.to_dict()==p.to_dict()


def test_pure_cash():
    p=blank_portfolio().copy(initial_cash=27.)
    r=solve(p)
    assert r.value==pytest.approx(27)
    assert r.duals['marginal_value_initial_cash']==pytest.approx(1)


def test_spot_profit():
    r=solve(one_period())
    assert r.value==pytest.approx(108)
    assert r.policy['spot_blocks'][1][0]==pytest.approx(1)
    assert r.audit['risk_recursion_error']<1e-8


def test_reject_negative_margin():
    r=solve(one_period(owned_unit_cost=[[0],[11]]))
    assert r.policy['spot_blocks'][1][0]==pytest.approx(0)
    assert r.value==pytest.approx(100)


def test_more_cash_monotone():
    p=one_period(initial_cash=0)
    assert solve(p.copy(initial_cash=2)).value >= solve(p).value+2-1e-8


def test_future_receipts_do_not_pay_earlier_bill():
    p=one_period(initial_cash=0,opening_cashflow=[-1,0],closing_cashflow=[0,100])
    assert solve(p).status=='infeasible'


def test_operating_costs_must_be_prefunded():
    r=solve(one_period(initial_cash=1))
    assert r.policy['spot_blocks'][1][0]==pytest.approx(.5)
    assert r.value==pytest.approx(5)


def test_firm_delivery_price_opportunity_cost():
    p=one_period();c=firm_strip(p,'unit','pool-0',1,1,1)
    quote=reservation_price(p,c,price_tolerance=1e-5)
    assert quote.amount==pytest.approx(10,abs=2e-5)


def test_rate_payment():
    p=one_period();c=firm_strip(p,'unit','pool-0',1,1,1)
    q=reservation_price(p,c,payment='closing_rate',price_tolerance=1e-6)
    assert q.amount==pytest.approx(10,abs=2e-6)


def test_no_capacity_cannot_be_fixed_by_high_price():
    p=one_period(initial_blocks=[0])
    c=firm_strip(p,'unit','pool-0',1,1,1)
    assert solve(p,c,premium=1000).status=='infeasible'
    with pytest.raises(ValueError,match='no admissible'):
        reservation_price(p,c,max_absolute_price=1000)


def test_prepaid_contract_can_cure_own_operating_funding():
    p=one_period(initial_cash=0)
    c=firm_strip(p,'unit','pool-0',1,1,1)
    q=reservation_price(p,c,price_tolerance=1e-5)
    assert q.amount==pytest.approx(2,abs=2e-5)
    with pytest.raises(ValueError,match='no admissible'):
        reservation_price(p,c,payment='closing_rate',max_absolute_price=100)


def test_infeasible_baseline_not_quoted():
    p=one_period(opening_cashflow=[-101,0])
    c=firm_strip(p,'unit','pool-0',1,1,1)
    with pytest.raises(ValueError,match='baseline'):reservation_price(p,c)


def test_negative_reservation_price_allowed():
    p=blank_portfolio().copy(initial_cash=100)
    c=Contract.empty(p);c.opening_receipts[0]=10
    q=reservation_price(p,c,price_tolerance=1e-5)
    assert q.amount==pytest.approx(-10,abs=2e-5)


def test_quote_menu_has_no_quote_option():
    p=one_period();c=firm_strip(p,'unit','pool-0',1,1,1)
    assert quote_menu(p,c,[8,9],[1,.5])['decision']=='no_quote'
    assert quote_menu(p,c,[11,15],[1,.8])['quote']==15


def test_integer_capacity_not_fractional():
    p=one_period(initial_cash=1,integer_blocks=True)
    r=solve(p)
    assert r.value==pytest.approx(1)
    assert r.audit['max_integrality_error']<1e-9


def test_expiry_no_future_service():
    p=one_period();c=firm_strip(p,'unit','pool-0',1,1,1)
    assert deterministic_gpu_hours(p,c)==1
    assert c.required_blocks[0,0]==0


def test_terminal_debt_not_erased():
    p=blank_portfolio().copy(initial_cash=100,terminal_other_liability=[30])
    assert solve(p).value==pytest.approx(70)
    assert solve(p.copy(terminal_other_liability=[101])).status=='infeasible'


def test_shortfall_needs_explicit_permission():
    p=one_period(initial_blocks=[0]);c=firm_strip(p,'unit','pool-0',1,1,1)
    c.shortfall_limit[:]=c.required_blocks;c.shortfall_cost[1,0]=5
    assert solve(p,c).value==pytest.approx(95)


def test_aggregation_rejects_different_optional_penalties():
    p=one_period();a=firm_strip(p,'a','pool-0',1,1,1);b=firm_strip(p,'b','pool-0',1,1,1)
    for c in (a,b):c.shortfall_limit[:]=c.required_blocks
    a.shortfall_cost[1,0]=1;b.shortfall_cost[1,0]=2
    with pytest.raises(ValueError):combine(a,b,p)
