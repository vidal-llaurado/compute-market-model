import numpy as np
import pytest
from compute_market.tree import ScenarioTree
from compute_market.examples import blank_portfolio
from compute_market.hedge_market import blank_market,NettingSet,HedgeMarket
from compute_market.dynamic_hedging import solve_dynamic
from compute_market.hedge_examples import information_case,dynamic_case


def simple():
    tr=ScenarioTree([-1,0,0],[1,.5,.5],[0,1,1])
    p=blank_portfolio(tr).copy(initial_cash=100.,opening_cashflow=[0,-100,100],risk_weight=1,alpha=.5)
    m=blank_market(tr,['compute']);m.mark=np.array([[100],[200],[0]],float);m.initial_mark=np.array([100.]);m.lower[:]=-2;m.upper[:]=2;m.trade_limit[:]=4
    return p,m


def test_exact_replication():
    p,m=simple();r=solve_dynamic(p,m)
    assert r.value==pytest.approx(100,abs=1e-5)
    assert r.policy['position'][0][0]==pytest.approx(1,abs=1e-5)
    assert np.max(np.abs(r.policy['position'][1:]))==0


def test_incoming_pnl_uses_parent_not_current_position():
    p,m=simple();m.im_per_unit[:]=10
    r=solve_dynamic(p,m,fixed_policy=[[1],[0],[0]])
    assert r.policy['incoming_settlement']==pytest.approx([0,100,-100])
    assert r.policy['financial_cash']==pytest.approx([-10,110,-90])
    assert r.value==pytest.approx(100)


def test_information_has_value_and_no_leaf_peeking():
    p,m=information_case();dynamic=solve_dynamic(p,m);static=solve_dynamic(p,m,mode='static')
    assert dynamic.value==pytest.approx(300,abs=1e-4)
    assert static.value==pytest.approx(250,abs=1e-4)
    assert np.asarray(dynamic.policy['position'])[1:3,0]==pytest.approx([10,20],abs=1e-4)


def test_no_trade_depth_cannot_be_overridden():
    p,m=simple();m.trade_limit[:]=0
    r=solve_dynamic(p,m)
    assert r.value==pytest.approx(0,abs=1e-5)


def test_margin_call_precedes_exit():
    tr=ScenarioTree([-1,0],[1,1],[0,1]);p=blank_portfolio(tr).copy(initial_cash=20)
    m=blank_market(tr,['f']);m.initial_position[:]=1;m.mark[:]=100;m.initial_mark[:]=100
    m.initial_im[:]=10;m.im_per_unit[:]=10;m.im_per_unit[1]=100
    m.trade_limit[:]=2
    a=solve_dynamic(p,m,fixed_policy=[[1],[0]])
    b=solve_dynamic(p,m,fixed_policy=[[1],[0]],enforce_incoming_margin=False)
    assert a.status=='infeasible';assert b.status=='optimal'


def test_initial_margin_is_returned_not_charged_as_expense():
    p,m=simple();m.im_per_unit[:]=20
    r=solve_dynamic(p,m,fixed_policy=[[1],[0],[0]])
    assert r.value==pytest.approx(100);assert r.policy['cash_open'][0]==pytest.approx(80)


def test_actual_turnover_cost_and_impact():
    p,m=simple();p=p.copy(initial_cash=1000);m.half_spread[:]=2;m.impact_slope[:]=3;m.impact_depth[:]=.5
    r=solve_dynamic(p,m,fixed_policy=[[1],[0],[0]])
    assert r.policy['turnover_cost']==pytest.approx([3.5,3.5,3.5]);assert r.value==pytest.approx(993)


def test_purchased_option_premium_and_expiry_cash():
    p,m=simple();p=p.copy(initial_cash=1000,opening_cashflow=np.zeros(3))
    m.kinds=['cash'];m.mark=np.array([[20],[0],[0]],float);m.distribution=np.array([[0],[100],[0]],float)
    m.survives[1:]=0
    r=solve_dynamic(p,m,fixed_policy=[[1],[0],[0]])
    assert r.policy['financial_cash']==pytest.approx([-20,100,0]);assert r.value==pytest.approx(980)
    assert r.policy['buy'][1:]==[[0.],[0.]];assert r.policy['sell'][1:]==[[0.],[0.]]


def otc_case(two=False):
    tr=ScenarioTree([-1,0,1],[1,1,1],[0,1,2]);p=blank_portfolio(tr).copy(initial_cash=1000)
    m=blank_market(tr,['a','b'] if two else ['a'],['otc','otc'] if two else ['otc'])
    k=len(m.names);m.groups=['bilateral']*k;m.netting_sets=[NettingSet('bilateral',0,10,.4)]
    m.default=np.zeros((3,1));m.default[2]=1;m.initial_posted_vm=np.zeros(1);m.initial_received_vm=np.zeros(1)
    m.mark=np.array([[0],[50],[100]],float)*np.ones((1,k));m.survives[2]=0;m.lower[2]=m.upper[2]=0
    return p,m


def test_otc_mtm_not_income_received_collateral_not_spendable():
    p,m=otc_case();r=solve_dynamic(p,m,fixed_policy=[[1],[1],[0]])
    assert r.policy['cash_open'][1]==pytest.approx(1000)
    assert r.policy['received_vm'][1][0]==pytest.approx(40)
    assert r.policy['incoming_settlement'][2]==pytest.approx(64)
    assert r.policy['credit_loss'][2]==pytest.approx(36)
    assert r.value==pytest.approx(1064)


def test_otc_negative_exposure_paid_in_full():
    p,m=otc_case();r=solve_dynamic(p,m,fixed_policy=[[-1],[-1],[0]])
    assert r.policy['cash_open'][1]==pytest.approx(950)
    assert r.policy['posted_vm'][1][0]==pytest.approx(50)
    assert r.policy['financial_cash'][2]==pytest.approx(-50)
    assert r.value==pytest.approx(900)


def test_otc_exposures_net_before_collateral_and_default():
    p,m=otc_case(True);r=solve_dynamic(p,m,fixed_policy=[[1,-1],[1,-1],[0,0]])
    assert np.max(np.abs(r.policy['posted_vm']))<1e-6
    assert np.max(np.abs(r.policy['received_vm']))<1e-6
    assert r.value==pytest.approx(1000)


def test_market_roundtrip(tmp_path):
    p,m=otc_case();f=tmp_path/'m.json';m.validate(p.tree).save(f)
    restored=HedgeMarket.load(f,p.tree)
    assert restored.to_dict()==m.to_dict()

@pytest.mark.parametrize('field,value',[('trade_limit',-1),('im_per_unit',-1),('settlement_recovery',1.1),('im_return',1.1),('survives',.2)])
def test_market_rejects_invalid_arrays(field,value):
    p,m=simple();getattr(m,field)[0,0]=value
    with pytest.raises(ValueError):m.validate(p.tree)


def test_node_exposure_mandate():
    p,m=simple();m.factor_names=['compute'];m.factor_loadings=np.ones((3,1,1))
    m.exposure_lower=np.zeros((3,1));m.exposure_upper=np.full((3,1),.4)
    r=solve_dynamic(p,m);assert r.policy['position'][0][0]==pytest.approx(.4,abs=1e-5)


def test_robust_risk_recursion_with_dynamic_hedges():
    p,m=simple();p=p.copy(ambiguity_radius=[.2,0,0])
    r=solve_dynamic(p,m);assert r.value==pytest.approx(100,abs=1e-5)
    assert r.audit['risk_recursion_error']<1e-6


def test_futures_counterparty_gain_recovery():
    p,m=simple();p=p.copy(initial_cash=1000,opening_cashflow=np.zeros(3));m.settlement_recovery[1]=.4
    r=solve_dynamic(p,m,fixed_policy=[[1],[0],[0]])
    assert r.policy['incoming_settlement']==pytest.approx([0,40,-100])
    assert r.policy['credit_loss']==pytest.approx([0,60,0])


def test_full_case_audit():
    p,m,b,c=dynamic_case(2);r=solve_dynamic(p,m,b,time_limit=60)
    assert r.status=='optimal'
    for key,value in r.audit.items():
        if 'error' in key or 'residual' in key or 'violation' in key:assert value<1e-4,(key,value)
