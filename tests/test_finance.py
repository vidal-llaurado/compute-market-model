import numpy as np
import pytest
from compute_market.examples import blank_portfolio
from compute_market.tree import ScenarioTree,regular_tree
from compute_market.optimizer import solve
from compute_market.schema import Contract
from compute_market.debt import TermLoan,collateral_cure


def financed(tree=None):
    tree=tree or ScenarioTree([-1,0],[1.,1.],[0,1]); n=tree.size
    return blank_portfolio(tree).copy(initial_cash=20,initial_blocks=[1],
        asset_sale_price=np.full((n,1),100),asset_eligible_value=np.full((n,1),100),
        terminal_asset_value=np.full((n,1),100),facility_names=['loan'],initial_debt=[50],
        commitment=np.full((n,1),70),draw_limit=np.zeros((n,1)),repayment_limit=np.full((n,1),100),
        facility_rate=np.zeros((n,1)),advance_rate=[[.7]],pledge_fraction=[[1]],
        receivable_support=np.zeros((n,1)),debt_transaction_fee=[0])


def test_debt_proceeds_not_income():
    p=financed().copy(initial_cash=70)
    q=blank_portfolio(p.tree).copy(initial_cash=20,initial_blocks=[1],terminal_asset_value=[[100],[100]])
    assert solve(p).value==pytest.approx(solve(q).value)


def test_borrowing_base_cure_cash():
    p=financed().copy(asset_eligible_value=[[100],[50]])
    r=solve(p)
    assert r.status=='optimal'
    assert r.policy['debt'][1][0] <= 35+1e-8
    assert r.policy['cash_close'][1] <= 5+1e-8


def test_covenant_can_force_infeasible():
    p=financed().copy(asset_eligible_value=[[100],[20]])
    assert solve(p).status=='infeasible'


def test_asset_sale_cures_and_reduces_future_holdings():
    p=financed().copy(initial_cash=5,asset_eligible_value=[[100],[50]],
                       asset_sale_price=[[100],[50]],sale_limit=[[0],[1]])
    r=solve(p)
    assert r.status=='optimal'
    assert r.policy['asset_sales'][1][0] > 0
    assert r.policy['holdings'][1][0] < 1


def test_commitment_maturity_forces_repayment():
    p=financed().copy(initial_cash=60,commitment=[[70],[0]])
    r=solve(p)
    assert r.policy['debt'][1][0]==pytest.approx(0)
    assert r.policy['cash_close'][1]==pytest.approx(10)


def test_debt_interest_on_inherited_not_new_draw():
    p=financed().copy(year_fraction=[0,1],facility_rate=[[.1],[.1]])
    r=solve(p)
    # The model must KEEP CASH for the next coupon; it cannot repay all 20.
    # With r paid now: 20-r = .1*(50-r), hence r=50/3.
    assert r.value==pytest.approx(200/3)
    assert r.policy['repay'][0][0]==pytest.approx(50/3)


def test_double_pledge_rejected():
    p=financed();n=p.tree.size
    with pytest.raises(ValueError,match='pledged'):
        p.copy(facility_names=['a','b'],initial_debt=[25,25],commitment=[[70,70]]*n,
               draw_limit=[[0,0]]*n,repayment_limit=[[100,100]]*n,facility_rate=[[0,0]]*n,
               advance_rate=[[.7],[.7]],pledge_fraction=[[.6],[.6]],
               receivable_support=[[0,0]]*n,debt_transaction_fee=[0,0])


def test_two_facilities_with_disjoint_pledge():
    p=financed();n=p.tree.size
    q=p.copy(facility_names=['a','b'],initial_debt=[25,25],commitment=[[70,70]]*n,
               draw_limit=[[0,0]]*n,repayment_limit=[[100,100]]*n,facility_rate=[[0,0]]*n,
               advance_rate=[[.7],[.7]],pledge_fraction=[[.5],[.5]],
               receivable_support=[[0,0]]*n,debt_transaction_fee=[0,0])
    assert solve(q).value==pytest.approx(solve(p).value)


def test_cure_amplification():
    a=collateral_cure(7e6,7e6,1e6,.7,1)
    b=collateral_cure(7e6,7e6,1e6,.7,.9)
    assert a['marked_asset_sale']==pytest.approx(3.6666666667e6)
    assert b['marked_asset_sale']==pytest.approx(5.5e6)


def test_impossible_cure():
    assert not collateral_cure(100,90,0,.8,.7)['feasible']
    assert not collateral_cure(100,100,0,.8,.9)['feasible']
    assert collateral_cure(100,70,0,.8,.9)['marked_asset_sale']==0


def test_fixed_vs_floating_loan():
    t=ScenarioTree([-1,0],[1,1],[0,1]);dt=[0,1]
    fixed=TermLoan('f',100,.1,{1:20})
    floating=TermLoan('l',100,.1,{1:20},True)
    pay,terminal=fixed.compile(t,dt,[0,.5])
    assert pay[1]==30
    assert terminal[1]==80
    assert floating.compile(t,dt,[0,.5])[0][1]==80


def test_term_beyond_horizon_retained():
    t=ScenarioTree([-1,0],[1,1],[0,1])
    _,liability=TermLoan('x',100,.1,{3:100}).compile(t,[0,1])
    assert liability[1]==100


def test_excess_amortization_rejected():
    with pytest.raises(ValueError):
        TermLoan('x',10,.1,{1:11}).compile(ScenarioTree([-1],[1],[0]),[0])


def hedged(tree=None):
    t=tree or ScenarioTree([-1,0,0],[1,.5,.5],[0,1,1]);n=t.size
    return blank_portfolio(t).copy(initial_cash=100,closing_cashflow=[0,-20,20],
        hedge_names=['index'],hedge_settlement=[[0],[1],[-1]],hedge_initial_price=[0],
        hedge_half_spread=[0],hedge_gross_limit=[20],hedge_margin=np.zeros((n,1)),
        hedge_terminal_value=np.zeros((n,1)),risk_weight=1,alpha=.5)


def test_hedge_sign_and_tail_reduction():
    p=hedged();r=solve(p)
    assert r.policy['hedge_net'][0]==pytest.approx(20)
    assert r.value==pytest.approx(100)
    assert solve(p.copy(hedge_gross_limit=[0])).value==pytest.approx(80)


def test_margin_posting_not_a_loss():
    p=hedged().copy(hedge_margin=[[30],[30],[30]],hedge_gross_limit=[1],
                   hedge_settlement=[[0],[10],[10]],closing_cashflow=[0,0,0],
                   hedge_initial_price=[10],risk_weight=0)
    assert solve(p).value==pytest.approx(100)


def test_path_margin_blocks_otherwise_attractive_hedge():
    p=hedged().copy(hedge_margin=[[0],[100],[100]],hedge_gross_limit=[20])
    r=solve(p)
    assert abs(r.policy['hedge_net'][0]) < 2
    assert r.value < 100


def test_terminal_derivative_mark_counted_once():
    t=ScenarioTree([-1],[1],[0])
    p=blank_portfolio(t).copy(initial_cash=100,hedge_names=['security'],hedge_settlement=[[0]],
          hedge_initial_price=[10],hedge_half_spread=[0],hedge_gross_limit=[2],hedge_margin=[[0]],
          hedge_terminal_value=[[10]])
    assert solve(p).value==pytest.approx(100)


def test_nonanticipative_root_hedge():
    r=solve(hedged())
    assert len(r.policy['hedge_net'])==1
    assert len(r.policy['hedge_long'])==1


def test_last_delivery_interval_interest_not_omitted():
    p=financed(ScenarioTree([-1],[1],[0])).copy(initial_cash=5,draw_limit=[[0]],
          repayment_limit=[[0]],facility_rate=[[.1]],operating_year_fraction=[1])
    assert solve(p).value==pytest.approx(50)
    r=solve(p)
    assert r.policy['cash_close'][0]==pytest.approx(0)
    assert r.audit['independent_closing_cash_error']<1e-9
