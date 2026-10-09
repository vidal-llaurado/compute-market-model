import numpy as np
import pytest
from scipy.optimize import linprog
from compute_market.risk import expected_shortfall,certainty_equivalent
from compute_market.examples import synthetic_case,blank_portfolio
from compute_market.tree import ScenarioTree
from compute_market.optimizer import solve
from compute_market.contracts import firm_strip
from compute_market.pricing import quote_catalogue
from compute_market.cli import main

@pytest.mark.parametrize('seed',range(10))
def test_es_matches_epigraph_linear_program(seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=9);p=rng.dirichlet(np.ones(9));alpha=.83
    objective=np.r_[1,p/(1-alpha)]
    aub=np.column_stack([-np.ones(9),-np.eye(9)])
    r=linprog(objective,A_ub=aub,b_ub=-x,bounds=[(None,None)]+[(0,None)]*9,method='highs')
    assert r.success
    assert expected_shortfall(x,p,alpha)==pytest.approx(r.fun)

@pytest.mark.parametrize('alpha',[0,.5,.9])
@pytest.mark.parametrize('weight',[0,.3,1.])
def test_solver_matches_analytic_risk(alpha,weight):
    t=ScenarioTree([-1,0,0],[1,.25,.75],[0,1,1])
    p=blank_portfolio(t).copy(initial_cash=100,initial_blocks=[1],hours=[0,1,1],
        spot_price=[[0],[5],[10]],spot_demand=[[0],[1],[1]],owned_unit_cost=[[0],[2],[2]],
        risk_weight=weight,alpha=alpha)
    assert solve(p).value==pytest.approx(certainty_equivalent([103,108],[.25,.75],alpha,weight))


def test_liquidity_shadow_price_vs_finite_difference():
    t=ScenarioTree([-1,0],[1,1],[0,1])
    p=blank_portfolio(t).copy(initial_cash=1,initial_blocks=[1],hours=[0,1],
        spot_price=[[0],[10]],spot_demand=[[0],[1]],owned_unit_cost=[[0],[2]])
    r=solve(p);bump=solve(p.copy(initial_cash=1.001))
    assert r.duals['marginal_value_initial_cash']==pytest.approx(5)
    assert (bump.value-r.value)/.001==pytest.approx(5)


def test_end_to_end_synthetic_audits():
    p,b,c=synthetic_case(depth=2);r=solve(p,b)
    assert r.status=='optimal'
    for key,value in r.audit.items():
        if 'error' in key or 'residual' in key or 'violation' in key:
            assert value < 1e-6
    assert r.audit['minimum_operating_prefund_headroom']>=-1e-6
    assert r.audit['minimum_borrowing_base_headroom']>=-1e-6


def test_joint_gpu_region_menu():
    t=ScenarioTree([-1,0],[1,1],[0,1]);p=blank_portfolio(t,2).copy(
        initial_cash=100,initial_blocks=[1,1],hours=[0,1],owned_unit_cost=[[0,0],[2,3]])
    contracts=[firm_strip(p,'a','pool-0',1,1,1),firm_strip(p,'b','pool-1',1,1,1)]
    r=quote_catalogue(p,contracts,[[5,10],[5,10]],[1,1],[2,2],.1)
    assert r['decision']=='quote';assert r['menus_evaluated']==4
    assert len(r['menu']['prices'])==2


def test_cli_roundtrip_validate_and_solve(tmp_path,capsys):
    from compute_market.cli import example
    path=tmp_path/'case.json';example(path,1)
    assert main(['validate',str(path)])==0
    assert main(['solve',str(path),'--output',str(tmp_path/'out.json')])==0
    assert (tmp_path/'out.json').exists()
    assert main(['solve',str(tmp_path/'missing.json')])==1


def test_cli_infeasible_exit_code(tmp_path):
    from compute_market.cli import example
    path=tmp_path/'case.json';case=example(path,1)
    case.portfolio=case.portfolio.copy(initial_cash=0,opening_cashflow=np.full(case.portfolio.tree.size,-1e12))
    case.save(path)
    assert main(['solve',str(path)])==2
