import json
from types import SimpleNamespace
import numpy as np
import pytest
from compute_market.hedge_examples import dynamic_case,information_case
from compute_market.dynamic_hedging import solve_dynamic
from compute_market.hedge_pricing import price_and_hedge
from compute_market.hedge_stress import tree_stress,tree_stress_suite,reverse_margin_stress,instrument_ablation,frozen_replay
from compute_market.hedge_analytics import effectiveness,centered_tail_contributions
from compute_market.cli import main
from compute_market.risk import expected_shortfall

@pytest.mark.parametrize('mode',['dynamic','static','unhedged'])
def test_exact_quote_returns_matching_book_difference(mode):
    p,m,b,c=dynamic_case(1)
    q=price_and_hedge(p,m,c,b,price_tolerance=.2,mode=mode)
    assert q['reservation']['accepted_value']>=q['reservation']['base_value']-1e-5
    assert np.asarray(q['incremental_hedge'])==pytest.approx(np.asarray(q['accepted_allocation'])-np.asarray(q['base_allocation']))
    assert q['reservation']['bracket'][1]-q['reservation']['bracket'][0]<=.2

@pytest.mark.parametrize('payment',['opening_rate','closing_rate'])
def test_rate_quote_reoptimizes_actual_payment_schedule(payment):
    p,m,b,c=dynamic_case(1)
    q=price_and_hedge(p,m,c,b,payment=payment,price_tolerance=1e-4)
    assert q['reservation']['kind']=='USD/GPU-hour'
    assert q['reservation']['accepted_value']>=q['reservation']['base_value']-1e-5
    assert q['reservation']['amount']>0

def test_invalid_payment_and_failed_existing_book():
    p,m,b,c=dynamic_case(1)
    with pytest.raises(ValueError):price_and_hedge(p,m,c,b,payment='later_maybe')
    with pytest.raises(ValueError):price_and_hedge(p.copy(initial_cash=0,opening_cashflow=np.full(p.tree.size,-1e12)),m,c,b)

@pytest.mark.parametrize('name',['basis','margin','depth','spread','limit','collateral','curve_dislocation','OTC_default','power_rate','compound'])
def test_stress_validated_without_mutating_inputs(name):
    p,m,b,c=dynamic_case(1);old=p.to_dict();market=m.to_dict()
    ps,ms=tree_stress(p,m,name)
    assert p.to_dict()==old and m.to_dict()==market
    ms.validate(ps.tree)

def test_stress_bad_name_and_severity():
    p,m,b,c=dynamic_case(1)
    with pytest.raises(ValueError):tree_stress(p,m,'invalid')
    with pytest.raises(ValueError):tree_stress(p,m,'margin',-1)

def test_frozen_plan_replays_nominal_cash_then_fails_depth():
    p,m,b,c=dynamic_case(2);base=solve_dynamic(p,m,b)
    assert frozen_replay(p,m,b,base)['executable']
    ps,ms=tree_stress(p,m,'depth')
    assert not frozen_replay(ps,ms,b,base)['executable']

def test_ablation_and_reverse_margin_have_interpretable_bounds():
    p,m,b,c=dynamic_case(1);base=solve_dynamic(p,m,b)
    r=instrument_ablation(p,m,b,base)
    assert len(r)==len(m.names)
    assert all(x['marginal_inclusion_value']>=-1e-3 for x in r if x['marginal_inclusion_value'] is not None)
    rev=reverse_margin_stress(p,m,b,base,maximum=100,tolerance=1)
    assert rev['crossing_found'] or rev['feasible_through']==100
    if rev['crossing_found']:assert 0<rev['first_infeasible_multiplier']-rev['last_feasible_multiplier']<=1

def test_tree_stress_exposes_separate_recourse_modes():
    p,m,b,c=dynamic_case(1);rows=tree_stress_suite(p,m,b)
    assert len(rows)==10
    assert all(set(('reoptimized','root_locked','financial_policy_locked','frozen_controls'))<=set(row) for row in rows)

def _result(wealth):
    w=np.asarray(wealth,float)
    return SimpleNamespace(policy='hedged',terminal_wealth=w,maximum_funding_need=np.zeros(len(w)),maximum_margin=np.zeros(len(w)),financing_failure=np.zeros(len(w),bool))

def test_signed_es_comparison_does_not_use_invalid_loss_ratio():
    a=_result([10,20,30,40]);b=_result([20,30,40,50]);r=effectiveness(a,b,bootstrap=20)
    assert r['ES_reduction_dollars']==pytest.approx(10)
    assert r['paired_wealth_change']['mean']==pytest.approx(10)

def test_zero_variance_effectiveness_is_undefined_not_division_by_zero():
    a=_result([10,10,10]);b=_result([11,11,11])
    assert effectiveness(a,b,bootstrap=0)['variance_reduction_fraction'] is None

@pytest.mark.parametrize('alpha',[0,.5,.95])
def test_fractional_tail_attribution_reconciles(alpha):
    x=np.array([[1.,3],[2,4],[9,-8],[5,7]])
    r=centered_tail_contributions(x,alpha)
    assert r['sum_error']<1e-10
    expected=expected_shortfall(-x.sum(axis=1)+x.sum(axis=1).mean(),np.full(4,.25),alpha)
    assert sum(r['contributions'])==pytest.approx(expected)

def test_analytics_validation():
    with pytest.raises(ValueError):centered_tail_contributions([1,2])
    with pytest.raises(ValueError):effectiveness(_result([1,2]),_result([1,2,3]))

@pytest.mark.parametrize('command',['solve','validate','quote'])
def test_cli_complete_case(tmp_path,command):
    from compute_market.case import ValuationCase
    p,m,b,c=dynamic_case(1)
    case=ValuationCase('test',p,m,b,c);case.save(tmp_path/'case.json')
    args=[command,str(tmp_path/'case.json'),'--output',str(tmp_path/'out.json')]
    if command=='quote':args+=['--tolerance','1']
    assert main(args)==0
    result=json.loads((tmp_path/'out.json').read_text())
    if command=='solve':
        assert 'position' in result['policy']
        assert main(['verify',str(tmp_path/'case.json'),str(tmp_path/'out.json')])==0
    if command=='quote':assert 'incremental_hedge' in result

def test_small_end_to_end_pipeline_and_saved_replay(tmp_path):
    from compute_market.research import run_study
    out=run_study(tmp_path,quick=True)
    assert all(out['serial_prefix_replay']['checks'].values())
    assert len(out['stresses'])==15
    assert (tmp_path/'valuation/quote.json').is_file()
    assert main(['replay',str(tmp_path),'--policy','joint','--paths','3'])==0
    assert out['valuation']['dynamic']['value']>=out['valuation']['static']['value']-1e-3
