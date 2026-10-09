from dataclasses import replace
import json
import numpy as np
import pytest
from compute_market.case import ValuationCase
from compute_market.hedge_examples import dynamic_case
from compute_market.dynamic_hedging import compile_dynamic,solve_dynamic,compiled_fingerprint
from compute_market.schema import Contract
from compute_market.tails import tail_weights,tail_diagnostics,tree_tail_diagnostics
from compute_market.execution import ExecutionMandate,dispatch_controls
from compute_market.risk_execution import solve_conditional,LinearControls,fit_kernel
from compute_market.scenarios import sampled_case
from compute_market.simulation import SimulationConfig
from compute_market.hedge_paths import financial_tape,simulate_hedges,FinancialTape,stress_tape
from compute_market.parallel import subset_tape,simulate_parallel

@pytest.fixture
def case():
    p,m,b,c=dynamic_case(1)
    return ValuationCase('test portfolio',p,m,b,c,{'data':'synthetic'})

def test_complete_case_roundtrip_and_book_required(case,tmp_path):
    path=tmp_path/'case.json';case.save(path)
    other=ValuationCase.load(path);assert case.fingerprint==other.fingerprint
    data=case.to_dict();del data['book']
    with pytest.raises(ValueError,match='existing book'):ValuationCase.from_dict(data)
    data=case.to_dict();data['book']=None
    with pytest.raises(ValueError):ValuationCase.from_dict(data)

def test_case_hash_changes_with_book_and_mandate(case):
    original=case.fingerprint
    case.book.opening_receipts[1]+=1
    assert original!=case.fingerprint

def test_policy_fingerprint_rejects_empty_book(case):
    result=case.solve();audit=case.verify_result(result)
    errors=[float(v) for k,v in audit.items() if 'error' in k]
    assert errors and max(errors)<1e-5
    case.book=Contract.empty(case.portfolio)
    with pytest.raises(ValueError,match='complete valuation case'):case.verify_result(result)

def test_unhedged_is_true_lp_and_certificate_is_retained(case):
    result=case.solve('unhedged')
    assert result.provenance['compiled_model']['class']=='LP'
    assert result.provenance['compiled_model']['integer_variables']==0
    assert result.provenance['primary_certificate']['success']
    assert 'secondary_certificate' in result.provenance
    assert result.provenance['primary_tie_break_slack']<=1e-3

def test_compiled_matrix_hash_changes_with_cash(case):
    p,m,b=case.portfolio,case.market,case.book
    a,_,o,_,_=compile_dynamic(p,m,b);h=compiled_fingerprint(a,o)
    a2,_,o2,_,_=compile_dynamic(p,m,b,premium=1.)
    assert h!=compiled_fingerprint(a2,o2)

def test_deep_execution_tier_removes_impact_binaries(case):
    case.market.impact_depth[:]=case.market.trade_limit
    lp,_,_,_,_=compile_dynamic(case.portfolio,case.market,case.book)
    assert not any(name.startswith('impact') and name.endswith('_sign') for name in lp.names)

def test_fixed_horizon_support_really_changes_tail_resolution():
    for b in (4,16):
        case=sampled_case(b,2,1201);r=case.solve(tie_break=False)
        assert max(case.portfolio.tree.stage)==2
        assert np.unique(case.portfolio.hours).tolist()==[0.,720.]
        d=tree_tail_diagnostics(case.portfolio.tree,r.policy['risk_loss'],.9)
        assert d['saturated_fraction']==pytest.approx(1. if b==4 else 0.)

@pytest.mark.parametrize('alpha',[.75,.9,.95,.99])
def test_three_branch_expected_shortfall_collapses(alpha):
    d=tail_diagnostics([1.,5.,20.],[.25,.5,.25],alpha)
    assert d['expected_shortfall']==20.;assert d['saturated_at_worst']

def test_fractional_tail_weights_and_tie_invariance():
    q=tail_weights([0.,1.,2.],[.5,.25,.25],.6)
    assert q==pytest.approx([0.,.375,.625]);assert q.sum()==pytest.approx(1.)
    assert tail_weights([1.,2.,2.],[.5,.25,.25],.75)==pytest.approx([0.,.5,.5])

@pytest.mark.parametrize('x,p,a',[([],[],.9),([1.],[.8],.9),([1.],[1.],1.),([np.nan],[1.],.9),([1.,2.],[-1.,2.],.9)])
def test_tail_input_validation(x,p,a):
    with pytest.raises(ValueError):tail_weights(x,p,a)


def helper(gains,**kwargs):
    n=len(gains)
    return solve_conditional([-10.,10.],gains,[0.],[1.],[10.],[0.],[0.],[10.],[10.],[0.],[2.],10.,
                             risk_weight=1.,funding_rate=0.,**kwargs)

def test_joint_dispatch_can_replace_a_hedge_when_cash_is_scarce():
    H=np.array([[-10.],[10.]])
    assert helper(H)['position'][0]==pytest.approx(1.)
    controls=LinearControls(['dispatch'],np.zeros(1),np.ones(1),np.array([10.]),np.full((2,1),-20.))
    result=helper(H,controls=controls)
    assert result['position'][0]==pytest.approx(0.)
    assert result['controls']['dispatch']==pytest.approx(1.)
    assert result['conditional_objective']==pytest.approx(-10.)

def test_joint_dispatch_does_not_displace_a_more_valuable_hedge():
    controls=LinearControls(['dispatch'],np.zeros(1),np.ones(1),np.array([10.]),np.full((2,1),-5.))
    result=helper(np.array([[-10.],[10.]]),controls=controls)
    assert result['position'][0]==pytest.approx(1.)
    assert result['controls']['dispatch']==pytest.approx(0.)

@pytest.mark.parametrize('convention',['adverse_only','centered','symmetric','total_return'])
def test_carry_conventions_replay(convention):
    result=helper(np.array([[-8.],[12.]]),carry_convention=convention)
    assert np.isfinite(result['conditional_objective'])
    assert abs(result['conditional_objective']-result['solver_objective'])<1e-5

def test_centered_mean_shift_does_not_create_alpha():
    a=helper(np.array([[-10.],[10.]]),carry_convention='centered')
    b=helper(np.array([[90.],[110.]]),carry_convention='centered')
    assert a['position']==pytest.approx(b['position'])

def test_signed_and_factor_mandates_are_explicit():
    m=ExecutionMandate();lo,hi=m.bounds(np.ones(6))
    assert hi[:2].tolist()==[0.,0.];assert lo[2:].tolist()==[0.,0.,0.,0.]
    both=replace(m,directions={k:'both' for k in m.directions});assert np.all(both.bounds(np.ones(6))[0]==-1)
    result=helper(np.array([[-10.],[10.]]),factor_loadings=[[2.]],exposure_lower=[0.],exposure_upper=[.5])
    assert result['position'][0]==pytest.approx(.25)

def test_annual_liquidity_rates_use_interval_duration():
    m=ExecutionMandate();m=replace(m,margin_opportunity_annual={k:.12 for k in m.directions})
    assert m.local_arguments(1/12)['funding_rate']==pytest.approx(np.full(6,.01))

@pytest.mark.parametrize('rate',[-1.,np.nan,np.inf])
def test_bad_liquidity_rate_rejected(rate):
    with pytest.raises(ValueError):
        solve_conditional([0.,1.],[[0.],[1.]],[0.],[1.],[1.],[0.],[0.],[1.],[1.],[0.],[1.],10.,funding_rate=rate)

def test_depth_is_not_position_limit_or_turnover_cap():
    tape=financial_tape(SimulationConfig(paths=2,steps=2))
    depth=tape.impact_depth.copy();tape.position_limit*=.1;tape.trade_limit*=.2
    tape.validate();assert np.array_equal(depth,tape.impact_depth)

def test_put_is_funded_without_initial_margin():
    tape=financial_tape(SimulationConfig(paths=2,steps=2))
    assert np.all(tape.im[:,:,5]==0.)
    assert np.any(tape.marks[:,:,5]>0.)

def test_joint_current_controls_and_timing_on_shared_paths():
    c=SimulationConfig(paths=4,steps=3,seed=471)
    train=financial_tape(replace(c,paths=30,seed=953))
    kernel=fit_kernel(train.state,32);tape=financial_tape(c)
    for policy in ('joint_unhedged','joint_static','joint'):
        r=simulate_hedges(tape,kernel,policy)
        assert r.audit['joint_optimizations']>0
        assert r.audit['max_cash_conservation_error']<1e-6
        assert r.audit['max_pnl_component_error']<1e-6
        if policy=='joint_unhedged':assert np.all(r.positions==0.)
    changed=tape.copy();changed.marks[:,2:,:2]*=2;changed.physical.local[:,2:]*=.5
    a=simulate_hedges(tape,kernel,'joint');b=simulate_hedges(changed,kernel,'joint')
    assert np.array_equal(a.positions[:,:2],b.positions[:,:2])

def test_parallel_replay_matches_serial():
    c=SimulationConfig(paths=4,steps=2,seed=128);t=financial_tape(c)
    k=fit_kernel(financial_tape(replace(c,paths=20,seed=185)).state,16)
    a=simulate_parallel(t,k,'joint',workers=1);b=simulate_parallel(t,k,'joint',workers=2)
    for name,val in vars(a).items():
        if isinstance(val,np.ndarray):assert np.array_equal(val,getattr(b,name)),name


def test_lp_certificate_has_a_real_dual_not_zero_mip_placeholder(case):
    c=case.solve('unhedged').provenance['primary_certificate']
    assert c['mip_dual_bound'] is None and c['mip_relative_gap'] is None
    assert abs(c['lp_primal_dual_gap'])<1e-6
    assert c['lp_stationarity_error']<1e-6
    assert c['lp_dual_objective']==pytest.approx(c['objective'])


def test_case_replay_respects_explicit_cash_premium(case):
    result=case.solve(premium=5000.)
    audit=case.verify_result(result)
    assert max(float(v) for k,v in audit.items() if 'error' in k)<1e-5


def test_case_verify_rejects_infeasible_result(case):
    with pytest.raises(ValueError,match='optimal'):
        case.verify_result({'status':'infeasible','provenance':{'case_sha256':case.fingerprint}})


def test_cli_rejects_an_unbundled_portfolio(tmp_path,case):
    from compute_market.cli import main
    path=tmp_path/'portfolio.json';case.portfolio.save(path)
    assert main(['solve',str(path)])==1

@pytest.mark.parametrize('payment',['upfront','opening_rate','closing_rate'])
def test_direct_lp_reservation_and_certificate(payment):
    from compute_market.scenarios import sampled_case
    c=sampled_case(4,2,77)
    q=c.quote(payment=payment,price_tolerance=.02,tie_break=False)
    assert q['pricing_method'].startswith('direct LP')
    assert q['pricing_certificate']['success']
    assert abs(q['pricing_certificate']['lp_primal_dual_gap'])<1e-5
    lo,hi=q['reservation']['bracket']
    assert 0<=hi-lo<.02
    assert q['accepted']['value']>=q['base']['value']-1e-5
    assert np.max(np.abs(np.array(q['accepted_allocation'])-np.array(q['base_allocation'])-q['incremental_hedge']))<1e-9


def test_direct_lp_allows_negative_reservation():
    from compute_market.scenarios import sampled_case
    from compute_market.schema import Contract
    c=sampled_case(4,1,81)
    offered=Contract.empty(c.portfolio)
    offered.closing_receipts[c.portfolio.tree.leaves]=1000.
    c.offered=offered
    q=c.quote(price_tolerance=.02,tie_break=False)
    assert q['reservation']['amount']<0


def test_mixed_integer_quote_retains_bracket_search():
    from compute_market.hedge_examples import dynamic_case
    from compute_market.case import ValuationCase
    p,m,b,c=dynamic_case(1)
    q=ValuationCase('settlement',p,m,b,c).quote(price_tolerance=.1)
    assert q['pricing_method'].startswith('bracketed')
