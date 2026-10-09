from dataclasses import replace
import numpy as np
import pytest
from compute_market.simulation import SimulationConfig,generate_tape,simulate,HedgePolicy,sale_proceeds,first_feasible_sale
from compute_market.statistics import wilson_interval,mean_interval,tail_summary,paired_comparison,importance_event,variance_decomposition


def test_tape_reproducible_and_prefix_independent_of_horizon():
    c=SimulationConfig(paths=12,steps=3)
    a=generate_tape(c);b=generate_tape(c)
    assert np.array_equal(a.local,b.local)
    longer=generate_tape(replace(c,steps=6))
    for name in ['local','hardware','power','customer_alive','demand_blocks']:
        assert np.array_equal(getattr(a,name),getattr(longer,name)[:,:4])

def test_cash_conservation_and_causal_prefix():
    c=SimulationConfig(paths=10,steps=4)
    t=generate_tape(c);policy=HedgePolicy('adaptive',.5,.5)
    a=simulate(t,policy,10);alter=t.copy()
    alter.local[:,2:]*=5;alter.hardware[:,2:]*=.2
    b=simulate(alter,policy,10)
    for name in ['cash','blocks','revolver','compute_hedge']:
        assert np.allclose(a.traces[name][:,:2],b.traces[name][:,:2],equal_nan=True)
    assert a.audit['max_cash_conservation_error']<1e-6
    assert a.audit['all_paths_terminated']

def test_fixed_term_coupon_does_not_reset_with_reference_rate():
    c=SimulationConfig(paths=5,steps=2,initial_revolver=0,revolver_commitment=0,
                       max_draw_per_step=0,initial_cash=1e6)
    tape=generate_tape(c);a=simulate(tape)
    shocked=tape.copy();shocked.reference_rate+=.3
    b=simulate(shocked)
    assert np.allclose(a.terminal_wealth,b.terminal_wealth)

def test_surviving_terminal_exit_conserves_wealth():
    c=SimulationConfig(paths=5,steps=2,initial_cash=1e6,initial_revolver=0,initial_term_debt=0,
                       revolver_commitment=0,max_draw_per_step=0)
    r=simulate(generate_tape(c))
    assert not r.financing_failure.any()
    assert np.all(r.terminal_wealth>0)
    assert np.allclose(r.terminal_wealth,r.equity_recovery)

def test_firm_shortfall_is_reported_not_fabricated():
    c=SimulationConfig(paths=4,steps=2,initial_cash=1e6)
    tape=generate_tape(c);tape.availability[:]=0;tape.replacement[:]=0
    r=simulate(tape)
    assert r.delivery_breach.all()
    assert (r.total_shortfall_gpu_hours>0).all()
    assert r.audit['max_delivery_equation_error']<1e-8

def test_catastrophic_financing_state_stops_paths():
    c=SimulationConfig(paths=4,steps=3,initial_cash=0,initial_revolver=20e6)
    r=simulate(generate_tape(c))
    assert r.financing_failure.all()
    assert (r.failure_step==0).all()
    assert (r.creditor_loss>=0).all()
    assert np.allclose(r.equity_recovery-r.creditor_loss,r.terminal_wealth)

def test_old_hedge_position_not_reused_after_counterparty_default():
    c=SimulationConfig(paths=3,steps=3,initial_cash=1e6)
    tape=generate_tape(c);tape.counterparty_alive[:,1:]=False
    r=simulate(tape,HedgePolicy('h',.6,.5),3)
    assert np.allclose(r.traces['compute_hedge'][:,1:],0)

def test_sale_scan_handles_nonmonotonic_headroom():
    root,_=first_feasible_sale(lambda x: -(x-.4)**2+.01,.9)
    assert root==pytest.approx(.3,abs=1e-7)
    assert first_feasible_sale(lambda x:-1.,.9)[0] is None

def test_liquidation_impact_has_dimensional_consistency():
    assert sale_proceeds(1e6,.5,.9,.1)==pytest.approx(425000)

@pytest.mark.parametrize('kwargs',[{'paths':0},{'steps':0},{'advance_rate':1.1},{'eligible_mark_lag':-1},
                                    {'seed':-1},{'factor_vol_scale':-1},{'liquidation_impact':.9}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):SimulationConfig(**kwargs)

def test_zero_events_interval_not_zero_upper_bound():
    r=wilson_interval(np.zeros(100))
    assert r['probability']==0 and r['upper']>0

def test_tail_discrete_atom_and_paired_mean():
    r=tail_summary([0,0,0,10],.5,bootstrap=0)
    assert r['expected_shortfall']==5
    assert paired_comparison([1,2,3],[3,4,5])['mean']==2

def test_constant_importance_weights():
    r=importance_event([0,1,0,1],np.zeros(4))
    assert r['estimate']==.5 and r['effective_sample_size']==pytest.approx(4)

def test_importance_likelihood_normal_event():
    rng=np.random.default_rng(8);theta=1.3;z=rng.normal(theta,1,100000)
    r=importance_event(z>2,-theta*z+.5*theta**2)
    from scipy.stats import norm
    assert abs(r['estimate']-norm.sf(2))<4*r['standard_error']

def test_variance_decomposition_exposes_mc_noise():
    r=variance_decomposition([[0,1,-1],[0,1,-1]])
    assert r['parameter_variance_mc_corrected_raw']<0
    assert r['parameter_variance_nonnegative_estimate']==0


def test_financing_neutral_cash_and_matching_debt():
    c=SimulationConfig(paths=5,steps=3,initial_cash=1e6,initial_revolver=0,initial_term_debt=0,
        initial_reference_rate=0,revolver_spread=0,debt_fee=0,revolver_maturity_step=20,
        max_draw_per_step=0)
    a=simulate(generate_tape(c))
    b=simulate(generate_tape(replace(c,initial_cash=c.initial_cash+100000,initial_revolver=100000)))
    assert np.allclose(a.terminal_wealth,b.terminal_wealth)

def test_zero_gaussian_tilt_has_unit_likelihood():
    t=generate_tape(SimulationConfig(paths=10,steps=5))
    assert np.array_equal(t.log_likelihood,np.zeros(10))


def test_cure_solver_includes_nonconvex_service_cost_kinks():
    from compute_market.simulation import _repair_one
    c=SimulationConfig(paths=3,steps=3,initial_cash=100000,shortfall_penalty_per_gpu_hour=.1)
    # Replacement costs exceed contractual remedy. Headroom is not globally concave.
    answer=_repair_one(c,np.array(c.blocks),100000,1600000,np.array(c.hardware_values),
        .7*np.array(c.hardware_values),np.ones(4),np.ones(4),np.full(4,10.),np.full(4,.5),
        c.revolver_commitment,True,0.,0.)
    if answer is not None:
        assert answer['headroom']>=-1e-5
        assert np.all(answer['holdings']<=np.array(c.blocks)+1e-8)


def test_liquidation_repair_recovers_analytical_collateral_cure():
    from compute_market.simulation import _repair_one
    c=SimulationConfig(paths=2,steps=3,firm_blocks=(0.,)*4,fixed_overhead_per_step=0,
        reserve=0,debt_fee=0,advance_rate=.7,liquidation_fraction=.9,liquidation_impact=0,
        maximum_sale_fraction=1.)
    answer=_repair_one(c,np.array([1.,0,0,0]),1e6,7e6,np.array([7e6,0,0,0]),
        np.array([7e6,0,0,0]),np.ones(4),np.zeros(4),np.ones(4),np.zeros(4),
        10e6,True,0.,0.)
    assert answer is not None
    assert answer['holdings'][0]==pytest.approx(1-5.5/7)
    assert answer['proceeds']==pytest.approx(4.95e6)
    assert answer['headroom']==pytest.approx(0,abs=1e-6)


def test_debt_fee_cannot_break_prefunding_denominator():
    with pytest.raises(ValueError):SimulationConfig(debt_fee=1.)
