from dataclasses import replace
import numpy as np
import pytest
from compute_market.simulation import SimulationConfig
from compute_market.hedge_paths import financial_tape,FinancialTape,stress_tape,simulate_hedges
from compute_market.risk_execution import fit_kernel,ConditionalKernel,quoted_marks,solve_conditional,planned_value

@pytest.fixture(scope='module')
def lab():
    c=SimulationConfig(paths=6,steps=4,seed=218)
    train=financial_tape(replace(c,paths=30,seed=178))
    return financial_tape(c),fit_kernel(train.state,24,seed=313)

@pytest.mark.parametrize('policy',['unhedged','static','dynamic'])
def test_live_ledger(lab,policy):
    tape,kernel=lab;r=simulate_hedges(tape,kernel,policy)
    assert np.isfinite(r.terminal_wealth).all()
    assert r.audit['max_cash_conservation_error']<1e-6
    assert r.audit['max_pnl_component_error']<1e-6
    assert r.audit['all_paths_terminated']

@pytest.mark.parametrize('stress',['basis_widening','benchmark_gap','margin_jump','depth_freeze','position_limit_cut',
    'spread_widening','option_dealer_default','clearing_failure','lender_haircut','power_spike','rate_jump','FX_jump',
    'volume_outage','correlation_break','compound'])
def test_stress_accounts_for_all_paths(lab,stress):
    tape,kernel=lab;r=simulate_hedges(stress_tape(tape,stress),kernel,'dynamic')
    assert np.isfinite(r.terminal_wealth).all()
    assert r.audit['max_pnl_component_error']<1e-5
    assert r.audit['max_cash_conservation_error']<1e-5


def test_saved_tape_replay(lab,tmp_path):
    tape,kernel=lab;f=tmp_path/'tape.npz';tape.save(f)
    restored=FinancialTape.load(f)
    a=simulate_hedges(tape,kernel);b=simulate_hedges(restored,kernel)
    for key in vars(a):
        if isinstance(getattr(a,key),np.ndarray):assert np.array_equal(getattr(a,key),getattr(b,key)),key


def test_causality_future_changes_do_not_change_first_action(lab):
    tape,kernel=lab;a=simulate_hedges(tape,kernel)
    altered=tape.copy();altered.marks[:,2:,:2]*=2;altered.physical.local[:,2:]*=.5
    b=simulate_hedges(altered,kernel)
    assert np.array_equal(a.positions[:,:2],b.positions[:,:2])


def test_fit_only_uses_supplied_training_data(lab):
    tape,kernel=lab;before=kernel.to_dict()
    simulate_hedges(tape,kernel)
    assert before==kernel.to_dict()
    clone=ConditionalKernel.from_dict(before)
    assert np.array_equal(kernel.forecast(tape.state[0,0]),clone.forecast(tape.state[0,0]))


def test_zero_mean_removes_pure_forecast_alpha():
    losses=np.zeros(20);H=np.ones((20,1))*100
    r=solve_conditional(losses,H,[0],[1],[0],[1],[0],[10],[10],[-5],[5],1000,center_gains=True)
    assert r['position'][0]==pytest.approx(0)
    alpha=solve_conditional(losses,H,[0],[1],[0],[1],[0],[10],[10],[-5],[5],1000,center_gains=False)
    assert alpha['position'][0]==pytest.approx(5)


def test_conditional_replication():
    loss=np.array([-10.,0.,10.]);H=loss[:,None]
    r=solve_conditional(loss,H,[0],[100],[0],[0],[0],[3],[3],[0],[2],1000,risk_weight=1)
    assert r['position'][0]==pytest.approx(1)


def test_execution_cash_budget_limits_margin():
    loss=np.array([-10.,0.,10.]);H=loss[:,None]
    r=solve_conditional(loss,H,[0],[100],[100],[0],[0],[3],[3],[0],[2],50,risk_weight=1)
    assert r['position'][0]==pytest.approx(.5)
    assert r['cash_used']==pytest.approx(50)


def test_funded_premium_is_paid():
    loss=np.array([-10.,0.,10.]);H=loss[:,None]
    r=solve_conditional(loss,H,[0],[100],[0],[0],[0],[3],[3],[0],[2],50,risk_weight=1,funded_prices=[100])
    assert r['position'][0]==pytest.approx(.5)


def test_quote_map_does_not_use_debt_coupon(lab):
    tape,kernel=lab;x=tape.state[0,0]
    a=quoted_marks(x,4,4,tape.physical.config.year_fraction)
    c=replace(tape.physical.config,fixed_term_coupon=.20,revolver_spread=.15)
    b=quoted_marks(x,4,4,c.year_fraction)
    assert np.array_equal(a,b)


def test_put_expiry_is_intrinsic(lab):
    tape,kernel=lab;x=tape.state[:,0].copy();x[:,0]-=.5
    marks=quoted_marks(x,0,4,tape.physical.config.year_fraction)
    strike=2*np.exp(-.08*4*tape.physical.config.year_fraction)
    assert np.allclose(marks[:,5],np.maximum(strike-marks[:,0],0))


def test_rate_exposure_distinguishes_fixed_term_coupon(lab):
    tape,kernel=lab;c=tape.physical.config;x=tape.state[0,0].copy();up=x.copy();up[3]+=.01
    a=planned_value(x,c.blocks,1e6,1e6,3,c)[0];b=planned_value(up,c.blocks,1e6,1e6,3,c)[0]
    a0=planned_value(x,c.blocks,0,1e6,3,c)[0];b0=planned_value(up,c.blocks,0,1e6,3,c)[0]
    assert b<a;assert a0==pytest.approx(b0)


def test_initial_insufficient_cash_failure_retained(lab):
    tape,kernel=lab;other=tape.copy();other.physical.config=replace(tape.physical.config,initial_cash=0,max_draw_per_step=0)
    other.physical.freeze[:]=True
    r=simulate_hedges(other,kernel)
    assert r.audit['all_paths_terminated'];assert np.isfinite(r.terminal_wealth).all()


def test_validation_and_shape_errors():
    with pytest.raises(ValueError):fit_kernel(np.zeros((1,1,2)))
    with pytest.raises(ValueError):quoted_marks(np.zeros(2),1,1,.1)
    with pytest.raises(ValueError):solve_conditional([1],[[1]],[0],[1],[1],[1],[1],[1],[1],[0],[1],10)


def test_insurance_carry_is_not_erased_by_alpha_neutrality():
    loss=np.array([-10.,0.,10.]);H=(loss-100)[:,None]
    r=solve_conditional(loss,H,[0],[100],[0],[0],[0],[3],[3],[0],[2],1000,risk_weight=1,center_gains=True)
    assert r['position'][0]==pytest.approx(0,abs=1e-5)

@pytest.mark.parametrize('name',['benchmark_gap','correlation_break'])
def test_stressed_option_settles_to_same_stressed_benchmark(name):
    c=SimulationConfig(paths=8,steps=4)
    tape=stress_tape(financial_tape(c),name)
    strike=2*np.exp(-.08*c.steps*c.year_fraction)
    assert tape.marks[:,-1,5]==pytest.approx(np.maximum(strike-tape.marks[:,-1,0],0))
