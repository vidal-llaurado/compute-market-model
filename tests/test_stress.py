from dataclasses import replace
import numpy as np
import pytest
from compute_market.simulation import SimulationConfig,generate_tape,HedgePolicy,simulate
from compute_market.stress import StressScenario,apply_stress,standard_stresses,reverse_stress,optimizer_stress_suite
from compute_market.examples import synthetic_case


def test_stress_local_basis_does_not_move_financial_index():
    tape=generate_tape(SimulationConfig(paths=5,steps=5))
    shock=StressScenario('basis','test',local_multiplier=(.5,)*4)
    out=apply_stress(tape,shock)
    assert np.array_equal(out.index,tape.index)
    assert np.array_equal(out.local[:,0],tape.local[:,0])
    assert np.allclose(out.local[:,1:],tape.local[:,1:]*.5)

def test_hardware_and_lender_timing_are_distinct():
    tape=generate_tape(SimulationConfig(paths=4,steps=5,eligible_mark_lag=2))
    out=apply_stress(tape,StressScenario('assets','test',hardware_multiplier=(.5,)*4))
    assert np.allclose(out.hardware[:,1:],tape.hardware[:,1:]*.5)
    assert np.array_equal(out.eligible[:,:3],tape.eligible[:,:3])
    assert np.allclose(out.eligible[:,3:],tape.eligible[:,3:]*.5)

def test_defaults_absorbing_even_when_stress_expires():
    tape=generate_tape(SimulationConfig(paths=3,steps=5))
    out=apply_stress(tape,StressScenario('credit','test',duration=1,customer_failure=True))
    assert not out.customer_alive[:,1:].any()

def test_every_named_stress_valid_and_simulatable():
    tape=generate_tape(SimulationConfig(paths=3,steps=24))
    for s in standard_stresses():
        out=apply_stress(tape,s);r=simulate(out)
        assert np.isfinite(r.terminal_wealth).all()
        assert r.audit['all_paths_terminated']

def test_reverse_grid_is_explicit_and_no_false_global_claim():
    tape=generate_tape(SimulationConfig(paths=4,steps=3))
    r=reverse_stress(tape,HedgePolicy(),target=.1,grid_points=3,refinements=1)
    assert r['scenario_probability'] is None
    assert len(r['evaluations'])>=3

def test_optimizer_stress_retains_numerical_status():
    p,b,_=synthetic_case(depth=1)
    rows=optimizer_stress_suite(p,b)
    assert len(rows)==8 and rows[0]['status']=='optimal'
    assert rows[-1]['status'] in ('infeasible','optimal')

@pytest.mark.parametrize('kwargs',[{'eligible_haircut':1.2},{'start':-1},{'duration':0},
                                 {'benchmark_multiplier':0},{'structural_liquidation_fraction':.5}])
def test_invalid_stress(kwargs):
    with pytest.raises(ValueError):StressScenario('x','y',**kwargs)
