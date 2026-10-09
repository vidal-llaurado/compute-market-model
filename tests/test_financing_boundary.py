"""Cash-feasible financing cures and their delivery consequences."""
import json
from pathlib import Path
import numpy as np
import pytest

from compute_market.simulation import SimulationConfig, _repair_one, first_feasible_sale

BOUNDARIES = json.loads((Path(__file__).parent / 'fixtures/financing_repairs.json').read_text())


@pytest.mark.parametrize('state', BOUNDARIES['cases'], ids=lambda state: str(state['path']))
def test_margin_repair_preserves_feasible_headroom(state):
    config = SimulationConfig(**BOUNDARIES['config'])
    inputs = {key: np.asarray(value) if isinstance(value, list) else value
              for key, value in state['inputs'].items()}
    repair = _repair_one(config, **inputs)
    assert repair is not None
    assert repair['headroom'] >= 0
    financing = repair['draw'] - repair['repay'] - config.debt_fee * (repair['draw'] + repair['repay'])
    cash = inputs['cash'] + repair['proceeds'] + financing
    assert cash - repair['cost'] - config.reserve >= inputs['margin'] - 1e-8


@pytest.mark.parametrize('scale', [1e4, 1e6, 1e9])
def test_sale_root_stays_on_the_cash_feasible_side(scale):
    def headroom(sale):
        return scale * (.876 * sale - .23 * sale**2 - .1234567891)

    sale, cash = first_feasible_sale(headroom, .9)
    assert cash >= 0
    assert headroom(sale) == cash
    assert sale == pytest.approx(.1465730980682472, abs=1e-14)


def test_sale_scan_requires_an_attainable_cash_balance():
    sale, cash = first_feasible_sale(lambda sale: -5e-9, .9)
    assert sale is None
    assert cash == -5e-9


def test_repair_preserves_a_genuine_financing_failure():
    config = SimulationConfig(firm_blocks=(0.,)*4, fixed_overhead_per_step=0., reserve=0.)
    zero = np.zeros(4)
    repair = _repair_one(config, zero, -50., 0., zero, zero,
                         np.ones(4), zero, np.ones(4), zero, 0., True, 0., 0.)
    assert repair is None


def test_static_delivery_survives_binding_margin_repairs():
    from compute_market.curves.portfolio import project_case, project_economy
    from compute_market.curves.continuation import train_continuation
    from compute_market.curves.observations import fit_physical_drift
    from compute_market.curves.control import run_controller

    case = project_case(paths=1024, steps=8, seed=20260918)
    training, _ = project_economy(case, 4096, 13001)
    surface = train_continuation(case, training)
    policy_model, _ = fit_physical_drift(case.model, training.curve)
    tape, _ = project_economy(case, 1024, 13002)
    paths = [215, 302, 696]
    tape.curve.factors = tape.curve.factors[paths]
    tape.curve.integrated_rates = tape.curve.integrated_rates[paths]
    for name, value in vars(tape).items():
        if isinstance(value, np.ndarray):
            setattr(tape, name, value[paths])

    result = run_controller(case, tape, surface, policy='static', scenarios=256,
                            policy_model=policy_model)
    assert not result.failed.any()
    assert np.array_equal(result.delivery_shortfall, np.zeros(3))
    assert result.audit['maximum_cash_equation_error'] < 1e-7
    assert result.audit['maximum_prefunding_violation'] < 1e-7
    assert result.posted_margin.shape == result.cash.shape
    assert np.all(result.posted_margin >= 0)
    assert np.array_equal(result.posted_margin[:, -1], np.zeros(3))
    assert np.all(result.posted_margin.max(axis=1) <= result.maximum_margin + 1e-8)
    assert result.summary()['mean_margin_cash_years'] == pytest.approx(
        result.posted_margin[:, :-1].sum(axis=1).mean() * case.fleet.year_fraction)
