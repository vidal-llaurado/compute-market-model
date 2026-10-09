#!/usr/bin/env python3
"""Check saved policy statistics, cash ledgers and optional complete holdout replays."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from compute_market.curves.portfolio import CurveCase, EconomyTape
from compute_market.curves.continuation import ContinuationSurface
from compute_market.curves.model import CurveModel,CurveTape
from compute_market.curves.control import run_controller,ConditionalInnovations
from compute_market.curves.fleet_options import FleetPolicy,evaluate_fleet_policy


def tail_mean(wealth, fraction=.05):
    ordered = np.sort(wealth)
    mass = len(ordered) * fraction
    whole = int(mass)
    return float((ordered[:whole].sum() + (mass-whole)*ordered[whole]) / mass)


def check_paths(path, summary, case, innovations):
    with np.load(path, allow_pickle=False) as saved:
        audit=json.loads(str(saved['metadata']))['audit']
        assert audit['case_sha256']==case.fingerprint, path
        assert audit['conditional_innovations_sha256']==innovations.fingerprint, path
        wealth = saved['terminal_wealth']
        failed = saved['failed']
        ledger_error = float(np.max(abs(case.fleet.initial_cash + saved['components'].sum(axis=1) - wealth)))
        required = case.fleet.steps * case.fleet.hours_per_step * np.dot(case.fleet.gpus_per_block, case.fleet.firm_blocks)
        checks = {
            'mean_wealth': float(wealth.mean()),
            'loss_es': -tail_mean(wealth),
            'wealth_variance': float(wealth.var()),
            'failure_fraction': float(failed.mean()),
            'mean_delivery_shortfall_gpu_hours': float(saved['delivery_shortfall'].mean()),
            'mean_delivery_shortfall_fraction': float(saved['delivery_shortfall'].mean()/required),
            'margin_p95': float(np.quantile(saved['maximum_margin'], .95)),
            'mean_margin_cash_years': float(saved['posted_margin'][:, :-1].sum(axis=1).mean()*case.fleet.year_fraction),
            'mean_interest_cost': float(-saved['components'][:, 1].mean()),
        }
        differences = {key: abs(value-summary[key]) for key, value in checks.items()}
        assert all(np.isclose(checks[key], summary[key], rtol=1e-12, atol=1e-8) for key in checks), (path, differences)
        assert ledger_error < 1e-4, (path, ledger_error)
        assert np.all(saved['posted_margin'] >= 0), path
        assert np.all(saved['posted_margin'][:, -1] == 0), path
        assert np.all(saved['posted_margin'].max(axis=1) <= saved['maximum_margin'] + 1e-8), path
        assert all(np.isfinite(saved[key]).all() for key in saved.files if key != 'metadata'), path
        return {'paths': len(wealth), 'failures': int(failed.sum()), 'maximum_ledger_error': ledger_error,
                'summary_differences': differences}


def replay(study, label, case, protocol):
    tape = EconomyTape.load(study/'holdout_tape.npz')
    surface = ContinuationSurface.load(study/'continuation.npz')
    model = CurveModel.from_dict(json.loads((study/'policy_model.json').read_text()))
    policy = 'adaptive' if label == 'futures_only' else label
    result = run_controller(case, tape, surface, policy=policy, scenarios=protocol['conditional_atoms'],
                            alpha=protocol['alpha_conditional'], risk_weight=protocol['risk_weight'],
                            seed=protocol['conditional_seed'], carry=protocol['carry'], policy_model=model,
                            innovations=ConditionalInnovations.load(study/'conditional_innovations.npz'))
    with np.load(study/f'holdout_{label}.npz', allow_pickle=False) as saved:
        checks = {key: bool(np.array_equal(saved[key], getattr(result, key))) for key in saved.files if key != 'metadata'}
    assert all(checks.values()), (label, checks)
    return {'paths': len(result.terminal_wealth), 'all_match': True, 'fields': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'results/study')
    parser.add_argument('--output', type=Path, default=ROOT/'work/validation')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    study = args.study
    case = CurveCase.load(study/'case.json')
    innovations=ConditionalInnovations.load(study/'conditional_innovations.npz')
    summaries = json.loads((study/'path_summary.json').read_text())
    checks = {}
    for scenario, policies in summaries.items():
        for policy, summary in policies.items():
            path = study/f'holdout_{policy}.npz' if scenario == 'nominal' else study/'stresses'/f'{scenario}_{policy}.npz'
            checks[f'{scenario}/{policy}'] = check_paths(path, summary, case, innovations)
    ablation = json.loads((study/'instrument_ablation.json').read_text())
    small=CurveCase.load(study/'futures_only_case.json')
    checks['nominal/futures_only'] = check_paths(study/'holdout_futures_only.npz', ablation['futures_only'], small, innovations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'study_checks.json').write_text(json.dumps({'passed': True, 'policies': checks}, indent=2)+'\n')
    print(f'Checked {len(checks)} saved policy samples.', flush=True)
    if args.replay:
        protocol = json.loads((study/'protocol.json').read_text())
        replays = {}
        for label in ('unhedged', 'static', 'adaptive', 'futures_only'):
            selected = CurveCase.load(study/'futures_only_case.json') if label == 'futures_only' else case
            replays[label] = replay(study, label, selected, protocol)
            print(f'{label}: all saved numeric fields replay exactly.', flush=True)
        (args.output/'holdout_replay.json').write_text(json.dumps({'all_match': True, 'policies': replays}, indent=2)+'\n')
        policy=FleetPolicy.load(study/'assets/operating_policy.npz')
        tape=CurveTape.load(study/'assets/evaluation_tape.npz')
        evaluated=evaluate_fleet_policy(case.model,policy,tape=tape)
        with np.load(study/'assets/policy_outcomes.npz',allow_pickle=False) as saved:
            fields={key:bool(np.array_equal(saved[key],evaluated[result]))
                    for key,result in [('cashflows','cashflows'),('upper','upper_cashflows'),('actions','actions')]}
        assert all(fields.values()), fields
        (args.output/'asset_replay.json').write_text(json.dumps({'all_match':True,'paths':len(tape.factors),'fields':fields},indent=2)+'\n')
        print('Operating asset: all saved cash flows and actions replay exactly.',flush=True)


if __name__ == '__main__':
    main()
