"""Compute Market research protocol: valuation, joint control and stress evidence."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json,platform,time
import numpy as np
import scipy
from .simulation import SimulationConfig
from .hedge_paths import financial_tape,stress_tape,FinancialTape,simulate_hedges
from .risk_execution import fit_kernel,ConditionalKernel
from .execution import ExecutionMandate
from .parallel import simulate_parallel,subset_tape
from .hedge_analytics import summarize_paths,effectiveness,centered_tail_contributions
from .hedge_examples import dynamic_case,information_case
from .case import ValuationCase
from .scenarios import sampled_case
from .tails import tree_tail_diagnostics,tail_diagnostics
from .dynamic_hedging import solve_dynamic
from .hedge_stress import tree_stress_suite,instrument_ablation

STRESSES=('basis_widening','benchmark_gap','margin_jump','depth_freeze','position_limit_cut',
    'spread_widening','option_dealer_default','clearing_failure','lender_haircut','power_spike',
    'rate_jump','FX_jump','volume_outage','correlation_break','compound')
POLICIES=('joint_unhedged','joint_static','joint')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=cast),encoding='utf-8')


def resolution_study(output,branches=(4,8,16,32,64),seeds=(1201,1202,1203)):
    rows=[]
    for b in branches:
        for seed in seeds:
            case=sampled_case(b,2,seed);result=case.solve(tie_break=False)
            quote=case.quote(price_tolerance=5.,tie_break=False)
            diagnostics=tree_tail_diagnostics(case.portfolio.tree,result.policy['risk_loss'],case.portfolio.alpha)
            row={'branches':b,'delivery_intervals':2,'hours_per_interval':720,'seed':seed,
                 'case_sha256':case.fingerprint,'nodes':case.portfolio.tree.size,'leaves':len(case.portfolio.tree.leaves),
                 'value':result.value,'upfront_reservation':quote['reservation']['amount'],
                 'reservation_bracket':quote['reservation']['bracket'],
                 'root_position':result.policy['position'][0],
                 'saturated_fraction':diagnostics['saturated_fraction'],
                 'conditional_tail_ess':diagnostics['mean_conditional_tail_ess'],
                 'minimum_conditional_tail_ess':diagnostics['minimum_conditional_tail_ess'],
                 'certificate':result.provenance['primary_certificate']}
            rows.append(row);write(output/'resolution.json',rows)
            if seed==seeds[0]:case.save(output/f'case_support_{b}.json')
    return rows


def run_study(output,paths=2048,train_paths=4000,stress_paths=1024,steps=12,scenarios=256,
              seed=20260918,bootstrap=200,workers=4,quick=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def log(s):print(s,flush=True)
    if quick:paths,train_paths,stress_paths,steps,scenarios,bootstrap,workers=16,64,12,4,32,30,1
    protocol={'project':'Compute Market','data':'synthetic','paths':paths,'training_paths':train_paths,
        'stress_paths':stress_paths,'delivery_intervals':steps,'hours_per_interval':720,
        'conditional_atoms':scenarios,'conditional_alpha':.90,'risk_weight':.65,'reporting_alpha':.95,
        'bootstrap_replicates':bootstrap,'workers':workers,'seeds':{'training':seed+1000,
        'holdout':seed+2000,'stress':seed+3000,'kernel':seed+4000,'diagnostics':seed+6000},
        'design':'Fit once on training paths. Apply frozen policies on matched independent paths. Named stresses are conditional experiments.'}
    mandate=ExecutionMandate();write(output/'protocol.json',protocol);write(output/'execution_mandate.json',mandate.to_dict())
    config=SimulationConfig(paths=train_paths,steps=steps,seed=seed+1000,term_maturity_step=steps,revolver_maturity_step=max(2,steps-3))
    train=financial_tape(config);kernel=fit_kernel(train.state,scenarios,seed+4000)
    write(output/'kernel.json',kernel.to_dict());write(output/'simulation_config.json',config.to_dict())
    np.savez_compressed(output/'training_states.npz',states=train.state)
    tape=financial_tape(replace(config,paths=paths,seed=seed+2000));tape.save(output/'holdout_tape.npz')
    results={};summaries={}
    for policy in POLICIES+('adaptive',):
        log('holdout '+policy)
        r=simulate_parallel(tape,kernel,policy,workers=workers,mandate=mandate);results[policy]=r
        r.save(output/f'holdout_{policy}.npz');summaries[policy]=summarize_paths(r,bootstrap,seed+5100)
        summaries[policy]['cash_tail_contributions']=centered_tail_contributions(r.pnl_components)
        summaries[policy]['tail_resolution']=tail_diagnostics(-r.terminal_wealth,alpha=.95,sample_kind='iid_paths')
        write(output/'holdout_summary.json',summaries)
    comparisons={p:effectiveness(results['joint_unhedged'],results[p],bootstrap,seed+5200) for p in POLICIES[1:]}
    comparisons['joint_vs_adaptive']=effectiveness(results['adaptive'],results['joint'],bootstrap,seed+5200)
    write(output/'comparisons.json',comparisons)
    stress_base=financial_tape(replace(config,paths=stress_paths,seed=seed+3000));stress_base.save(output/'stress_tape.npz')
    stresses=[]
    for name in STRESSES:
        log('stress '+name);st=stress_tape(stress_base,name);row={'stress':name,'policies':{}};rr={}
        for policy in POLICIES:
            r=simulate_parallel(st,kernel,policy,workers=workers,mandate=mandate);rr[policy]=r
            r.save(output/'stresses'/f'{name}_{policy}.npz')
            row['policies'][policy]=summarize_paths(r,max(30,bootstrap//2),seed+5300)
        row['joint_effectiveness']=effectiveness(rr['joint_unhedged'],rr['joint'],max(30,bootstrap//2),seed+5400)
        stresses.append(row);write(output/'stress_summary.json',stresses)
    log('complete valuation case')
    p,m,b,c=dynamic_case(3);case=ValuationCase('contract and treasury book',p,m,b,c,{'data':'synthetic','role':'worked complete-contract example'})
    exact=output/'valuation';exact.mkdir(exist_ok=True);case.save(exact/'case.json')
    solved={};exact_summary={}
    for mode in ('unhedged','static','dynamic'):
        r=case.solve(mode);solved[mode]=r;write(exact/f'{mode}.json',r.to_dict())
        exact_summary[mode]={'value':r.value,'certificate':r.provenance['primary_certificate'],
                            'compiled_model':r.provenance['compiled_model']}
    quote=case.quote(price_tolerance=.05);write(exact/'quote.json',quote)
    exact_summary['upfront_reservation']=quote['reservation']['amount']
    exact_summary['tails']=tree_tail_diagnostics(p.tree,solved['dynamic'].policy['risk_loss'],p.alpha)
    write(exact/'summary.json',exact_summary)
    write(exact/'stress.json',tree_stress_suite(p,m,b,solved['dynamic']))
    write(exact/'instrument_ablation.json',instrument_ablation(p,m,b,solved['dynamic']))
    log('fixed-horizon scenario refinement')
    resolution=resolution_study(output/'support',branches=(4,8) if quick else (4,8,16,32,64),seeds=(1201,) if quick else (1201,1202,1203))
    log('conditional kernel and funding sensitivity')
    dtape=financial_tape(replace(config,paths=8 if quick else 128,seed=seed+6000))
    sensitivities=[]
    for atoms in ((32,64) if quick else (64,256,1024)):
        for risk_alpha in ((.9,) if quick else (.9,.95)):
            k=fit_kernel(train.state,atoms,seed+4000)
            r=simulate_parallel(dtape,k,'joint',workers=workers,alpha=risk_alpha,mandate=mandate)
            sensitivities.append({'kind':'conditional_support','atoms':atoms,'alpha':risk_alpha,
                'nominal_conditional_tail_atoms':atoms*(1-risk_alpha),
                'summary':summarize_paths(r,max(30,bootstrap//2),seed+5700)})
    for rate in ((0.,.05) if quick else (0.,.025,.10)):
        active=replace(mandate,margin_opportunity_annual={name:rate for name in mandate.directions})
        r=simulate_parallel(dtape,kernel,'joint',workers=workers,mandate=active)
        sensitivities.append({'kind':'margin_opportunity_annual','rate':rate,'summary':summarize_paths(r,max(30,bootstrap//2),seed+5700)})
    for carry in ('centered','symmetric','total_return'):
        r=simulate_parallel(dtape,kernel,'joint',workers=workers,mandate=replace(mandate,carry_convention=carry))
        sensitivities.append({'kind':'carry_mandate','carry':carry,'summary':summarize_paths(r,max(30,bootstrap//2),seed+5700)})
    write(output/'sensitivities.json',sensitivities)
    # Replay a declared prefix through the same public path kernel without multiprocessing.
    replay_n=min(32,paths);restored=FinancialTape.load(output/'holdout_tape.npz')
    replay=simulate_hedges(subset_tape(restored,np.arange(replay_n)),ConditionalKernel.from_dict(kernel.to_dict()),'joint',mandate=mandate)
    checks={key:bool(np.array_equal(value[:replay_n],getattr(replay,key))) for key,value in vars(results['joint']).items() if isinstance(value,np.ndarray)}
    summary={'protocol':protocol,'holdout':summaries,'comparisons':comparisons,'stresses':stresses,
        'valuation':exact_summary,'resolution':resolution,'sensitivities':sensitivities,
        'serial_prefix_replay':{'paths':replay_n,'checks':checks},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'elapsed_seconds':time.perf_counter()-start}
    write(output/'summary.json',summary);log('study complete')
    return summary
