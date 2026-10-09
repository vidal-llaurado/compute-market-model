"""Synthetic holdout, paired-policy, stress and sensitivity experiments.

Save inputs, path summaries, uncertainty intervals and accounting checks."""
from __future__ import annotations
from dataclasses import replace,asdict
from pathlib import Path
import json,platform,time,sys
import numpy as np
import scipy
from .simulation import SimulationConfig,HedgePolicy,generate_tape,simulate
from .statistics import (simulation_summary,paired_comparison,variance_decomposition,
                         mean_interval,wilson_interval,tail_summary,importance_event)
from .stress import apply_stress,standard_stresses,reverse_stress,optimizer_stress_suite,severity_scenario
from .examples import synthetic_case
from .optimizer import solve
from .pricing import reservation_price
from .workloads import Cluster,Workload,Outage,simulate_workloads


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def _path_payload(results):
    payload={}
    for name,r in results.items():
        for key in ('terminal_wealth','financing_failure','delivery_breach','failure_step',
                    'minimum_cash_headroom','maximum_borrowing_gap','total_asset_sales',
                    'total_shortfall_gpu_hours','hedge_cash_pnl','creditor_loss','equity_recovery'):
            payload[f'{name}__{key}']=getattr(r,key)
        for key,val in r.traces.items():payload[f'{name}__trace__{key}']=val
    return payload


def _note(message,verbose):
    if verbose:print(message,flush=True)


def run_suite(output,paths=4000,stress_paths=1000,train_paths=1000,steps=24,seed=20260918,
              bootstrap=250,quick=False,verbose=False):
    start=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if min(paths,stress_paths,train_paths)<2 or steps<3:raise ValueError('at least two paths per set and three steps required')
    config=SimulationConfig(paths=paths,steps=steps,seed=seed)
    _note('Generating independent training and held-out tapes.',verbose)
    trainconfig=replace(config,paths=train_paths,seed=seed+101)
    train=generate_tape(trainconfig)
    candidates=[HedgePolicy('candidate-unhedged',0,0),HedgePolicy('candidate-25',.25,.5),
                HedgePolicy('candidate-50',.5,.7),HedgePolicy('candidate-75',.75,.7)]
    training=[]
    for p in candidates:
        r=simulate(train,p,0)
        score=.7*float(r.terminal_wealth.mean())-.3*tail_summary(-r.terminal_wealth,.95,0)['expected_shortfall']
        training.append({'policy':asdict(p),'score':score,'failure_probability':float(r.financing_failure.mean())})
    index=int(np.argmax([x['score'] for x in training]));selected=replace(candidates[index],name='train_selected')
    write_json(output/'training.json',{'config':trainconfig.to_dict(),'candidates':training,'selected':asdict(selected),
        'criterion':'.7 E[terminal wealth] -.3 ES95(-terminal wealth); terminal comparison, not nested optimizer objective',
        'selection_uses_test_data':False,'scope':'finite causal hedge-rule selection; no global policy-optimality claim'})
    tape=generate_tape(config)
    policies=[HedgePolicy('unhedged'),HedgePolicy('static_50',.5,.7,False),
              HedgePolicy('adaptive_50',.5,.7,True),selected]
    results={};summaries={}
    for k,p in enumerate(policies):
        _note(f'Held-out evaluation: {p.name}',verbose)
        r=simulate(tape,p);results[p.name]=r;summaries[p.name]=simulation_summary(r,bootstrap,seed+200+k)
    write_json(output/'holdout.json',{'config':config.to_dict(),'summaries':summaries,
        'paired_wealth_comparisons':{name:paired_comparison(results['unhedged'].terminal_wealth,r.terminal_wealth)
                                    for name,r in results.items() if name!='unhedged'}})
    np.savez_compressed(output/'holdout_paths.npz',**_path_payload(results))
    # Tape retained for exact policy replay, with no hidden network calls.
    np.savez_compressed(output/'holdout_tape.npz',**{k:v for k,v in vars(tape).items() if isinstance(v,np.ndarray)})
    write_json(output/'tape_metadata.json',tape.metadata)
    _note('Running named joint stress scenarios on a shared independent tape.',verbose)
    stressconfig=replace(config,paths=stress_paths,seed=seed+202)
    stress_tape=generate_tape(stressconfig);stresses=[];stress_payload={}
    for k,s in enumerate(standard_stresses(steps)):
        transformed=apply_stress(stress_tape,s);row={'scenario':s.to_dict(),'policies':{}}
        for p in (policies[0],selected):
            r=simulate(transformed,p,2)
            row['policies'][p.name]=simulation_summary(r,min(bootstrap,100),seed+300+k)
            stress_payload.update(_path_payload({s.name+'__'+p.name:r}))
        stresses.append(row)
    write_json(output/'stresses.json',{'config':stressconfig.to_dict(),'scenarios':stresses,
        'warning':'conditional on each deterministic stress; no joint scenario likelihood assigned'})
    np.savez_compressed(output/'stress_paths.npz',**stress_payload)
    _note('Evaluating finite-tree ambiguity and frozen-initial-hedge cases.',verbose)
    p,b,offered=synthetic_case(depth=2 if quick else 3)
    optimrows=optimizer_stress_suite(p,b)
    # Premium comparison changes the objective, not represented stress support.
    quotes=[]
    for radius in [0.,.05,.15]:
        model=p.copy(ambiguity_radius=np.full(p.tree.size,radius))
        quote=reservation_price(model,offered,b,price_tolerance=.05)
        quotes.append({'TV_radius':radius,'quote':quote.to_dict()})
    write_json(output/'optimizer_stresses.json',{'cases':optimrows,'upfront_quotes':quotes})
    p.save(output/'optimizer_portfolio.json');write_json(output/'optimizer_book.json',b.to_dict());write_json(output/'optimizer_offer.json',offered.to_dict())
    _note('Searching a prespecified reverse-stress ray and capital/liquidity surface.',verbose)
    small=replace(config,paths=min(stress_paths,256),seed=seed+303)
    smalltape=generate_tape(small)
    reverse=reverse_stress(smalltape,selected,target=.1,grid_points=5 if quick else 11,refinements=2 if quick else 5)
    write_json(output/'reverse_stress.json',reverse)
    surface=[]
    for haircut in ([0,.3] if quick else [0.,.15,.3,.45]):
        for cash_mult in ([.5,1.5] if quick else [.5,1.,1.5,2.]):
            tc=replace(small,initial_cash=small.initial_cash*cash_mult)
            st=apply_stress(generate_tape(tc),severity_scenario(.55))
            st.eligible*=1-haircut
            r=simulate(st,selected,0)
            surface.append({'additional_lender_haircut':haircut,'initial_cash_multiplier':cash_mult,
                            'failure':wilson_interval(r.financing_failure),'mean_wealth':float(r.terminal_wealth.mean())})
    write_json(output/'liquidity_surface.json',{'rows':surface,'conditional_stress':severity_scenario(.55).to_dict()})
    _note('Testing parameter scenarios, time-grid sensitivity, and Monte Carlo precision.',verbose)
    params=[{'asset_decline_annual':.08,'customer_default_annual_hazard':.01,'factor_vol_scale':.8},
            {'asset_decline_annual':.14,'customer_default_annual_hazard':.03,'factor_vol_scale':1.},
            {'asset_decline_annual':.22,'customer_default_annual_hazard':.06,'factor_vol_scale':1.2},
            {'asset_decline_annual':.30,'customer_default_annual_hazard':.10,'factor_vol_scale':1.4}]
    matrix=[];paramrows=[]
    for k,param in enumerate(params):
        # Independent inner streams required by finite-inner-MC variance correction.
        pc=replace(small,seed=seed+400+k,**param)
        r=simulate(generate_tape(pc),selected,0);matrix.append(r.terminal_wealth)
        paramrows.append({'parameters':param,'wealth':mean_interval(r.terminal_wealth),
                          'failure':wilson_interval(r.financing_failure)})
    write_json(output/'parameter_sensitivity.json',{'scenarios':paramrows,
        'variance_decomposition':variance_decomposition(matrix),
        'warning':'equally weighted sensitivity scenarios, not an estimated posterior'})
    np.savez_compressed(output/'parameter_paths.npz',terminal_wealth=np.array(matrix))
    gridrows=[]
    # Fixed horizon; default intensities and financing accruals scale with dt. Mark
    # lag, timing, checkpoint and control frequency are distinct economic changes.
    for factor in ([1,2] if quick else [1,2,4]):
        gc=replace(small,steps=steps*factor,hours_per_step=small.hours_per_step/factor,
                   term_amortization=small.term_amortization/factor,
                   max_draw_per_step=small.max_draw_per_step/factor,
                   term_maturity_step=small.term_maturity_step*factor,
                   revolver_maturity_step=small.revolver_maturity_step*factor,
                   eligible_mark_lag=small.eligible_mark_lag*factor,
                   fixed_overhead_per_step=small.fixed_overhead_per_step/factor,
                   seed=seed+500+factor)
        gr=simulate(generate_tape(gc),selected,0)
        gridrows.append({'steps':gc.steps,'hours_per_step':gc.hours_per_step,
                         'wealth':mean_interval(gr.terminal_wealth),'failure':wilson_interval(gr.financing_failure)})
    convergence=[]
    for size in sorted(set(min(paths,x) for x in [250,500,1000,2000,4000,8000])):
        r=results['train_selected']
        convergence.append({'paths':size,'wealth':mean_interval(r.terminal_wealth[:size]),
                            'failure':wilson_interval(r.financing_failure[:size]),
                            'tail':tail_summary(-r.terminal_wealth[:size],.95,min(bootstrap,100),seed+size)})
    write_json(output/'precision.json',{'prefix_sample_diagnostics':convergence,'time_grid_sensitivity':gridrows,
        'grid_warning':'decision/settlement frequency and pooled-demand aggregation also change; not a proof of continuous-time convergence'})
    # Importance sampling estimates the same event under the same conditional
    # stress map; only Gaussian proposal law is tilted. Baseline/Q outcomes use
    # independently seeded tapes; parameter and stress probabilities are not tilted.
    itc=replace(small,paths=max(1000,stress_paths),seed=seed+601)
    it=generate_tape(itc,gaussian_tilt=-.18)
    it=apply_stress(it,severity_scenario(.45))
    ir=simulate(it,selected,0)
    plain=apply_stress(generate_tape(replace(itc,seed=seed+602)),severity_scenario(.45))
    pr=simulate(plain,selected,0)
    write_json(output/'importance_sampling.json',{
        'tilted':importance_event(ir.financing_failure,it.log_likelihood),
        'ordinary':wilson_interval(pr.financing_failure),
        'tilt':-.18,'conditional_stress':severity_scenario(.45).to_dict(),
        'warning':'importance interval asymptotic; inspect overlap and effective sample size'})
    _note('Running event-driven placement and checkpoint stress micro-cases.',verbose)
    clusters=[Cluster('A',8,1.,2.),Cluster('B',8,1.4,3.)]
    jobs=[Workload('training',0,12,64,8,('A','B'),checkpoint_hours=1.,interruption_allowed=False),
          Workload('inference-batch',1,8,24,4,('A','B'),checkpoint_hours=.5)]
    wr={'normal':simulate_workloads(clusters,jobs),
        'outage':simulate_workloads(clusters,jobs,[Outage('A',1.4,4.2)]),
        'fragmentation':simulate_workloads([Cluster('A',4,1,2),Cluster('B',4,1,2)],
                        [Workload('gang',0,8,16,8,('A','B'))])}
    write_json(output/'workload_stress.json',wr)
    summary={'project':'Compute Market','synthetic':True,'release_seed':seed,
             'paths':paths,'training_paths':train_paths,'stress_paths':stress_paths,'steps':steps,
             'selected_policy':asdict(selected),'stress_scenarios':len(stresses),
             'elapsed_seconds':time.time()-start,'python':platform.python_version(),
             'numpy':np.__version__,'scipy':scipy.__version__,
             'no_live_market_calibration':True,
             'risk_scope':'finite-tree robust optimizer + separate causal out-of-sample simulation',
             'headline':summaries['train_selected']}
    write_json(output/'summary.json',summary)
    _note('Experiment outputs written.',verbose)
    return summary
