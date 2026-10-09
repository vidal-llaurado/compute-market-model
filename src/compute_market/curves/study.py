"""Reproducible curve identities, calibration, pricing, control and stress evidence."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json,time,os
import numpy as np
from .portfolio import (example_case,economy_tape,stress_economy,CurveCase,EconomyTape,hardware_marks)
from .model import CurveModel
from .defaults import example_model
from .claims import (DeliveryStrip,continuous_strip,forecast_spot,european_option,train_swing,evaluate_swing,finite_difference_greeks)
from .observations import Quote,observe,FactorFilter,fit_physical_drift,fit_initial_curve
from .continuation import train_continuation,ContinuationSurface
from .control import run_controller,ConditionalInnovations
from .risk import project,factor_attribution,constrained_project
from .adapter import finite_case
from ..tails import tree_tail_diagnostics
from ..risk import expected_shortfall


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [plain(a) for a in v]
    return v


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(plain(value),indent=2,allow_nan=False))


def identity_experiments(case,paths=32768):
    m=case.model;x=np.zeros(m.dimension);t=m.simulate([0,.25,.5,1.,2.],paths,6201,'Q');rows=[]
    for name in ('G1','G2','power_US'):
        for n,u in enumerate(t.times[1:],1):
            F=m.value(name,u,t.factors[:,n],u);se=F.std(ddof=1)/paths**.5
            rows.append({'market':name,'delivery':u,'initial_future':m.initial(name,u),'mean_terminal_spot':F.mean(),'standard_error':se,'z_score':(F.mean()-m.initial(name,u))/se})
    D=np.exp(-t.integrated_rates[:,-1]);B=m.discount(0,x,2.)
    fp=m.value('G1',2,t.factors[:,-1],2.);forward=m.forward('G1',0,x,2.)
    ratio=np.mean(D*fp)/D.mean();ratio_se=np.std(D*(fp-ratio),ddof=1)/(D.mean()*paths**.5)
    overlap=continuous_strip(m,'G1',0,x,.1,1.7)-(.7*continuous_strip(m,'G1',0,x,.1,.8)+.9*continuous_strip(m,'G1',0,x,.8,1.7))/1.6
    p=m.simulate([0,1.],paths,6202,'P');real=m.value('G1',1.,p.factors[:,-1],1.)
    return {'paths':paths,'futures_martingales':rows,'discount':{'analytic':B,'monte_carlo':D.mean(),'standard_error':D.std(ddof=1)/paths**.5},
       'terminal_forward':{'analytic':forward,'ratio_mc':ratio,'ratio_standard_error':ratio_se,'ratio_z_score':(ratio-forward)/ratio_se,'futures':m.initial('G1',2.)},'delivery_overlap_error':overlap,
       'physical_forecast':{'analytic':forecast_spot(m,'G1',0,x,1.),'mc':real.mean(),'standard_error':real.std(ddof=1)/paths**.5,'futures':m.initial('G1',1.)}}


def filtering_experiment(case,dates=60):
    m=case.model;tape=m.simulate(np.linspace(0,.75,dates),1,7135,'P');rng=np.random.default_rng(635);f=FactorFilter(0,np.zeros(m.dimension),np.eye(m.dimension)*.005)
    estimates=[];ranks=[];observed=[];rmse=[]
    for n,t in enumerate(tape.times):
        f.predict(m,float(t));quotes=[]
        for name in ('G1','G2','G1_SG','G2_US','power_US'):
            for lag in (.25,1.,2.):
                T=t+lag;y=float(m.value(name,t,tape.factors[0,n],T));se=.008 if 'power' not in name else .3
                quotes.append(Quote(name,T,T,y+rng.normal()*se,se))
        diagnostic=f.update(m,quotes);ranks.append(diagnostic['observed_rank']);estimates.append(f.mean.copy());observed.append([q.price for q in quotes])
        test=[Quote('G2_SG',t+.7,t+.7,0.,.01),Quote('G1_US',t+1.5,t+1.5,0.,.01)]
        rmse.append([(observe(m,q,t,f.mean)-observe(m,q,t,tape.factors[0,n])) for q in test])
    return {'times':tape.times,'truth':tape.factors[0],'estimates':estimates,'minimum_observed_rank':min(ranks),'maximum_observed_rank':max(ranks),
            'held_out_quote_rmse':float(np.sqrt(np.mean(np.asarray(rmse)**2))),'observations_per_date':15,'filter':'extended Kalman, exact affine transition moments, nonlinear strip measurement'},np.array(estimates)


def calibration_experiment(case):
    m=case.model;rng=np.random.default_rng(63);quotes=[]
    for T in m.curves['G1'].dates:quotes.append(Quote('G1',T,T,float(m.initial('G1',T))+rng.normal()*.003,.006))
    for a,b in ((0,.5),(.5,1),(1,2),(2,3)):
        quotes.append(Quote('G1',a,b,float(continuous_strip(m,'G1',0,np.zeros(m.dimension),a,b))+rng.normal()*.003,.006))
    d=m.to_dict();d['curves'][0]['values']=[2.]*len(d['curves'][0]['values']);fit,report=fit_initial_curve(CurveModel.from_dict(d),'G1',quotes,.01)
    report['fitted_knot_values']=fit.curves['G1'].values;report['true_knot_values']=m.curves['G1'].values;return report


def greek_experiment(case):
    # CRN re-optimized finite-control Greeks validate the curve risk map at a small
    # model-generated quadrature. They are not fitted to the hedging experiment.
    seed=816;bump=.001;d=[];basecase=finite_case(case,4,2,seed);base=basecase.solve(tie_break=False)
    for i in range(case.model.dimension):
        e=np.zeros(case.model.dimension);e[i]=bump
        up=finite_case(case,4,2,seed,initial_state=e).solve(tie_break=False).value
        dn=finite_case(case,4,2,seed,initial_state=-e).solve(tie_break=False).value
        d.append((up-dn)/(2*bump))
    d=np.array(d);H=np.array([h.delta(case.model,0,np.zeros(case.model.dimension)) for h in case.instruments]);_,_,C=case.model.transition_moments(1/12,'P');proj=project(d,H,C)
    # A collateral appraisal is a separate functional; its delta travels into the borrowing base.
    hw=finite_difference_greeks(lambda x:hardware_marks(case.model,0,x,case.fleet)@np.asarray(case.fleet.blocks),np.zeros(case.model.dimension),1e-4)
    contrib=factor_attribution(d,C)
    bounds=np.array([h.bounds(0) for h in case.instruments]);im=np.array([float(h.margin(case.model,0,np.zeros(case.model.dimension))) for h in case.instruments])
    bounded=constrained_project(d,H,C,bounds[:,0],bounds[:,1],im,case.fleet.initial_cash-case.fleet.reserve,trade_limit=np.array([h.trade_limit for h in case.instruments]))
    return {'factors':case.model.factors,'portfolio_value':base.value,'factor_delta':d,'bump':bump,'instrument_names':[h.name for h in case.instruments],
        'instrument_deltas':H,'projection':proj,'constrained_projection':bounded,'covariance':C,'variance_euler_contributions':contrib,'collateral_delta':hw['delta'],
        'borrowing_base_delta':case.fleet.advance_rate*hw['delta'],'diagnostic':'local covariance projection; jump convexity and funding remain in full revaluation'}


def _path_job(payload):
    casefile,tapefile,surfacefile,policyfile,policy,atoms,output=payload
    case=CurveCase.load(casefile);tape=EconomyTape.load(tapefile);surface=ContinuationSurface.load(surfacefile);pm=CurveModel.from_dict(json.loads(Path(policyfile).read_text()))
    protocol=json.loads((Path(casefile).parent/'protocol.json').read_text())
    innovations=ConditionalInnovations.load(Path(casefile).parent/'conditional_innovations.npz')
    result=run_controller(case,tape,surface,policy=policy,scenarios=atoms,policy_model=pm,
        alpha=protocol['alpha_conditional'],risk_weight=protocol['risk_weight'],
        seed=protocol['conditional_seed'],carry=protocol['carry'],innovations=innovations)
    result.save(output);return tape.name,policy,result.summary()


def paired_summary(a,b,seed=19,bootstrap=200):
    with np.load(a,allow_pickle=False) as z:wa=z['terminal_wealth'];fa=z['failed']
    with np.load(b,allow_pickle=False) as z:wb=z['terminal_wealth'];fb=z['failed']
    n=len(wa);p=np.full(n,1/n);rng=np.random.default_rng(seed);dif=[]
    for _ in range(bootstrap):
        ix=rng.integers(0,n,n);dif.append(expected_shortfall(-wa[ix],p,.95)-expected_shortfall(-wb[ix],p,.95))
    return {'mean_wealth_change':float((wb-wa).mean()),'mean_change_standard_error':float((wb-wa).std(ddof=1)/n**.5),
            'es_reduction':float(expected_shortfall(-wa,p,.95)-expected_shortfall(-wb,p,.95)),
            'paired_es_bootstrap_interval_95':np.quantile(dif,[.025,.975]).tolist(),'failure_fraction_change':float(fb.mean()-fa.mean()),'bootstrap_resamples':bootstrap}


def run_study(output,*,quick=False,workers=4,paths=1024,training_paths=4096,stress_paths=512,atoms=256):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);start=time.time()
    if quick:paths=32;training_paths=256;stress_paths=16;atoms=32
    from .portfolio import project_case,project_economy
    case=project_case(paths=paths,steps=8,seed=20260918);case.save(root/'case.json')
    protocol={'data':'synthetic','market_state':'nine-factor affine OU--Levy Musiela curves','training_paths':training_paths,'holdout_paths':paths,'stress_paths_per_case':stress_paths,
      'conditional_atoms':atoms,'delivery_dates':list(np.arange(9)/8),'alpha_conditional':.9,'alpha_report':.95,'training_seed':13001,'holdout_seed':13002,'stress_seed':13003,
      'conditional_seed':938,'risk_weight':.55,'carry':'adverse_only','policies':['unhedged','static','adaptive'],'workers':workers,'continuation':'quadratic state regression of a specified physical plan','rebalancing':'joint current dispatch, financing, dated futures and funded claims', 'liquidity_law':case.assumptions['liquidity_spec']}
    write(root/'protocol.json',protocol)
    train,train_liquidity=project_economy(case,training_paths,13001);surface=train_continuation(case,train);surface.save(root/'continuation.npz')
    pm,fit=fit_physical_drift(case.model,train.curve);write(root/'policy_model.json',pm.to_dict());write(root/'physical_fit.json',fit)
    ConditionalInnovations.generate(pm,case.fleet.steps,atoms,case.fleet.year_fraction,protocol['conditional_seed']).save(root/'conditional_innovations.npz')
    train.curve.save(root/'training_factors.npz');write(root/'continuation_diagnostics.json',surface.diagnostics)
    tape,holdout_liquidity=project_economy(case,paths,13002);np.savez_compressed(root/'liquidity.npz',state=holdout_liquidity['state']);tape.save(root/'holdout_tape.npz')
    base,stress_liquidity=project_economy(case,stress_paths,13003);base.save(root/'stress_base.npz')
    names=['level_collapse','long_end_repricing','generation_transition','regional_scarcity','power_and_rates','basis_break','benchmark_dislocation','margin_call','execution_freeze','lender_haircut','outage','liquidity_spiral','counterparty_default']
    if quick:names=['long_end_repricing','margin_call']
    tapes=[('nominal',root/'holdout_tape.npz')];(root/'stresses').mkdir(exist_ok=True)
    for name in names:
        stressed=stress_economy(case,base,name);p=root/'stresses'/f'{name}_tape.npz';stressed.save(p);tapes.append((name,p))
    tasks=[]
    for name,path in tapes:
        for policy in ('unhedged','static','adaptive'):
            dest=root/f'holdout_{policy}.npz' if name=='nominal' else root/'stresses'/f'{name}_{policy}.npz'
            tasks.append(tuple(map(str,(root/'case.json',path,root/'continuation.npz',root/'policy_model.json')))+(policy,atoms,str(dest)))
    summary={}
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name,policy,result in pool.map(_path_job,tasks):
                summary.setdefault(name,{})[policy]=result;write(root/'path_summary.json',summary)
    else:
        for task in tasks:
            name,policy,result=_path_job(task);summary.setdefault(name,{})[policy]=result
        write(root/'path_summary.json',summary)
    write(root/'policy_comparison.json',{'adaptive_vs_unhedged':paired_summary(root/'holdout_unhedged.npz',root/'holdout_adaptive.npz',bootstrap=30 if quick else 300),
          'adaptive_vs_static':paired_summary(root/'holdout_static.npz',root/'holdout_adaptive.npz',bootstrap=30 if quick else 300)})
    identity=identity_experiments(case,2048 if quick else 65536);write(root/'pricing_identities.json',identity)
    filtration,_=filtering_experiment(case,12 if quick else 80);write(root/'filtering.json',filtration)
    write(root/'initial_curve_fit.json',calibration_experiment(case));write(root/'curve_greeks.json',greek_experiment(case))
    options=[]
    for T in (.25,.5,1.):
        options.append({'expiry':T,'delivery':1.,'jump_put':european_option(case.model,'G1',0,np.zeros(9),T,1.,2.3,paths=2048 if quick else 65536),
               'diffusion_put':european_option(example_model(False),'G1',0,np.zeros(9),T,1.,2.3)})
    write(root/'options.json',options)
    swing=train_swing(case.model,'G1',[0,.25,.5,.75,1.],2.3,2,paths=512 if quick else 8192)
    sw=evaluate_swing(case.model,swing,paths=512 if quick else 8192);np.savez_compressed(root/'swing_policy.npz',coefficients=swing.coefficients,scales=swing.scales,dates=swing.dates)
    np.savez_compressed(root/'swing_payoffs.npz',payoffs=sw.pop('payoffs'));write(root/'swing.json',sw)
    prices=[]
    (root/'pricing').mkdir(exist_ok=True)
    for b in ([4,8] if quick else [8,16,32]):
        for seed in ([611] if quick else [611,722,833]):
            c=finite_case(case,b,2,seed);q=c.quote(price_tolerance=.05);key=f'b{b}_s{seed}';c.save(root/'pricing'/f'{key}_case.json');write(root/'pricing'/f'{key}_quote.json',q)
            value=c.solve(tie_break=False);tails=tree_tail_diagnostics(c.portfolio.tree,value.policy['risk_loss'],c.portfolio.alpha)
            prices.append({'branches':b,'seed':seed,'nodes':c.portfolio.tree.size,'reservation':q['reservation'],'root_incremental_hedge':q['root_incremental_hedge'],'certificate':q['pricing_certificate'],'tail_resolution':tails})
    write(root/'reservation_surface.json',prices)
    write(root/'summary.json',{'project':'Compute Market','model_sha256':case.model.fingerprint,'case_sha256':case.fingerprint,'protocol':protocol,'paths':summary,'wall_seconds':time.time()-start})
    from .system_study import run_system_study,run_instrument_ablation
    run_system_study(root,quick);run_instrument_ablation(root,quick)
    return root
