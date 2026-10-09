"""Curve risk, operating options and calibration experiments.

Each experiment saves its inputs, training decisions and independent evaluation."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json, time
import numpy as np
from .model import CurveModel,CurveSpec,CurveTape
from .portfolio import project_case,CurveCase,EconomyTape
from .claims import DeliveryStrip,finite_difference_greeks
from .buckets import key_delivery_buckets,bucket_greeks,bucket_factor_jacobian,bridge_risk,bump_curves
from .fleet_options import AssetSpec,train_fleet_policy,evaluate_fleet_policy,hardware_value_set
from .calibration import Parameter,Parameterization,MarketDatum,CalibrationData,calibrate,bootstrap_calibration,predictive_decomposition,latent_loglikelihood,calibrate_latent,latent_profile
from .option_pricing import option_price,option_convergence
from .study import write


def calibration_case(paths=4096):
    model=CurveModel(('level','slope'),[.8,2.],[[.25,0],[.05,.18]],
        [CurveSpec('compute','log',(0.,1.,2.,3.,5.),(2.5,2.4,2.2,2.,1.7),(1.,.5),'USD/GPU-hour')],
        physical_drift=[.03,-.01],jump_intensity_q=.3,jump_intensity_p=.5,
        jump_mean_q=[-.15,.03],jump_mean_p=[-.2,.04],jump_cov_q=[[.02,0],[0,.01]],jump_cov_p=[[.025,0],[0,.015]])
    params=[Parameter('level mean reversion',('kappa',0),.1,2.,.8),
      Parameter('level diffusion',('diffusion',0,0),.05,.6,.25),
      Parameter('physical level drift',('physical_drift',0),-.2,.2,.1),
      Parameter('pricing jump intensity',('jump_intensity_q',),.05,1.,.3)]
    rng=np.random.default_rng(15101);quotes=[]
    for T in (.5,1.,2.):
        for K in (1.8,2.3,2.8):
            q=MarketDatum('compute',0,T,0,.001,'option',expiry=T,strike=K,source='synthetic noisy option observation')
            quotes.append(replace(q,value=q.predict(model)+rng.normal()*q.standard_error))
    history=model.simulate(np.linspace(0,2,25),paths,15102,'P')
    anchors=['Initial curve fixed by a separate clean-price observation block.',
             'Factor signs/scales fixed by compute loading (1, 0.5).',
             'Slope decay/diffusion, jump means/covariances and all other coordinates fixed.',
             'Synthetic transition factors are observed for this recovery experiment.']
    return model,params,CalibrationData(quotes,history),anchors


def calibration_experiment(root,quick=False):
    out=Path(root)/'calibration';out.mkdir(parents=True,exist_ok=True)
    m,p,d,anchors=calibration_case(512 if quick else 4096)
    write(out/'inputs.json',{'model':m.to_dict(),'parameters':[asdict(z) for z in p],
      'anchors':anchors,'quotes':[asdict(z) for z in d.quotes],'initial':[.5,.2,0.,.2],
      'transition_file':'physical_history.npz','frequency_grid':d.frequencies,
      'scope':'synthetic recovery with explicit fixed nuisance parameters; real data not used'})
    d.transitions.save(out/'physical_history.npz')
    fit=calibrate(m,p,d,anchors=anchors,initial=[.5,.2,0.,.2]);fit.save(out/'fit.json')
    truth=Parameterization(m,p,anchors).vector()
    models,boot=bootstrap_calibration(fit.model,p,d,anchors=anchors,replicates=4 if quick else 24,seed=15103,max_evaluations=120)
    write(out/'parameter_uncertainty.json',boot)
    write(out/'bootstrap_models.json',[z.to_dict() for z in models])
    pred=predictive_decomposition(models,'compute',0,np.zeros(2),1,paths=512 if quick else 4096,seed=15104)
    held=[]
    for T in (.25,.75,1.5):
        for K in (2.,2.5):
            exact=option_price(m,'compute',0,np.zeros(2),T,T,K)
            fitted=option_price(fit.model,'compute',0,np.zeros(2),T,T,K)
            held.append({'expiry':T,'strike':K,'truth':float(exact),'fit':float(fitted),'error':float(fitted-exact)})
    # A distinct latent-history experiment sees prices, not the simulated states.
    times=np.linspace(0,3,49 if quick else 145);tape=m.simulate(times,1,15105,'P')
    rng=np.random.default_rng(15106);history=[]
    for n,t in enumerate(times):
        qs=[]
        for lag in (.1,.5,1.5):
            q=MarketDatum('compute',float(t),float(t+lag),0.,.008,source='synthetic historical clean quote')
            qs.append(replace(q,value=q.predict(m,tape.factors[0,n])+rng.normal()*.008))
        history.append((float(t),qs))
    latent=calibrate_latent(m,[p[2]],history,anchors=['Observation loading/gauge, jump law and remaining parameters fixed.'],max_iterations=100)
    latent.save(out/'latent_fit.json');_,filtered=latent_loglikelihood(latent.model,history)
    write(out/'latent_profile.json',latent_profile(latent.model,p[2],history,np.linspace(-.2,.2,21),anchors=['all nuisance coordinates fixed']))
    np.savez_compressed(out/'latent_history.npz',times=times,true_factors=tape.factors[0],filtered=np.asarray(filtered['filtered_states']))
    write(out/'latent_quotes.json',[{'time':t,'quotes':[asdict(z) for z in qs]} for t,qs in history])
    # A prior can select an estimate without increasing data rank.
    unidentified=calibrate(m,[replace(p[2],prior_mean=0.,prior_standard_error=.1)],
       CalibrationData([MarketDatum('compute',0,1,2.4,.01)]),anchors=['All other coordinates fixed.'])
    result={'names':[z.name for z in p],'truth':truth,'estimates':fit.report['estimates'],
       'report':fit.report,'bootstrap':{k:v for k,v in boot.items() if k!='estimates'},
       'predictive_uncertainty':pred,'heldout_options':held,
       'heldout_price_rmse':float(np.sqrt(np.mean([z['error']**2 for z in held]))),
       'latent_report':latent.report,'latent_factor_rmse':np.sqrt(np.mean((np.asarray(filtered['filtered_states'])-tape.factors[0])**2,axis=0)),
       'unidentified_data_rank':unidentified.report['data_rank'],'independent_paths':len(d.transitions.factors)}
    write(out/'summary.json',result);return result


def asset_experiment(case,root,quick=False):
    out=Path(root)/'assets';out.mkdir(parents=True,exist_ok=True)
    # A marginal cohort makes switching and retirement economically active.
    asset=AssetSpec('G1_US','power_US',64,.00065,1.65,12000,18000,3500,90000,
       500000,.65,(.25,.0,.18,0,0,-.01,-.1,-.35,0),min_run_steps=2,utilisation=.8)
    dates=np.linspace(0,3,25);paths=512 if quick else 8192
    policy=train_fleet_policy(case.model,asset,dates,paths=paths,seed=15201,measure='Q');policy.save(out/'operating_policy.npz')
    tape=case.model.simulate(dates,paths,15202,'Q');tape.save(out/'evaluation_tape.npz')
    result=evaluate_fleet_policy(case.model,policy,tape=tape)
    np.savez_compressed(out/'policy_outcomes.npz',cashflows=result.pop('cashflows'),upper=result.pop('upper_cashflows'),actions=result.pop('actions'))
    # Feasible comparison: start once, keep operating to the horizon.
    cf=np.zeros(paths)
    for n,t in enumerate(dates[:-1]):
        r=asset.running_cash(case.model,t,tape.factors[:,n],dates[n+1]-t)-(asset.startup_cost if n==0 else 0)
        cf+=np.exp(-tape.integrated_rates[:,n])*r
    cf+=np.exp(-tape.integrated_rates[:,-1])*asset.terminal_residual
    result['always_run_value']=float(cf.mean());result['always_run_standard_error']=float(cf.std(ddof=1)/np.sqrt(paths))
    result['immediate_sale_value']=float(asset.resale(0,np.zeros(9)))
    result['asset']=asdict(asset);result['decision_dates']=dates
    result['hardware_values']=hardware_value_set(case,result)
    write(out/'summary.json',result);return result


def risk_experiment(case,root,quick=False):
    out=Path(root)/'risk';out.mkdir(parents=True,exist_ok=True)
    m=case.model;x=np.zeros(m.dimension)
    dates=[0,.25,.5,1.,2.,3.,7.]
    buckets=sum([key_delivery_buckets(k,dates,bump=1e-4 if k!='power_US' else .01)
                 for k in ('G1_US','G2_SG','power_US')],[])
    a=DeliveryStrip('G1_US',(.25,.5,1.,2.,3.),(200000,180000,150000,100000,50000))
    b=DeliveryStrip('G2_SG',(.5,1.,2.),(120000,80000,40000))
    c=DeliveryStrip('power_US',(.25,.5,1.,2.,3.),(120,108,90,60,30))
    def clean(model):return float(a.mark(model,0,x)+b.mark(model,0,x)-c.mark(model,0,x))
    r=bucket_greeks(m,clean,buckets,cross_gamma=not quick);J=bucket_factor_jacobian(m,buckets)
    factor=a.greeks(m,0,x)[0]+b.greeks(m,0,x)[0]-c.greeks(m,0,x)[0]
    bridge=bridge_risk(r['delta'],J,factor)
    residual=bridge['unspanned_bucket_delta'];shock=residual/max(np.linalg.norm(residual),1)*.01
    nonlinear=float(clean(bump_curves(m,buckets,shock))-clean(m))
    r.update({'bridge':bridge,'factor_delta':factor,'out_of_span_test':{'bucket_shock':shock,
      'factor_projection_norm':float(np.linalg.norm(J.T@shock)),
      'linear_pnl':float(r['delta']@shock),'full_revaluation_pnl':nonlinear},
      'claim_scope':'node-supported linear clean delivery claims; negative electricity strip is an input-cost exposure'})
    write(out/'bucket_risk.json',r)
    # Reoptimized financed-book sensitivity, same innovations across every bump.
    from .adapter import finite_case
    small=[key_delivery_buckets('G1_US',[0,.5,1.,3.,7.])[i] for i in (1,2,3)]
    def private(model):return finite_case(replace(case,model=model),branches=3,periods=2,seed=15301).solve(tie_break=False).value
    private_risk=bucket_greeks(m,private,small)
    private_risk['numerical_method']='three branches, two intervals, common innovations; operating/financing/hedging reoptimized'
    write(out/'financed_book_buckets.json',private_risk)
    quadrature=[{'expiry':E,**option_convergence(m,'G1',0,x,E,1.,2.3)} for E in (.125,.5,1.)]
    write(out/'option_transform.json',quadrature)
    return {'bucket_count':len(buckets),'factor_span_rank':bridge['rank'],
      'factor_bridge_max_error':float(np.max(abs(bridge['interpolation_residual']))),
      'bucket_half_step_max_change':float(np.max(abs(r['bump_consistency']))),
      'out_of_span':r['out_of_span_test'],'private_value':private_risk['value'],
      'private_bucket_unit_pnl':private_risk['unit_pnl'],'option_quadrature':quadrature}


def run_system_study(root,quick=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.time()
    case=CurveCase.load(root/'case.json') if (root/'case.json').exists() else project_case()
    output={}
    for name,func in [('risk',lambda:risk_experiment(case,root,quick)),
                      ('assets',lambda:asset_experiment(case,root,quick)),
                      ('calibration',lambda:calibration_experiment(root,quick))]:
        output[name]=func();write(root/'system_summary.json',output);print(name+' experiment complete',flush=True)
    output['wall_seconds']=time.time()-start;write(root/'system_summary.json',output)
    return output


def run_instrument_ablation(root,quick=False):
    """Same operating tape and training, restricting the financial universe."""
    from .continuation import ContinuationSurface
    from .control import run_controller,ConditionalInnovations
    from .study import paired_summary
    root=Path(root);case=CurveCase.load(root/'case.json')
    small=replace(case,instruments=[h for h in case.instruments if not h.funded]);small.save(root/'futures_only_case.json')
    tape=EconomyTape.load(root/'holdout_tape.npz');continuation=ContinuationSurface.load(root/'continuation.npz')
    model=CurveModel.from_dict(json.loads((root/'policy_model.json').read_text()))
    protocol=json.loads((root/'protocol.json').read_text())
    result=run_controller(small,tape,continuation,policy='adaptive',scenarios=protocol['conditional_atoms'],seed=protocol['conditional_seed'],policy_model=model,
        alpha=protocol['alpha_conditional'],risk_weight=protocol['risk_weight'],carry=protocol['carry'],
        innovations=ConditionalInnovations.load(root/'conditional_innovations.npz'))
    result.save(root/'holdout_futures_only.npz')
    report={'futures_only':result.summary(),
      'mixed_vs_futures_only':paired_summary(root/'holdout_futures_only.npz',root/'holdout_adaptive.npz',bootstrap=30 if quick else 300),
      'matched':{'same_market_tape':True,'same_financial_and_physical_start':True,'same_conditional_seed':True,
        'futures_instruments':len(small.instruments),'mixed_instruments':len(case.instruments)},
      'interpretation':'independent conditional optimizations; a richer one-step feasible set does not guarantee better realized or global policy value'}
    write(root/'instrument_ablation.json',report);return report
