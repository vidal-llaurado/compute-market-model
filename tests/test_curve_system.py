"""Economic identities, independent bounds and causal control regressions."""
from dataclasses import replace, asdict
import json
import numpy as np
import pytest
from compute_market.curves.defaults import example_model
from compute_market.curves.model import CurveModel, CurveSpec, AffineCurveModel, CurveTape
from compute_market.curves.interfaces import MarketModel, TradableClaim
from compute_market.curves.buckets import (CurveBucket,key_delivery_buckets,bump_curves,bucket_greeks,bucket_factor_jacobian,bridge_risk)
from compute_market.curves.claims import DeliveryStrip, finite_difference_greeks, european_option
from compute_market.curves.option_pricing import option_price, option_convergence
from compute_market.curves.instruments import CurveClaim, diversified_instruments, instrument_from_dict
from compute_market.curves.risk import default_instruments
from compute_market.curves.settlement import settle, marks, funded_prices, conditional_gains, reference_model
from compute_market.curves.liquidity import LiquiditySpec, apply_liquidity
from compute_market.curves.portfolio import project_case, project_economy, example_case, economy_tape, EconomyTape, CurveCase
from compute_market.curves.continuation import train_continuation
from compute_market.curves.control import run_controller
from compute_market.curves.fleet_options import (AssetSpec,train_fleet_policy,evaluate_fleet_policy,FleetPolicy,liquidation_proceeds,hardware_value_set)
from compute_market.curves.calibration import (Parameter,Parameterization,MarketDatum,CalibrationData,calibrate,bootstrap_calibration,predictive_decomposition,latent_loglikelihood,calibrate_latent)

@pytest.fixture(scope='module')
def m():return example_model()
@pytest.fixture(scope='module')
def x():return np.zeros(9)

def test_public_interfaces(m):
    assert isinstance(m,MarketModel); assert isinstance(default_instruments()[0],TradableClaim)
    assert AffineCurveModel is CurveModel
    assert isinstance(diversified_instruments(m)[-1],TradableClaim)

@pytest.mark.parametrize('dates',[[0,.25,1,3],[0,1],[.5,1,2,4]])
def test_key_delivery_partition(dates):
    b=key_delivery_buckets('G1',dates);z=np.linspace(dates[0],dates[-1],89)
    assert sum(v.weights(z) for v in b)==pytest.approx(np.ones(len(z)))
    assert np.array([v.weights(dates) for v in b])==pytest.approx(np.eye(len(dates)))

@pytest.mark.parametrize('left,center,right,bump',[(-1,0,1,.1),(0,2,1,.1),(0,0,0,.1),(0,1,2,0),(0,1,2,np.nan)])
def test_bucket_bad_spec(left,center,right,bump):
    with pytest.raises(ValueError):CurveBucket('G1',left,center,right,bump)

@pytest.mark.parametrize('market',['G1','G2_SG','power_US','funding','rate'])
def test_curve_bump_locality_and_initial_condition(m,market):
    b=key_delivery_buckets(market,[0,.5,1,2,7]);s=np.zeros(len(b));s[2]=.001
    perturbed=bump_curves(m,b,s)
    if market=='rate':assert np.interp(1,perturbed.rate_dates,perturbed.rate_forwards)-np.interp(1,m.rate_dates,m.rate_forwards)==pytest.approx(.001)
    else:
        a=m.initial(market,1);v=perturbed.initial(market,1)
        assert (v-a if m.curves[market].kind=='normal' else np.log(v/a))==pytest.approx(.001)
        assert perturbed.initial(market,0)==pytest.approx(m.initial(market,0))
    assert perturbed.fingerprint!=m.fingerprint

@pytest.mark.parametrize('market',['G1','power_US'])
def test_bucket_delta_bridge_exact_at_delivery_nodes(m,x,market):
    b=key_delivery_buckets(market,[0,.5,1,2,7]);claim=DeliveryStrip(market,(.5,1,2),(100,200,300))
    r=bucket_greeks(m,lambda mm:claim.mark(mm,0,x),b,cross_gamma=True)
    J=bucket_factor_jacobian(m,b);bridge=bridge_risk(r['delta'],J,claim.greeks(m,0,x)[0])
    assert bridge['interpolation_residual']==pytest.approx(np.zeros(9),abs=5e-5)
    assert np.max(np.abs(r['bump_consistency']))<1e-4
    assert r['gamma']==pytest.approx(r['gamma'].T)

def test_unspanned_shape_shock(m):
    b=key_delivery_buckets('G1',np.linspace(0,7,20));J=bucket_factor_jacobian(m,b)
    r=bridge_risk(np.arange(20.),J);P=r['bucket_shock_projector']
    assert P@P==pytest.approx(P,abs=1e-8)
    assert np.linalg.norm(J.T@r['unspanned_bucket_delta'])<1e-7
    assert np.linalg.norm(r['unspanned_bucket_delta'])>0

@pytest.mark.parametrize('expiry',[.125,.5,1.])
@pytest.mark.parametrize('put',[True,False])
def test_transform_matches_independent_mc(m,x,expiry,put):
    price=option_price(m,'G1',0,x,expiry,1,2.3,put)
    mc=european_option(m,'G1',0,x,expiry,1,2.3,put=put,paths=20000,seed=984)
    assert abs(price-mc['price'])<5*mc['standard_error']+1e-8

@pytest.mark.parametrize('jump',[False,True])
@pytest.mark.parametrize('strike',[1.8,2.3,2.8])
def test_transform_batch_parity_and_bounds(jump,strike):
    m=example_model(jump);X=np.zeros((3,9));X[:,0]=[-.2,0,.2];T=1.
    p=option_price(m,'G1',0,X,T,T,strike,True);c=option_price(m,'G1',0,X,T,T,strike,False)
    expected=m.discount(0,X,T)*(m.forward('G1',0,X,T)-strike)
    assert c-p==pytest.approx(expected,abs=2e-10)
    assert np.all(p>=0);assert np.all(c>=0)
    assert np.all(p<=strike*m.discount(0,X,T)+1e-10)
    assert option_price(m,'G1',0,X[1],T,T,strike,True)==pytest.approx(p[1])

def test_option_quadrature_converges(m,x):
    r=option_convergence(m,'G1',0,x,.75,1,2.3)
    assert abs(r['quadrature_change'])<1e-7

def test_option_expiry_and_invalid(m,x):
    assert option_price(m,'G1',.5,x,.5,1,3.)==pytest.approx(max(3-m.value('G1',.5,x,1),0))
    with pytest.raises(ValueError):option_price(m,'power_US',0,x,1,1,70)
    with pytest.raises(ValueError):option_price(m,'G1',1,x,.5,1,2.3)

@pytest.mark.parametrize('kind',['option','forward','swap','cap'])
def test_claim_roundtrip_and_protocol(m,kind):
    base=dict(name=kind,market='G1',delivery=1.,multiplier=100.,position_limit=10,trade_limit=5,impact_depth=2,spread=.1,impact_slope=.1,kind=kind)
    if kind in ('swap','cap'):base.update(payment_dates=(.5,1),weights=(.5,.5))
    c=CurveClaim(**base);copy=instrument_from_dict(json.loads(json.dumps(asdict(c))))
    assert isinstance(copy,TradableClaim);assert c.mark(m,0,np.zeros(9))==pytest.approx(copy.mark(m,0,np.zeros(9)))
    assert c.bounds(1)==(0,0)

@pytest.mark.parametrize('kind',['option','cap'])
def test_naked_option_requires_margin(m,x,kind):
    c=CurveClaim('short','G1',1,100,10,5,2,.1,.1,kind=kind,direction='short',payment_dates=(1,),weights=(1,))
    with pytest.raises(ValueError):c.margin(m,0,x)
    c=replace(c,minimum_margin=10);assert c.margin(m,0,x)==10

def test_paid_put_has_no_initial_margin(m,x):
    c=diversified_instruments(m)[11];assert c.funded
    assert c.margin(m,0,x)==0;assert c.mark(m,0,x)>0
    assert np.linalg.norm(c.delta(m,0,x))>0
    G=c.gamma(m,0,x);assert G==pytest.approx(G.T)

def test_funded_marks_do_not_become_cash(m,x):
    put=diversified_instruments(m)[11];old=np.array([2.]);initial=marks([put],m,0,x)
    y=x.copy();y[0]=-.2
    r=settle([put],m,0,.5,y,old,initial)
    assert r['cash']==0;assert r['positions']==pytest.approx(old)
    assert r['marks'][0]>initial[0]
    cash=settle([put],m,.5,1,y,old,r['marks'])
    assert cash['cash']==pytest.approx(2*put.multiplier*put.settlement(m,1,y))
    assert cash['positions'][0]==0

def test_futures_mark_is_cash(m,x):
    fut=default_instruments()[2];y=x.copy();y[0]=.1
    previous=marks([fut],m,0,x);r=settle([fut],m,0,.5,y,[3],previous)
    assert r['cash']==pytest.approx(3*fut.multiplier*(fut.mark(m,.5,y)-previous[0]))
    assert funded_prices([fut],previous)[0]==0

def test_periodic_cash_claim_aggregation(m,x):
    c=CurveClaim('swap','G1',1,100,10,5,2,.1,.1,kind='swap',payment_dates=(.5,1),weights=(.5,.5))
    r=settle([c],m,0,.5,x,[1],marks([c],m,0,x))
    assert r['cash']==pytest.approx(50*(m.value('G1',.5,x,.5)-c.strike))
    assert r['positions'][0]==1
    h=conditional_gains([c],m,0,x,.5,np.array([x,x]))
    expected=100*(c.mark(m,.5,x)+c.settlement(m,.5,x)-c.mark(m,0,x))
    assert h[:,0]==pytest.approx([expected,expected])

def test_periodic_cap_is_not_cap_on_average(m,x):
    c=diversified_instruments(m)[-1]
    assert 'periodic' in c.name
    assert c.mark(m,0,x)==pytest.approx(sum(.25*option_price(m,'G1',0,x,T,T,c.strike,False) for T in c.payment_dates))

@pytest.mark.parametrize('q',[1.,-1.])
def test_counterparty_closeout_sign(m,x,q):
    c=replace(diversified_instruments(m)[11],counterparty='supplier',recovery=.35)
    r=settle([c],m,0,.5,x,[q],marks([c],m,0,x),counterparty_alive=False)
    exposure=q*c.multiplier*c.mark(m,.5,x)
    assert r['cash']==pytest.approx(exposure*.35 if exposure>0 else exposure)
    assert r['positions'][0]==0

def test_basis_reprices_option_not_just_its_mark(m,x):
    p=diversified_instruments(m)[11]
    high=reference_model(m,[1.4,1.]);assert p.mark(high,0,x)<p.mark(m,0,x)
    assert high.value('G1_US',0,x,1)==m.value('G1_US',0,x,1)

def test_registry_adapter_handles_cash_claims():
    from compute_market.curves.adapter import finite_case
    c=project_case(paths=2);v=finite_case(c,3,2,818)
    assert set(v.market.kinds)=={'futures','cash','otc'}
    j=v.market.names.index('G1 downside put');assert v.market.mark[0,j]>0
    assert np.all(v.market.mark[v.portfolio.tree.leaves,j]==0)
    assert np.any(v.market.distribution[v.portfolio.tree.leaves,j]>0)
    r=v.solve();assert r.audit['independent_cash_error']<1e-5 if 'independent_cash_error' in r.audit else r.status=='optimal'

@pytest.mark.parametrize('field,value',[('mean_reversion',-1),('annual_volatility',-1),('min_multiplier',0),('max_multiplier',.5)])
def test_liquidity_spec_validation(field,value):
    with pytest.raises(ValueError):LiquiditySpec(**{field:value})

def test_liquidity_wrong_way_and_replay(m):
    s=LiquiditySpec();z=np.array([-1.,0.,1.,4.]);spread,depth,margin=s.multipliers(z)
    assert np.all(np.diff(spread)>0);assert np.all(np.diff(margin)>0);assert np.all(np.diff(depth)<0)
    assert depth[-1]==0
    tape=m.simulate([0,.25,.5],100,51)
    assert np.array_equal(s.simulate(m,tape,521),s.simulate(m,tape,521))

def test_counterparty_tape_saved(tmp_path):
    c=project_case(paths=4);t,d=project_economy(c,4,882);t.save(tmp_path/'t.npz')
    r=EconomyTape.load(tmp_path/'t.npz');assert np.array_equal(t.counterparty_alive,r.counterparty_alive)
    assert np.array_equal(t.copy().depth_multiplier,t.depth_multiplier)
    c.save(tmp_path/'c.json');assert CurveCase.load(tmp_path/'c.json').fingerprint==c.fingerprint

@pytest.fixture(scope='module')
def joint():
    c=project_case(paths=4);train,_=project_economy(c,256,193);t,_=project_economy(c,4,194)
    return c,t,train_continuation(c,train)

@pytest.mark.parametrize('policy',['unhedged','static','adaptive'])
def test_joint_claim_controller_accounting(joint,policy):
    c,t,s=joint;r=run_controller(c,t,s,policy=policy,scenarios=32)
    assert np.isfinite(r.terminal_wealth).all()
    assert r.audit['maximum_component_error']<1e-5
    assert r.audit['maximum_prefunding_violation']<1e-5
    assert set(r.audit['claim_types'])=={'futures','cap','forward','option'}

def test_joint_controller_prefix_causality(joint):
    c,t,s=joint;modified=t.copy();modified.curve.factors[:,4:,0]-=.3
    a=run_controller(c,t,s,scenarios=32);b=run_controller(c,modified,s,scenarios=32)
    assert np.array_equal(a.positions[:,:4],b.positions[:,:4])

def test_joint_controller_default_and_no_resurrection(joint):
    c,t,s=joint;t=t.copy();t.counterparty_alive[:,3:]=False
    r=run_controller(c,t,s,scenarios=32)
    j=next(i for i,h in enumerate(c.instruments) if getattr(h,'counterparty','cleared')=='supplier')
    assert np.all(r.positions[:,3:,j]==0)

def test_liquidation_value_different_from_appraisal():
    marks=np.array([100.,200.]);q=np.array([2.,3.]);p=liquidation_proceeds(marks,q,[2,3],.2,.1)
    assert p==pytest.approx(marks*q*.7)
    with pytest.raises(ValueError):liquidation_proceeds(100,-1,2)

@pytest.fixture(scope='module')
def asset_objects(m):
    a=AssetSpec('G1_US','power_US',8,.001,1.5,800,2500,500,2000,3000,.3,tuple([.2,0,.2,0,0,0,0,-.2,0]),2)
    p=train_fleet_policy(m,a,np.linspace(0,2,17),paths=512,seed=56)
    return a,p

def test_fleet_policy_independent_bound(m,asset_objects):
    a,p=asset_objects;r=evaluate_fleet_policy(m,p,paths=256,seed=57)
    assert r['value']<=r['perfect_information_upper_mean']+1e-6
    assert np.all(r['cashflows']<=r['upper_cashflows']+1e-6)
    assert r['maximum_bound_violation']<1e-6

def test_fleet_minimum_operating_commitment(m,asset_objects):
    a,p=asset_objects;r=evaluate_fleet_policy(m,p,paths=128,seed=58)
    for path in r['actions']:
        starts=np.flatnonzero((path==1)&np.r_[True,path[:-1]!=1])
        for start in starts:assert np.all(path[start:start+2]==1)

def test_fleet_saved_policy_replays(m,asset_objects,tmp_path):
    a,p=asset_objects;p.save(tmp_path/'p.npz');q=FleetPolicy.load(tmp_path/'p.npz')
    r=evaluate_fleet_policy(m,p,paths=32,seed=88);s=evaluate_fleet_policy(m,q,paths=32,seed=88)
    assert np.array_equal(r['cashflows'],s['cashflows'])

def test_fleet_decision_no_future_information(m,asset_objects):
    a,p=asset_objects;X=np.repeat(np.zeros((1,9)),2,axis=0)
    act,state,r=p.choose(m,0,X,[0,0]);assert act[0]==act[1];assert r[0]==r[1]

def test_asset_states_and_bad_economics(asset_objects):
    a,_=asset_objects
    assert a.actions(2)==('run',)
    with pytest.raises(ValueError):a.next_state(2,'retire')
    with pytest.raises(ValueError):replace(a,gpus=-1)
    with pytest.raises(ValueError):replace(a,min_run_steps=0)

@pytest.fixture(scope='module')
def calibration_objects():
    m=CurveModel(('level','slope'),[.8,2.],[[.25,0],[.05,.18]],
        [CurveSpec('compute','log',(0.,1.,2.,3.),(2.5,2.4,2.2,2.),(1.,.5),'USD/GPU-hour')],
        physical_drift=[.03,-.01],jump_intensity_q=.3,jump_intensity_p=.5,
        jump_mean_q=[-.15,.03],jump_mean_p=[-.2,.04],jump_cov_q=[[.02,0],[0,.01]],jump_cov_p=[[.025,0],[0,.015]])
    params=[Parameter('decay',('kappa',0),.1,2.,.8),Parameter('vol',('diffusion',0,0),.05,.6,.25),
            Parameter('drift',('physical_drift',0),-.2,.2,.1),Parameter('Q jumps',('jump_intensity_q',),.05,1.,.3)]
    qs=[]
    for T in (.5,1.,2.):
        for K in (1.8,2.3,2.8):
            q=MarketDatum('compute',0,T,0,.001,'option',expiry=T,strike=K)
            qs.append(replace(q,value=q.predict(m)))
    d=CalibrationData(qs,m.simulate(np.linspace(0,2,25),512,21,'P'))
    return m,params,d

def test_joint_parameter_recovery(calibration_objects):
    m,p,d=calibration_objects;r=calibrate(m,p,d,anchors=['fixed loadings and remaining jump law'],initial=[.5,.2,0.,.2],max_evaluations=100)
    assert r.report['success'];assert r.report['locally_identified']
    assert r.report['estimates']==pytest.approx([.8,.25,.03,.3],abs=.035)

def test_calibration_requires_anchors(calibration_objects):
    m,p,d=calibration_objects
    with pytest.raises(ValueError):Parameterization(m,p,[])
    with pytest.raises(ValueError):Parameter('bad',('kappa',0),1,0)

def test_parameter_covariance_stays_psd(calibration_objects):
    m,_,_=calibration_objects;p=[Parameter('jump sigma',('jump_cholesky_q',0,0),.01,.5)]
    mapping=Parameterization(m,p,['other jump Cholesky entries fixed']);mm=mapping.model([.2])
    assert np.linalg.eigvalsh(mm.jump_cov_q).min()>0
    assert mm.jump_cov_q[0,0]==pytest.approx(.04)

@pytest.mark.parametrize('kind',['futures','forward','discount','rate','option'])
def test_calibration_observation_operators(calibration_objects,kind):
    m,_,_=calibration_objects;q=MarketDatum('compute',0,1,2.3,.01,kind,expiry=.5,strike=2.3)
    assert np.isfinite(q.predict(m))

def test_calibration_data_identification_not_prior(calibration_objects):
    m,_,_=calibration_objects
    params=[Parameter('P drift',('physical_drift',0),-.2,.2,.1,0,.05)]
    q=MarketDatum('compute',0,1,2.4,.01)
    r=calibrate(m,params,CalibrationData([q]),anchors=['all other coordinates fixed'])
    assert r.report['data_rank']==0;assert not r.report['locally_identified']

def test_bootstrap_and_total_variance(calibration_objects):
    m,p,d=calibration_objects
    params=[p[2]];data=CalibrationData([],d.transitions)
    models,r=bootstrap_calibration(m,params,data,anchors=['all but physical drift fixed'],replicates=3,seed=885,max_evaluations=50)
    assert r['converged']==3
    v=predictive_decomposition(models,'compute',0,np.zeros(2),1,paths=128)
    assert v['total_variance']==pytest.approx(v['process_variance']+v['parameter_variance'])

@pytest.fixture(scope='module')
def latent_history(calibration_objects):
    m,_,_=calibration_objects;t=m.simulate(np.linspace(0,1,13),1,991,'P');out=[]
    for n,time in enumerate(t.times):
        qs=[]
        for lag in (.2,1):
            q=MarketDatum('compute',float(time),float(time+lag),0.,.01)
            qs.append(replace(q,value=q.predict(m,t.factors[0,n])))
        out.append((float(time),qs))
    return out

def test_latent_filter_likelihood(calibration_objects,latent_history):
    m,_,_=calibration_objects;value,r=latent_loglikelihood(m,latent_history)
    assert np.isfinite(value);assert len(r['filtered_states'])==13
    assert np.linalg.eigvalsh(r['final_covariance']).min()>-1e-10

def test_latent_calibration_runs(calibration_objects,latent_history):
    m,_,_=calibration_objects;p=[Parameter('drift',('physical_drift',0),-.2,.2,.1)]
    r=calibrate_latent(m,p,latent_history,anchors=['factor gauge and all other laws fixed'],max_iterations=25)
    assert r.report['success'];assert np.isfinite(r.report['negative_loglikelihood'])


def test_latent_profile(calibration_objects,latent_history):
    from compute_market.curves.calibration import latent_profile
    m,p,_=calibration_objects
    r=latent_profile(m,p[2],latent_history,[-.1,0.,.1],anchors=['fixed nuisance parameters'])
    assert min(r['twice_objective_above_grid_minimum'])==0
    assert np.isfinite(r['negative_quasi_loglikelihood']).all()
    with pytest.raises(ValueError):latent_profile(m,p[2],latent_history,[0.,.1],anchors=['fixed'])


def test_serialized_calibration_input(calibration_objects,tmp_path):
    from compute_market.curves.calibration import fit_input_file
    m,p,d=calibration_objects
    d.transitions.save(tmp_path/'history.npz')
    (tmp_path/'input.json').write_text(json.dumps({'model':m.to_dict(),'parameters':[asdict(p[2])],
      'anchors':['all other coordinates fixed'],'transition_file':'history.npz','quotes':[]}))
    r=fit_input_file(tmp_path/'input.json',tmp_path/'fit.json')
    assert r.report['success'];assert (tmp_path/'fit.json').exists()


def test_public_exports():
    import compute_market.curves as c
    assert all(hasattr(c,k) for k in c.__all__)
