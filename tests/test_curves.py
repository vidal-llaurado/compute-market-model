from dataclasses import replace
import json
import numpy as np
import pytest
from compute_market.curves.model import CurveModel,CurveSpec,CurveTape,decay_integral
from compute_market.curves.defaults import example_model
from compute_market.curves.claims import (DeliveryStrip,continuous_strip,forecast_spot,european_option,finite_difference_greeks,train_swing,evaluate_swing)
from compute_market.curves.observations import (Quote,observe,observation_jacobian,FactorFilter,fit_initial_curve,fit_physical_drift)
from compute_market.curves.risk import project,Instrument,default_instruments,factor_attribution
from compute_market.curves.portfolio import example_case,CurveCase,economy_tape,stress_economy,hardware_marks
from compute_market.curves.continuation import train_continuation,ContinuationSurface
from compute_market.curves.control import run_controller
from compute_market.curves.adapter import finite_case

@pytest.fixture(scope='module')
def m():return example_model()
@pytest.fixture(scope='module')
def g():return example_model(False)

def test_zero_reversion_limit():
    assert decay_integral(np.array([0.,1.]),.5)==pytest.approx([.5,1-np.exp(-.5)])

@pytest.mark.parametrize('name',['G1','G2','G1_SG','power_US','power_SG','funding'])
def test_initial_curve_exact(m,name):
    c=m.curves[name]
    for T,v in zip(c.dates,c.values):assert m.value(name,0,np.zeros(m.dimension),T)==pytest.approx(v)

def test_model_json_roundtrip(m):
    c=CurveModel.from_dict(json.loads(json.dumps(m.to_dict())));assert c.fingerprint==m.fingerprint

def test_tape_roundtrip(m,tmp_path):
    t=m.simulate([0,.25,.5],20,13);p=tmp_path/'tape.npz';t.save(p);r=CurveTape.load(p)
    assert np.array_equal(t.factors,r.factors);assert r.model_sha256==m.fingerprint

@pytest.mark.parametrize('key,value',[('jump_intensity_q',-1),('kappa',[-1]*9),('diffusion',np.zeros((2,2)).tolist()),('rate_dates',[1,2])])
def test_model_invalid(m,key,value):
    d=m.to_dict();d[key]=value
    with pytest.raises(ValueError):CurveModel.from_dict(d)

def test_equivalent_measure_support_enforced(m):
    d=m.to_dict();d['jump_intensity_p']=0
    with pytest.raises(ValueError,match='support'):CurveModel.from_dict(d)
    d=m.to_dict();d['jump_cov_p']=np.zeros((9,9)).tolist()
    with pytest.raises(ValueError,match='support'):CurveModel.from_dict(d)

def test_drift_span_enforced(g):
    d=g.to_dict();d['diffusion']=np.zeros((9,9)).tolist()
    with pytest.raises(ValueError,match='span'):CurveModel.from_dict(d)

@pytest.mark.parametrize('name',['G1','G2','power_US'])
def test_q_martingale_analytic(m,name):
    t=.3;u=.7;T=1.4;x=np.arange(m.dimension)*.03
    # E[exp(ell X_u)] has OU cumulant known in closed quadrature.
    v,w=m._grid(T-u,T-t);l=m.loading(name,v)
    if m.curves[name].kind=='log':
        implied=m.initial(name,T)*np.exp(m.loading(name,T-t)@x-m.compensator(name,u,T)+float(w@m.cumulant(l)))
    else:implied=m.initial(name,T)+m.loading(name,T-t)@x-m.compensator(name,u,T)+float(w@(l@(m.jump_intensity_q*m.jump_mean_q)))
    assert implied==pytest.approx(m.value(name,t,x,T),rel=1e-11,abs=1e-11)

def test_fixed_delivery_rolls_without_reanchoring(g):
    t=.2;u=.6;T=1.;x=np.ones(g.dimension)*.2
    deterministic=x*np.exp(-g.kappa*(u-t))
    # Factor loading cancels mean reversion at fixed calendar delivery.
    assert g.loading('G1',T-u)@deterministic==pytest.approx(g.loading('G1',T-t)@x)
    assert g.loading('G1',T-t)[1]<g.loading('G1',T-u)[1]

@pytest.mark.parametrize('name',['G1','G2','power_US','power_SG'])
def test_q_mc_martingale(m,name):
    tape=m.simulate([0,1.],18000,481,'Q');z=m.value(name,1.,tape.factors[:,-1],1.)
    assert abs(z.mean()-m.initial(name,1.)) < 6*z.std()/len(z)**.5+1e-7

@pytest.mark.parametrize('name',['G1','power_US'])
def test_p_forecast_mc(m,name):
    tape=m.simulate([0,.7],18000,944,'P');z=m.value(name,.7,tape.factors[:,-1],.7)
    theory=forecast_spot(m,name,0,np.zeros(9),.7)
    assert abs(z.mean()-theory)<6*z.std()/len(z)**.5+1e-7

def test_exact_initial_discount_curve(m):
    # Piecewise-linear initial instantaneous USD rate: 3.5% to 3.7% in first year.
    assert m.discount(0,np.zeros(9),1.)==pytest.approx(np.exp(-.036),abs=1e-12)

def test_bond_mc_and_forward_convexity(m):
    z=m.simulate([0,1],30000,995,'Q');D=np.exp(-z.integrated_rates[:,-1]);F=m.value('G1',1,z.factors[:,-1],1)
    assert abs(D.mean()-m.discount(0,np.zeros(9),1))<6*D.std()/len(D)**.5
    y=D*(F-m.forward('G1',0,np.zeros(9),1))
    assert abs(y.mean())<6*y.std()/len(y)**.5
    assert abs(m.forward('G1',0,np.zeros(9),1)-m.value('G1',0,np.zeros(9),1))>1e-5

def test_normal_forward_convexity(m):
    z=m.simulate([0,1],16000,723,'Q');D=np.exp(-z.integrated_rates[:,-1]);F=m.value('power_US',1,z.factors[:,-1],1)
    y=D*(F-m.forward('power_US',0,np.zeros(9),1));assert abs(y.mean())<6*y.std()/len(y)**.5

def test_rates_allow_negative(g):
    x=np.zeros(9);x[6]=-2
    assert g.short_rate(0,x)<0;assert np.isfinite(g.discount(0,x,1))

def test_power_allows_negative(m):
    x=np.zeros(9);x[5]=-3
    assert m.value('power_US',0,x,0)<0

def test_zero_rate_loading_removes_convexity(m):
    d=m.to_dict();d['rate_loading']=[0]*9;z=CurveModel.from_dict(d)
    assert z.forward('G1',.3,np.zeros(9),1)==pytest.approx(z.value('G1',.3,np.zeros(9),1))

def test_funding_is_not_priced_as_tradable(m):
    with pytest.raises(ValueError):m.forward('funding',0,np.zeros(9),1)

def test_conditional_covariance_moments(m):
    dt=.3;x=np.ones((25000,9))*.1;out,_=m.step(x,dt,np.random.default_rng(818),'P');A,b,C=m.transition_moments(dt)
    assert np.max(np.abs(out.mean(0)-(A*x[0]+b))/(np.sqrt(np.diag(C)/len(x))+1e-10))<6
    assert np.diag(np.cov(out.T))==pytest.approx(np.diag(C),rel=.08)

def test_integrated_factors_zero_step(m):
    x=np.ones((3,9));y,I=m.step(x,0,np.random.default_rng(1),'Q')
    assert y==pytest.approx(x);assert I==pytest.approx(np.zeros_like(x))

@pytest.mark.parametrize('name',['G1','power_US'])
def test_greeks_match_bumps(m,name):
    x=np.arange(9)*.01;d,G=m.greeks(name,.1,x,1)
    b=finite_difference_greeks(lambda y:m.value(name,.1,y,1),x,1e-4)
    assert d==pytest.approx(b['delta'],abs=1e-7);assert G==pytest.approx(b['gamma'],abs=1e-5)

def test_overlap_delivery_aggregation(m):
    x=np.zeros(9);a=continuous_strip(m,'G1',.1,x,.2,.8);b=continuous_strip(m,'G1',.1,x,.8,1.4)
    all=continuous_strip(m,'G1',.1,x,.2,1.4);assert all==pytest.approx((a+b)/2,abs=1e-10)

def test_strip_linearity_and_greeks(m):
    s=DeliveryStrip('G1',(.5,1.),(200.,500.));x=np.ones(9)*.1
    assert s.mark(m,0,x)==pytest.approx(200*m.value('G1',0,x,.5)+500*m.value('G1',0,x,1))
    assert s.greeks(m,0,x)[0]==pytest.approx(finite_difference_greeks(lambda y:s.mark(m,0,y),x)['delta'],abs=1e-5)

def test_option_put_call_parity_gaussian(g):
    x=np.zeros(9);p=european_option(g,'G1',0,x,.5,1,2.3);c=european_option(g,'G1',0,x,.5,1,2.3,put=False)
    v,w=g._grid(0,.5);ell=g.loading('G1',.5+v);B=g.rate_B(v);conv=-float(w@np.einsum('ni,ij,nj->n',ell,g.cov,B))
    assert c['price']-p['price']==pytest.approx(g.discount(0,x,.5)*(g.value('G1',0,x,1)*np.exp(conv)-2.3))

def test_jump_option_seed_replay_and_positive(m):
    a=european_option(m,'G1',0,np.zeros(9),.5,1,2.3,paths=2000)
    b=european_option(m,'G1',0,np.zeros(9),.5,1,2.3,paths=2000)
    assert a==b;assert a['price']>0;assert a['standard_error']>0

def test_expiry_option_intrinsic(m):
    f=m.value('G1',.5,np.zeros(9),1)
    assert european_option(m,'G1',.5,np.zeros(9),.5,1,3)['price']==pytest.approx(max(3-f,0))

@pytest.mark.parametrize('convention',['futures','forward','normalized_physical'])
def test_observation_jacobian(m,convention):
    q=Quote('G1',.4,1.1,2.4,.01,convention,.2 if convention=='normalized_physical' else None);x=np.ones(9)*.01
    J=observation_jacobian(m,q,.2,x)
    numerical=finite_difference_greeks(lambda y:observe(m,q,.2,y),x)['delta']
    assert J==pytest.approx(numerical,abs=1e-6)

def test_physical_quote_requires_normalization():
    with pytest.raises(ValueError):Quote('G1',.5,1,3,.01,'normalized_physical')

def test_filter_reduces_observation_error(m):
    x=np.array([.12,.08,-.05,.03,.04,0,0,0,0]);qs=[]
    for name in ('G1','G2','G1_SG','G2_US','power_US'):
        for T in (.5,1.,2.):qs.append(Quote(name,T,T,float(m.value(name,.2,x,T)),.005))
    f=FactorFilter(.2,np.zeros(9),np.eye(9)*.1);prior=sum((observe(m,q,.2,f.mean)-q.price)**2 for q in qs);d=f.update(m,qs)
    after=sum((observe(m,q,.2,f.mean)-q.price)**2 for q in qs)
    assert after<prior*.05;assert d['observed_rank']<9;assert np.linalg.eigvalsh(f.covariance).min()>-1e-10
    r=FactorFilter.from_dict(f.to_dict());assert r.mean==pytest.approx(f.mean)
    with pytest.raises(ValueError):f.predict(m,.1)

def test_initial_strip_fit(m):
    quotes=[Quote('G1',T,T,float(m.initial('G1',T)),.001) for T in m.curves['G1'].dates]
    d=m.to_dict();d['curves'][0]['values']=[2.0]*len(d['curves'][0]['values']);bad=CurveModel.from_dict(d)
    fitted,diag=fit_initial_curve(bad,'G1',quotes,penalty=1e-7)
    assert diag['weighted_quote_rmse']<1e-4;assert fitted.initial('G1',1)==pytest.approx(2.3,abs=1e-6)

def test_projection_span_and_residual():
    C=np.diag([2.,3.,4.]);d=np.array([1.,2.,3.]);H=np.array([[1.,0,0],[0,1.,0]])
    r=project(d,H,C);assert r['positions']==pytest.approx([-1.,-2.]);assert r['residual_delta']==pytest.approx([0,0,3]);assert r['residual_variance']==pytest.approx(36.)
    assert sum(factor_attribution(d,C))==pytest.approx(r['unhedged_variance'])

def test_projection_not_euclidean():
    C=np.array([[2.,.8],[.8,1.]])
    r=project(np.array([1.,1.]),np.array([[1.,0.]]),C)
    assert r['positions'][0]==pytest.approx(-1.4);assert r['orthogonality_error']<1e-10

def test_duplicate_instruments_do_not_create_span():
    r=project([1.,1.],[[1.,0.],[2.,0.]],np.eye(2));assert r['span_rank']==1;assert r['residual_variance']==pytest.approx(1.)

def test_full_delta_span_does_not_remove_nonlinear_jump_loss():
    jumps=np.array([-.6,0,.7]);F=np.exp(jumps);claim=F**2
    linear=project([2.],[[1.]],[[1.]])
    assert linear['residual_variance']<1e-12
    actual=claim-1+linear['positions'][0]*(F-1)
    assert np.var(actual)>.01

def test_instrument_rates_and_expiry(m):
    h=default_instruments()[0];assert h.margin(m,0,np.zeros(9))>0;assert h.bounds(h.delivery)==(0.,0.)
    assert h.margin(m,h.delivery,np.zeros(9))==0
    r=default_instruments()[-1];assert r.delta(m,0,np.zeros(9))[6]>0

@pytest.mark.parametrize('field,value',[('trade_limit',-1),('direction','foo'),('multiplier',0)])
def test_bad_instrument(field,value):
    d=default_instruments()[0].__dict__.copy();d[field]=value
    with pytest.raises(ValueError):Instrument(**d)

def test_hardware_calibration_and_curve_link(m):
    c=example_case(paths=2).fleet;x=np.zeros(9)
    assert hardware_marks(m,0,x,c)==pytest.approx(c.hardware_values)
    x[2]=-.5;assert np.all(hardware_marks(m,0,x,c)<c.hardware_values)
    x=np.zeros(9);x[7]=1;assert np.all(hardware_marks(m,0,x,c)<c.hardware_values)

def test_curve_case_book_and_hash(tmp_path):
    c=example_case(paths=2);p=tmp_path/'case.json';c.save(p);r=CurveCase.load(p);assert c.fingerprint==r.fingerprint
    d=c.to_dict();d['existing_book']=None
    with pytest.raises(ValueError):CurveCase.from_dict(d)

@pytest.fixture(scope='module')
def controller_objects():
    c=example_case(paths=4);train=economy_tape(c,256,49);t=economy_tape(c,4,51);s=train_continuation(c,train)
    return c,t,s

@pytest.mark.parametrize('policy',['unhedged','static','adaptive'])
def test_controller_cash_and_replay(controller_objects,policy):
    c,t,s=controller_objects;a=run_controller(c,t,s,policy=policy,scenarios=32);b=run_controller(c,t,s,policy=policy,scenarios=32)
    assert np.array_equal(a.terminal_wealth,b.terminal_wealth);assert a.audit['maximum_component_error']<1e-5;assert a.audit['maximum_prefunding_violation']<1e-5
    if policy=='unhedged':assert np.max(abs(a.positions))==0

def test_tape_and_surface_replay(controller_objects,tmp_path):
    from compute_market.curves.portfolio import EconomyTape
    c,t,s=controller_objects;t.save(tmp_path/'t.npz');s.save(tmp_path/'s.npz');tt=EconomyTape.load(tmp_path/'t.npz');ss=ContinuationSurface.load(tmp_path/'s.npz')
    a=run_controller(c,t,s,scenarios=32);b=run_controller(c,tt,ss,scenarios=32)
    assert np.array_equal(a.cash,b.cash);assert np.array_equal(a.positions,b.positions)

def test_saved_conditional_draws_replay_without_resampling(controller_objects,tmp_path,monkeypatch):
    from compute_market.curves.control import ConditionalInnovations
    c,t,s=controller_objects
    draws=ConditionalInnovations.generate(c.model,c.fleet.steps,32,c.fleet.year_fraction,938)
    path=tmp_path/'conditional.npz';draws.save(path)
    expected=run_controller(c,t,s,scenarios=32,innovations=draws)
    def unexpected_sampling(*args,**kwargs):
        raise AssertionError('saved controller draws must be reused')
    monkeypatch.setattr(c.model,'step',unexpected_sampling)
    actual=run_controller(c,t,s,scenarios=32,innovations=ConditionalInnovations.load(path))
    for field in ('terminal_wealth','positions','posted_margin','cash','debt'):
        assert np.array_equal(getattr(expected,field),getattr(actual,field))

@pytest.mark.parametrize('field,value',[('model_sha256','different'),('step_years',.9),('seed',939)])
def test_conditional_draws_require_matching_inputs(controller_objects,field,value):
    from compute_market.curves.control import ConditionalInnovations
    c,t,s=controller_objects
    draws=ConditionalInnovations.generate(c.model,c.fleet.steps,32,c.fleet.year_fraction,938)
    setattr(draws,field,value)
    with pytest.raises(ValueError,match='conditional innovations must match'):
        run_controller(c,t,s,scenarios=32,innovations=draws)

def test_conditional_draw_checksum(controller_objects,tmp_path):
    from compute_market.curves.control import ConditionalInnovations
    c,t,s=controller_objects
    draws=ConditionalInnovations.generate(c.model,c.fleet.steps,32,c.fleet.year_fraction,938)
    path=tmp_path/'conditional.npz';draws.save(path)
    with np.load(path,allow_pickle=False) as saved:
        values=saved['values'].copy();metadata=saved['metadata']
    values[0,0,0]+=1
    np.savez_compressed(path,values=values,metadata=metadata)
    with pytest.raises(ValueError,match='checksum mismatch'):
        ConditionalInnovations.load(path)

@pytest.mark.parametrize('name',['long_end_repricing','margin_call','basis_break','liquidity_spiral','outage','benchmark_dislocation'])
def test_stress_propagation(controller_objects,name):
    c,t,s=controller_objects;u=stress_economy(c,t,name);assert u.name==name
    assert np.array_equal(t.curve.factors[:,:4],u.curve.factors[:,:4])
    r=run_controller(c,u,s,scenarios=32);assert np.isfinite(r.terminal_wealth).all();assert r.audit['maximum_component_error']<1e-5

def test_training_drift_not_future_tape(m):
    tape=m.simulate([0,.25,.5],1500,484,'P');fit,d=fit_physical_drift(m,tape)
    assert fit.fingerprint!=m.fingerprint
    assert fit.jump_intensity_q==m.jump_intensity_q
    assert max(abs(np.asarray(d['drift'])-m.physical_drift)/np.asarray(d['standard_error']))<6

def test_swing_adapted_exercise(g):
    p=train_swing(g,'G1',[0,.25,.5,.75,1.],2.3,2,paths=1000)
    a=evaluate_swing(g,p,paths=1000,seed=728);b=evaluate_swing(g,p,paths=1000,seed=728)
    assert a['value']>0;assert a['maximum_exercises']<=2;assert np.array_equal(a['payoffs'],b['payoffs'])

def test_curve_export_is_model_driven_and_fixed_horizon():
    c=example_case(paths=4);a=finite_case(c,3,2,111);b=finite_case(c,5,2,111)
    assert max(a.assumptions['dates'])==max(b.assumptions['dates'])==1.
    r=a.solve();assert r.status=='optimal';assert r.audit['independent_delivery_error']<1e-8
    assert a.assumptions['curve_model_sha256']==c.model.fingerprint
    assert r.provenance['compiled_model']['class']=='LP'

def test_new_contract_exact_quote_hedge():
    c=finite_case(example_case(paths=2),3,2,112);q=c.quote(price_tolerance=.2)
    assert q['pricing_certificate']['success'];assert np.isfinite(q['reservation']['amount'])


def test_constrained_projection_margin():
    from compute_market.curves.risk import constrained_project
    r=constrained_project([10.],[[1.]],[[1.]],[-20.],[0.],[2.],8.)
    assert r['positions'][0]==pytest.approx(-4.,abs=1e-7)
    assert r['residual_variance']==pytest.approx(36.,abs=1e-6)
    assert r['initial_margin']==pytest.approx(8.)


def test_constrained_projection_depth_and_validation():
    from compute_market.curves.risk import constrained_project
    r=constrained_project([10.],[[1.]],[[1.]],[-20.],[0.],[1.],20.,trade_limit=[3.])
    assert r['positions'][0]==pytest.approx(-3.)
    with pytest.raises(ValueError):constrained_project([1.],[[1.]],[[1.]],[-1.],[1.],[-1.],10.)
    with pytest.raises(ValueError):constrained_project([1.],[[1.]],[[1.]],[-1.],[1.],[1.],10.,trade_limit=[-1.])


def test_curve_cli_complete_case(tmp_path,capsys):
    from compute_market.cli import main
    import json
    f=tmp_path/'case.json'
    assert main(['curves','example','--output',str(f)])==0
    assert main(['curves','validate',str(f)])==0
    assert main(['curves','surface',str(f),'--output',str(tmp_path/'surface.json')])==0
    data=json.loads((tmp_path/'surface.json').read_text())
    assert len(data['calendar_dates'])==25
    assert data['futures']['G1'][0]==pytest.approx(2.5)
    assert main(['curves','validate',str(tmp_path/'missing')])==1


def test_curve_cli_price(tmp_path,capsys):
    from compute_market.cli import main
    f=tmp_path/'case.json';example_case(paths=4,steps=4).save(f)
    out=tmp_path/'quote.json'
    assert main(['curves','quote',str(f),'--branches','2','--seed','61','--output',str(out)])==0
    assert out.exists() and out.with_name('quote_finite_case.json').exists()


def test_curve_loan_interest_is_paid_through_terminal():
    c=example_case(paths=4,steps=4)
    tape=economy_tape(c,4,152);v=train_continuation(c,economy_tape(c,64,151))
    r=run_controller(c,tape,v,policy='unhedged',scenarios=32)
    expected=np.zeros(4);term=np.full(4,c.fleet.initial_term_debt)
    for u,t in enumerate(tape.curve.times[:-1]):
        term-=np.minimum(term,term if u+1>=c.fleet.term_maturity_step else c.fleet.term_amortization)
        rate=c.model.short_rate(t,tape.curve.factors[:,u])+c.model.value('funding',t,tape.curve.factors[:,u],t)
        expected+=c.fleet.year_fraction*(r.debt[:,u]*rate+term*c.fleet.fixed_term_coupon)
    assert not r.failed.any()
    assert np.allclose(-r.components[:,1],expected,atol=1e-7)


def test_curve_policy_has_no_future_state_access():
    c=example_case(paths=4,steps=4)
    tape=economy_tape(c,4,189);stress=stress_economy(c,tape,'long_end_repricing')
    v=train_continuation(c,economy_tape(c,64,188))
    a=run_controller(c,tape,v,policy='adaptive',scenarios=32)
    b=run_controller(c,stress,v,policy='adaptive',scenarios=32)
    # The intervention starts halfway through the tape. Earlier actions match.
    assert np.array_equal(a.positions[:,:2],b.positions[:,:2])
    assert np.array_equal(a.cash[:,:2],b.cash[:,:2])
