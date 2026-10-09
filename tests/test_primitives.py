import numpy as np
import pytest
from compute_market.market import (RidgeLogPrice,fit_delivery_curve,state_price_bounds,
                                  minimum_variance_hedge,logit_choice)
from compute_market.stochastic import HybridFactors,structural_log_price
from compute_market.filtering import normalize_log_weights,systematic_resample,update_particles
from compute_market.allocation import ClusterRequest,allocate
from compute_market.contracts import average_cap,margin_floor,swing_value
from compute_market.tree import ScenarioTree,regular_tree


def test_log_price_fit():
    x=np.column_stack([np.ones(50),np.linspace(-1,1,50)])
    y=np.exp(x@np.array([1,.5]))
    m=RidgeLogPrice.fit(x,y,ridge=0)
    assert m.coefficients==pytest.approx([1,.5])
    assert m.median_price(x)==pytest.approx(y)


def test_forward_strip_identification():
    w=np.array([[1,0,0],[.5,.5,0],[1/3]*3]);f=np.array([2.5,2.35,2.2])
    result=fit_delivery_curve(w,w@f,smoothing=0)
    assert result['forwards']==pytest.approx(f)
    assert result['unidentified_dimensions']==0


def test_sparse_curve_reports_unidentified():
    r=fit_delivery_curve([[.5,.5]],[2],smoothing=1)
    assert r['unidentified_dimensions']==1


def test_financial_state_price_bounds():
    r=state_price_bounds(np.array([[1],[3]]),[2],[1,9])
    assert r['consistent']
    assert r['lower']==pytest.approx(5)
    assert r['upper']==pytest.approx(5)
    incomplete=state_price_bounds(np.empty((2,0)),[],[1,9])
    assert incomplete['lower']==1
    assert incomplete['upper']==9


def test_inconsistent_market_marks():
    assert not state_price_bounds([[1],[1]],[2],[0,1])['consistent']


def test_minimum_variance_sign():
    h=np.arange(10.)[:,None]
    r=minimum_variance_hedge(-2*h[:,0],h)
    assert r['long_positions']==pytest.approx([-2])
    assert r['residual_variance'] < 1e-20


def test_choice_eligibility_outside_and_price_monotonicity():
    a=logit_choice([1,2],[1,1],[0,0],1,[True,False])
    b=logit_choice([2,2],[1,1],[0,0],1,[True,False])
    assert a['products'][1]==0
    assert a['products'][0]>b['products'][0]
    assert a['outside']+sum(a['products'])==pytest.approx(1)


def model(diffusion=0):
    return HybridFactors(mean=[[0]],kappa=[[1]],diffusion=[[[diffusion]]],
              generator=[[0]],jump_intensity=np.zeros((1,0)),jump_mean=np.zeros((0,1)),
              jump_loading=np.zeros((0,1,1)))


def test_ou_exact_deterministic():
    x,z,events=model().step([1],0,2,np.random.default_rng(1))
    assert x[0]==pytest.approx(np.exp(-2));assert z==0;assert events==[]


def test_ou_moments():
    mean,cov=model(2).segment_moments([1],0,1)
    assert mean[0]==pytest.approx(np.exp(-1))
    assert cov[0,0]==pytest.approx(2*(1-np.exp(-2)))


def test_zero_mean_reversion_limit():
    m=model(2);m.kappa[:]=0
    mean,cov=m.segment_moments([1],0,2)
    assert mean[0]==1;assert cov[0,0]==8


def test_shared_jump_and_seed_reproducibility():
    m=HybridFactors([[0,0]],[[0,0]],[np.zeros((2,2))],[[0]],[[20]],[[-1,-2]],[np.zeros((2,2))])
    a=m.step([0,0],0,1,np.random.default_rng(11))
    b=m.step([0,0],0,1,np.random.default_rng(11))
    assert a[0]==pytest.approx(b[0]);assert len(a[2])>0
    assert a[0][1]==pytest.approx(2*a[0][0])


def test_regime_generator_validation():
    with pytest.raises(ValueError):HybridFactors([[0]],[[0]],[[[0]]],[[1]],[[0]],[[0]],[[[0]]])


def test_scarcity_increases_price_no_zero_capacity_quote():
    a=structural_log_price([0],[[0]],[0],[1],[2])
    b=structural_log_price([0],[[0]],[0],[1],[1])
    assert b[0]>a[0]
    with pytest.raises(ValueError):structural_log_price([0],[[0]],[0],[1],[0])


def test_log_weights_numerical_stability():
    p=normalize_log_weights([-10000,-10001])
    assert p.sum()==pytest.approx(1);assert p[0]>p[1]
    with pytest.raises(ValueError):normalize_log_weights([-np.inf,-np.inf])


def test_resampling_indices():
    idx=systematic_resample([0,1,0],np.random.default_rng(4))
    assert np.array_equal(idx,[1,1,1])


def test_particle_update():
    r=update_particles([[-1],[0],[1]],[1/3]*3,[1],[[1]],[[.1]],np.random.default_rng(1),0)
    assert r['posterior_mean'][0]>.9
    assert not r['resampled']
    assert r['weights'].sum()==pytest.approx(1)


def test_disconnected_gpus_not_a_cluster():
    req=ClusterRequest('job',8,0,2,(0,1),10)
    assert allocate([[4,4],[4,4]],[req])['status']=='infeasible'


def test_no_migration_between_intervals():
    req=ClusterRequest('job',8,0,2,(0,1),10)
    assert allocate([[8,0],[0,8]],[req])['status']=='infeasible'


def test_allocation_selects_highest_contribution():
    a=ClusterRequest('a',8,0,1,(0,),10,False)
    b=ClusterRequest('b',8,0,1,(0,),20,False)
    r=allocate([[8]],[a,b])
    assert r['assignments']=={'b':0}


def test_average_cap_jensen():
    prices=np.array([[0,10],[3,4],[2,2]])
    cap=average_cap(prices,[.5,.5],1,3)
    separate=np.mean(np.maximum(prices-3,0),axis=1)
    assert np.all(cap<=separate)
    assert cap[0]==2


def test_margin_floor_units():
    assert margin_floor([[2,3]],[[100,100]],[.5,.5],.01,100,2)[0]==pytest.approx(50)


def test_swing_nonanticipative_buyer():
    # First choice is worth 5 now; next period is 0 or 10, so value is 5, not 7.5.
    t=ScenarioTree([-1,0,1,1],[1,1,.5,.5],[0,1,2,2])
    r=swing_value(t,[0,5,0,10],1,0,1)
    assert r['value']==pytest.approx(5)
    assert r['value']<7.5


def test_swing_minimum_volume():
    t=ScenarioTree([-1,0],[1,1],[0,1])
    assert swing_value(t,[0,-2],1,1,1)['value']==-2
    assert not swing_value(t,[0,-2],1,2,2)['feasible']
