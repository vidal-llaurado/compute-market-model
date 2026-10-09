"""Sampling error, paired comparisons, tail diagnostics and importance estimates.

Monte Carlo intervals condition on the model. Bootstrap resamples whole
independent paths; neither establishes empirical calibration accuracy."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, t as student_t
from scipy.special import logsumexp
from .risk import expected_shortfall


def _sample(x):
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('at least two finite independent path observations required')
    return x


def mean_interval(x, confidence=.95):
    x=_sample(x)
    if not 0<confidence<1: raise ValueError('confidence in (0,1) required')
    mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    z=float(student_t.ppf((1+confidence)/2,len(x)-1))
    return {'mean':mean,'standard_error':se,'lower':mean-z*se,'upper':mean+z*se,
            'confidence':confidence,'n':len(x),'method':'Student-t mean interval; asymptotic for nonnormal paths'}


def wilson_interval(events, confidence=.95):
    x=np.asarray(events)
    if x.ndim!=1 or not len(x) or not np.all((x==0)|(x==1)) or not 0<confidence<1:
        raise ValueError('nonempty Bernoulli path events and confidence in (0,1) required')
    n=len(x);k=int(x.sum());p=k/n;z=float(norm.ppf((1+confidence)/2));den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'probability':p,'lower':max(0.,center-half),'upper':min(1.,center+half),
            'events':k,'n':n,'confidence':confidence,'method':'Wilson; ordinary IID Monte Carlo only'}


def tail_summary(losses, alpha=.95, bootstrap=200, seed=41):
    x=_sample(losses)
    if not 0<=alpha<1 or bootstrap<0: raise ValueError('invalid alpha or bootstrap count')
    n=len(x);p=np.full(n,1/n);es=expected_shortfall(x,p,alpha)
    quantile=float(np.quantile(x,alpha,method='inverted_cdf'))
    interval=None
    if bootstrap:
        rng=np.random.default_rng(seed)
        esb=np.array([expected_shortfall(x[rng.integers(0,n,size=n)],p,alpha) for _ in range(bootstrap)])
        interval={'lower':float(np.quantile(esb,.025)),'upper':float(np.quantile(esb,.975)),
                  'confidence':.95,'method':'whole-path percentile bootstrap; not a finite-sample coverage guarantee'}
    return {'value_at_risk':quantile,'expected_shortfall':es,'alpha':alpha,'n':n,
            'effective_tail_count':n*(1-alpha),'bootstrap_interval':interval,
            'warning':'tail sample too small for reliable precision' if n*(1-alpha)<50 else None}


def paired_comparison(a,b):
    a=_sample(a);b=_sample(b)
    if a.shape!=b.shape: raise ValueError('paired common-random-number arrays must match')
    result=mean_interval(b-a)
    result['interpretation']='second policy minus first policy on the same path tape'
    return result


def importance_event(events, log_likelihood):
    """Unbiased P-event estimator E_Q[w 1_A], not self-normalized. No Wilson CI."""
    y=_sample(events);logw=_sample(log_likelihood)
    if y.shape!=logw.shape or not np.all((y==0)|(y==1)):
        raise ValueError('matching Bernoulli outcomes and log-likelihoods required')
    if np.max(logw)>700: raise ValueError('importance weights overflow; proposal is unusable')
    w=np.exp(logw);weighted=y*w
    answer=mean_interval(weighted)
    normalized=np.exp(logw-logsumexp(logw));ess=1/float(normalized@normalized)
    answer.update(estimate=float(weighted.mean()),weight_mean=float(w.mean()),
                  effective_sample_size=ess,maximum_normalized_weight=float(normalized.max()),
                  method='unnormalized likelihood-ratio estimate with asymptotic mean interval',
                  warning='poor overlap: effective sample size below 10% of paths' if ess<.1*len(y) else None)
    return answer


def variance_decomposition(samples):
    """Balanced independent process replicates for each fixed parameter set.

    Correct between-parameter variance for finite inner-MC noise. Negative raw
    correction is reported, not silently interpreted as zero epistemic variance.
    Parameter sets are scenario assumptions unless a posterior is supplied.
    """
    a=np.asarray(samples,float)
    if a.ndim!=2 or min(a.shape)<2 or not np.isfinite(a).all():
        raise ValueError('finite [parameter, independent process replicate] matrix required')
    within=float(a.var(axis=1,ddof=1).mean());between_raw=float(a.mean(axis=1).var(ddof=1))
    corrected=between_raw-within/a.shape[1]
    return {'mean_conditional_process_variance':within,'variance_of_parameter_means_raw':between_raw,
            'parameter_variance_mc_corrected_raw':corrected,
            'parameter_variance_nonnegative_estimate':max(0.,corrected),
            'noise_correction':within/a.shape[1],
            'interpretation':'parameter-set sensitivity, not posterior uncertainty unless sets are posterior draws'}


def simulation_summary(result, bootstrap=200, seed=41):
    initial_note='Terminal wealth is signed economic value, not limited-liability equity.'
    return {'policy':result.policy,'wealth':mean_interval(result.terminal_wealth),
            'loss_tail':tail_summary(-result.terminal_wealth,.95,bootstrap,seed),
            'financing_failure':wilson_interval(result.financing_failure),
            'delivery_breach':wilson_interval(result.delivery_breach),
            'mean_preterminal_forced_blocks_sold':float(result.total_asset_sales.mean()),
            'mean_shortfall_gpu_hours':float(result.total_shortfall_gpu_hours.mean()),
            'mean_hedge_cash_pnl':float(result.hedge_cash_pnl.mean()),
            'mean_creditor_loss_proxy':float(result.creditor_loss.mean()),
            'mean_equity_recovery_proxy':float(result.equity_recovery.mean()),
            'cash_headroom_quantiles':np.quantile(result.minimum_cash_headroom,[0,.01,.05,.5]).tolist(),
            'audit':result.audit,'note':initial_note}
