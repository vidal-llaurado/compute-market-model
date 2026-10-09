"""Bootstrap filtering primitives; simulator truth is not an admissible control input."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


def normalize_log_weights(log_weights):
    w=np.asarray(log_weights,float)
    if w.ndim != 1 or w.size == 0 or np.any(np.isnan(w)) or np.any(np.isposinf(w)) or not np.isfinite(w).any():
        raise ValueError("at least one finite log weight and no NaN/+inf required")
    return np.exp(w-logsumexp(w))


def systematic_resample(weights,rng: np.random.Generator):
    w=np.asarray(weights,float)
    if w.ndim != 1 or not w.size or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(),1,atol=1e-10,rtol=0):
        raise ValueError("normalized nonnegative particle weights required")
    positions=(rng.random()+np.arange(w.size))/w.size
    cumulative=np.cumsum(w); cumulative[-1]=1.
    return np.searchsorted(cumulative,positions,side="right")


def update_particles(particles,prior_weights,observation,observation_matrix,
                     observation_covariance,rng: np.random.Generator,resample_threshold: float=.5):
    x,w,y,h,r=map(lambda v:np.asarray(v,float),(particles,prior_weights,observation,observation_matrix,observation_covariance))
    if x.ndim != 2 or not x.shape[0] or w.shape != (x.shape[0],) or y.ndim != 1 or h.shape != (y.size,x.shape[1]) or r.shape != (y.size,y.size) or any(not np.isfinite(v).all() for v in (x,w,y,h,r)) or np.any(w < 0) or not np.isclose(w.sum(),1,atol=1e-10,rtol=0) or not 0 <= resample_threshold <= 1:
        raise ValueError("invalid particle observation update")
    if not np.allclose(r,r.T): raise ValueError("observation covariance must be symmetric")
    chol=np.linalg.cholesky(r)
    residual=y[None,:]-x@h.T
    z=np.linalg.solve(chol,residual.T)
    logw=np.full_like(w,-np.inf); np.log(w,out=logw,where=w>0)
    logw-=.5*np.sum(z*z,axis=0)
    posterior=normalize_log_weights(logw)
    mean=posterior@x
    centered=x-mean
    covariance=(centered.T*posterior)@centered
    ess=float(1/(posterior@posterior))
    indices=np.arange(x.shape[0]); resampled=ess < resample_threshold*x.shape[0]
    if resampled:
        indices=systematic_resample(posterior,rng)
        x=x[indices].copy(); posterior=np.full(len(w),1/len(w))
    return {"particles":x,"weights":posterior,"posterior_mean":mean,
            "posterior_covariance":covariance,"effective_sample_size":ess,
            "resampled":resampled,"ancestor_indices":indices}
