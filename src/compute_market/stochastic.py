"""Regime-switching OU factors with shared compound-Poisson shocks.

Diffusion segments and event times are exact for the time-homogeneous law,
up to floating-point error. State-dependent intensities require another method."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HybridFactors:
    mean: np.ndarray                    # [regime, factor]
    kappa: np.ndarray                   # [regime, factor], nonnegative
    diffusion: np.ndarray               # [regime, factor, factor], loading matrix
    generator: np.ndarray               # [regime, regime], CTMC generator
    jump_intensity: np.ndarray          # [regime, event]
    jump_mean: np.ndarray               # [event, factor]
    jump_loading: np.ndarray            # [event, factor, factor]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            setattr(self,name,np.asarray(getattr(self,name),float))
        if self.mean.ndim != 2:
            raise ValueError("mean must be [regimes, factors]")
        r,d=self.mean.shape
        if r < 1 or d < 1 or self.jump_intensity.ndim != 2:
            raise ValueError("positive state dimension required")
        e=self.jump_intensity.shape[1]
        shapes={"kappa":(r,d),"diffusion":(r,d,d),"generator":(r,r),
                "jump_intensity":(r,e),"jump_mean":(e,d),"jump_loading":(e,d,d)}
        for name,shape in shapes.items():
            a=getattr(self,name)
            if a.size == 0 and np.prod(shape) == 0:
                a=a.reshape(shape); setattr(self,name,a)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"invalid {name}")
        if not np.isfinite(self.mean).all() or np.any(self.kappa < 0) or np.any(self.jump_intensity < 0):
            raise ValueError("finite parameters, nonnegative kappa and intensities required")
        off=self.generator.copy(); np.fill_diagonal(off,0)
        if np.any(off < 0) or not np.allclose(self.generator.sum(axis=1),0,atol=1e-10,rtol=0):
            raise ValueError("invalid CTMC generator")

    def segment_moments(self, x, regime: int, dt: float):
        x=np.asarray(x,float)
        r,d=self.mean.shape
        if x.shape != (d,) or not np.isfinite(x).all() or not 0 <= regime < r or dt < 0 or not np.isfinite(dt):
            raise ValueError("invalid factor state, regime or duration")
        decay=np.exp(-self.kappa[regime]*dt)
        mean=self.mean[regime]+decay*(x-self.mean[regime])
        rates=self.kappa[regime][:,None]+self.kappa[regime][None,:]
        integration=np.full_like(rates,dt)
        np.divide(-np.expm1(-rates*dt),rates,out=integration,where=rates != 0)
        covariance=(self.diffusion[regime]@self.diffusion[regime].T)*integration
        return mean,covariance

    def _segment(self,x,regime,dt,rng):
        mean,cov=self.segment_moments(x,regime,dt)
        values,vectors=np.linalg.eigh((cov+cov.T)/2)
        if values.min(initial=0) < -1e-9:
            raise ValueError("OU covariance is not positive semidefinite")
        return mean+vectors@(np.sqrt(np.maximum(values,0))*rng.standard_normal(len(mean)))

    def step(self,x,regime: int,dt: float,rng: np.random.Generator,max_events: int=10000):
        self.segment_moments(x,regime,dt)  # validation
        current=np.array(x,float,copy=True); remaining=float(dt); events=[]
        while remaining > 0:
            transitions=self.generator[regime].copy(); transitions[regime]=0
            intensities=np.r_[transitions,self.jump_intensity[regime]]
            total=float(intensities.sum())
            waiting=float(rng.exponential(1/total)) if total > 0 else np.inf
            duration=min(waiting,remaining)
            current=self._segment(current,regime,duration,rng)
            remaining-=duration
            if waiting >= duration and remaining <= 0: break
            if len(events) >= max_events:
                raise RuntimeError("event limit exceeded; no silent event truncation")
            event=int(rng.choice(len(intensities),p=intensities/total))
            if event < self.mean.shape[0]:
                old=regime; regime=event
                events.append({"time":dt-remaining,"type":"regime","from":old,"to":regime})
            else:
                e=event-self.mean.shape[0]
                mark=self.jump_mean[e]+self.jump_loading[e]@rng.standard_normal(self.mean.shape[1])
                current+=mark
                events.append({"time":dt-remaining,"type":"jump","event":e,"mark":mark.tolist()})
        return current,regime,events


def structural_log_price(intercept, loadings, factors, desired_capacity, usable_capacity,
                         scarcity_slope: float=1.) -> np.ndarray:
    """Positive reduced-form marks for products with nonzero capacity.

    These marks imply neither guaranteed delivery nor an executable offer."""
    b,l,x,d,c=map(lambda a:np.asarray(a,float),(intercept,loadings,factors,desired_capacity,usable_capacity))
    if b.ndim != 1 or l.shape != (b.size,x.size) or x.ndim != 1 or d.shape != b.shape or c.shape != b.shape or any(not np.isfinite(a).all() for a in (b,l,x,d,c)) or np.any(d <= 0) or np.any(c <= 0) or scarcity_slope < 0:
        raise ValueError("finite compatible inputs and strictly positive supply/demand required")
    log_ratio=np.log(d/c)
    scarcity=scarcity_slope*np.logaddexp(0,log_ratio)
    return b+l@x+scarcity
