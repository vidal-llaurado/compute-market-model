"""Execution liquidity is an observable stochastic state, not a traded price curve."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class LiquiditySpec:
    mean_reversion: float = 2.0
    annual_volatility: float = .55
    jump_annual_rate: float = .25
    jump_mean: float = .5
    funding_factor: str = 'funding liquidity'
    funding_loading: float = .6
    market_downside_loading: float = .2
    spread_loading: float = .8
    depth_loading: float = .8
    margin_loading: float = .6
    outage_threshold: float = 3.5
    min_multiplier: float = .05
    max_multiplier: float = 20.
    def __post_init__(self):
        numbers=[v for k,v in asdict(self).items() if k!='funding_factor']
        if not np.isfinite(numbers).all() or min(self.mean_reversion,self.annual_volatility,
            self.jump_annual_rate,self.jump_mean,self.spread_loading,self.depth_loading,self.margin_loading)<0:
            raise ValueError('invalid liquidity law')
        if not 0 < self.min_multiplier <= 1 <= self.max_multiplier:
            raise ValueError('liquidity multiplier bounds must contain one')
    def multipliers(self,state):
        z=np.asarray(state,float)
        if not np.isfinite(z).all(): raise ValueError('nonfinite liquidity state')
        lo,hi=np.log(self.min_multiplier),np.log(self.max_multiplier)
        spread=np.exp(np.clip(self.spread_loading*z,lo,hi))
        depth=np.exp(np.clip(-self.depth_loading*z,lo,hi))
        depth=np.where(z>=self.outage_threshold,0.,depth)
        margin=np.exp(np.clip(self.margin_loading*z,lo,hi))
        return spread,depth,margin
    def simulate(self,model,tape,seed):
        """Exact OU idiosyncratic endpoint plus observed shared market loadings.

        Jump ages are integrated exactly at the decision grid. Shared funding and
        market factors induce wrong-way dependence without changing the Q curve law.
        """
        rng=np.random.default_rng(seed); X=tape.factors; n,N,k=X.shape
        residual=np.zeros((n,N)); state=np.zeros_like(residual)
        if self.funding_factor not in model.factors:
            raise ValueError('liquidity funding factor is not present in the market state')
        idx=model.factors.index(self.funding_factor)
        for u,dt in enumerate(np.diff(tape.times),1):
            decay=np.exp(-self.mean_reversion*dt)
            variance=dt if self.mean_reversion==0 else -np.expm1(-2*self.mean_reversion*dt)/(2*self.mean_reversion)
            residual[:,u]=residual[:,u-1]*decay+self.annual_volatility*np.sqrt(variance)*rng.normal(size=n)
            counts=rng.poisson(self.jump_annual_rate*dt,n)
            ids=np.repeat(np.arange(n),counts)
            if len(ids):
                ages=rng.uniform(0,dt,len(ids))
                marks=rng.exponential(self.jump_mean,len(ids)) if self.jump_mean else np.zeros(len(ids))
                np.add.at(residual[:,u],ids,marks*np.exp(-self.mean_reversion*ages))
        shared=0. if idx is None else self.funding_loading*X[:,:,idx]
        state=residual+shared+self.market_downside_loading*np.maximum(-X[:,:,0],0.)
        return state

def apply_liquidity(case,tape,spec:LiquiditySpec,seed=8321):
    """Apply the same observable liquidity shocks to all compared policies."""
    out=tape.copy(); state=spec.simulate(case.model,tape.curve,seed)
    spread,depth,margin=spec.multipliers(state)
    out.spread_multiplier*=spread;out.depth_multiplier*=depth;out.margin_multiplier*=margin
    return out,{'state':state,'spec':asdict(spec),'seed':seed,
                'spread_multiplier':spread,'depth_multiplier':depth,'margin_multiplier':margin}
