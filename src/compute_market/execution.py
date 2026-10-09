"""Observable execution mandates and joint dispatch/financing coefficients."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from .risk_execution import INSTRUMENTS, LinearControls

@dataclass(frozen=True)
class ExecutionMandate:
    directions: dict[str,str]=field(default_factory=lambda:{
        'G1 future':'short','G2 future':'short','power future':'long',
        'rate future':'long','FX future':'long','G1 put':'long'})
    carry_convention: str='adverse_only'
    margin_opportunity_annual: dict[str,float]=field(default_factory=lambda:{name:0. for name in INSTRUMENTS})
    premium_opportunity_annual: dict[str,float]=field(default_factory=lambda:{name:0. for name in INSTRUMENTS})
    factor_names: tuple[str,...]=()
    factor_loadings: tuple[tuple[float,...],...]=()
    factor_lower: tuple[float,...]=()
    factor_upper: tuple[float,...]=()

    def __post_init__(self):
        if set(self.directions)!=set(INSTRUMENTS) or any(v not in ('long','short','both','none') for v in self.directions.values()):
            raise ValueError('each named instrument requires an explicit signed mandate')
        if self.carry_convention not in ('adverse_only','centered','total_return','symmetric'):raise ValueError('invalid carry mandate')
        for mapping in (self.margin_opportunity_annual,self.premium_opportunity_annual):
            if set(mapping)!=set(INSTRUMENTS) or any(not np.isfinite(x) or x<0 for x in mapping.values()):
                raise ValueError('annual cash opportunity rates must cover all instruments')
        n=len(self.factor_names)
        if n:
            a=np.asarray(self.factor_loadings,float);lo=np.asarray(self.factor_lower,float);hi=np.asarray(self.factor_upper,float)
            if len(set(self.factor_names))!=n or a.shape!=(n,len(INSTRUMENTS)) or lo.shape!=(n,) or hi.shape!=(n,) or not all(np.isfinite(x).all() for x in (a,lo,hi)) or np.any(lo>hi):
                raise ValueError('invalid factor mandate')
        elif any((self.factor_loadings,self.factor_lower,self.factor_upper)):raise ValueError('name the constrained factors')

    def bounds(self,limits):
        lim=np.asarray(limits,float)
        if lim.shape!=(len(INSTRUMENTS),) or np.any(lim<0) or not np.isfinite(lim).all():raise ValueError('finite instrument limits required')
        lower=-lim.copy();upper=lim.copy()
        for k,name in enumerate(INSTRUMENTS):
            direction=self.directions[name]
            if direction in ('long','none'):lower[k]=0.
            if direction in ('short','none'):upper[k]=0.
        return lower,upper

    def local_arguments(self,year_fraction):
        if year_fraction<=0:raise ValueError('positive accrual fraction required')
        return {'carry_convention':self.carry_convention,
            'funding_rate':np.array([self.margin_opportunity_annual[n] for n in INSTRUMENTS])*year_fraction,
            'premium_liquidity_rate':np.array([self.premium_opportunity_annual[n] for n in INSTRUMENTS])*year_fraction,
            **({'factor_loadings':self.factor_loadings,'exposure_lower':self.factor_lower,'exposure_upper':self.factor_upper} if self.factor_names else {})}

    def to_dict(self):return asdict(self)


def dispatch_controls(config,spot_limit,unit_cost,spot_price,forecast_rate,remaining,
                      current_debt,draw_available,repay_limit,current_reference_rate=None):
    """Joint current spot dispatch and incremental financing after incoming cures.

    Debt principal cancels between cash received and the terminal liability.
    Scenario loss retains incremental interest and transaction fees. Spot cash
    costs must be funded before its closing revenue is received.
    """
    gh=np.asarray(config.gpus_per_block)*config.hours_per_step
    cost=gh*np.asarray(unit_cost);profit=gh*(np.asarray(spot_price)-unit_cost)
    j=len(gh);s=len(forecast_rate)
    first_rate=np.asarray(forecast_rate) if current_reference_rate is None else current_reference_rate
    interest=((np.maximum(first_rate,-.02)+config.revolver_spread)
        +max(0,remaining-1)*(np.maximum(forecast_rate,-.02)+config.revolver_spread))*config.year_fraction
    D=np.column_stack([-np.broadcast_to(profit,(s,j)),interest+config.debt_fee,-interest+config.debt_fee])
    return LinearControls([f'spot:{a}' for a in range(j)]+['draw','repay'],np.zeros(j+2),
        np.r_[np.maximum(spot_limit,0),max(0,draw_available),max(0,min(current_debt,repay_limit))],
        np.r_[cost,-(1-config.debt_fee),1+config.debt_fee],D)
