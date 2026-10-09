"""Dated funded claims and margined futures on a common curve state.

Marks are ex-distribution. settlement() pays cash events; inventory trades pay
the current funded mark. Transferable forwards keep their original strike,
and swaps and caps keep their coupon dates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .option_pricing import option_price
from .risk import Instrument

@dataclass(frozen=True)
class CurveClaim:
    name: str
    market: str
    delivery: float
    multiplier: float
    position_limit: float
    trade_limit: float
    impact_depth: float
    spread: float
    impact_slope: float
    kind: str = 'option'
    direction: str = 'long'
    strike: float = 2.3
    expiry: float | None = None
    put: bool = True
    payment_dates: tuple[float,...] = ()
    weights: tuple[float,...] = ()
    minimum_margin: float = 0.
    margin_sigma: float = 0.
    margin_days: float = 10.
    liquidity_charge_annual: float = 0.
    premium_charge_annual: float = 0.
    counterparty: str = 'cleared'
    recovery: float = .4
    def __post_init__(self):
        if self.kind not in ('option','forward','swap','cap'):raise ValueError('unknown funded claim kind')
        vals=[self.delivery,self.multiplier,self.position_limit,self.trade_limit,self.impact_depth,
              self.spread,self.impact_slope,self.minimum_margin,self.margin_sigma,self.margin_days,
              self.liquidity_charge_annual,self.premium_charge_annual,self.recovery,self.strike]
        if not np.isfinite(vals).all() or min(vals[:-1])<0 or self.multiplier<=0 or not 0<=self.recovery<=1:
            raise ValueError('invalid claim economics')
        if self.direction not in ('both','long','short','none'):raise ValueError('invalid signed mandate')
        if self.kind in ('option','cap') and self.strike<=0:raise ValueError('positive option strike required')
        if self.expiry is not None and not 0<=self.expiry<=self.delivery:raise ValueError('option expiry must precede delivery')
        if self.kind in ('swap','cap'):
            d=np.asarray(self.payment_dates,float);w=np.asarray(self.weights,float)
            if (d.ndim!=1 or not len(d) or w.shape!=d.shape or not np.isfinite(d).all()
                    or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0 or d[0]<=0
                    or np.any(np.diff(d)<=0) or abs(d[-1]-self.delivery)>1e-9):
                raise ValueError('positive weighted, increasing coupon dates ending at delivery required')
    @property
    def maturity(self):return self.delivery if self.expiry is None or self.kind!='option' else self.expiry
    @property
    def funded(self):return True
    @property
    def event_dates(self):
        return tuple(self.payment_dates) if self.kind in ('swap','cap') else (self.maturity,)
    def bounds(self,t):
        if t>=self.maturity-1e-10:return 0.,0.
        return (0. if self.direction in ('long','none') else -self.position_limit,
                0. if self.direction in ('short','none') else self.position_limit)
    def mark(self,model,t,x):
        """Ex-dividend price per multiplier unit."""
        x=np.asarray(x,float)
        if t>=self.maturity-1e-10:return np.zeros(x.shape[:-1])
        if self.kind=='option':return option_price(model,self.market,t,x,self.maturity,self.delivery,self.strike,self.put)
        if self.kind=='forward':return model.discount(t,x,self.delivery)*(model.forward(self.market,t,x,self.delivery)-self.strike)
        v=np.zeros(x.shape[:-1])
        for T,w in zip(self.payment_dates,self.weights):
            if T>t+1e-10:
                if self.kind=='swap':v+=w*model.discount(t,x,T)*(model.forward(self.market,t,x,T)-self.strike)
                else:v+=w*option_price(model,self.market,t,x,T,T,self.strike,False)
        return v
    def settlement(self,model,t,x):
        x=np.asarray(x,float);value=np.zeros(x.shape[:-1])
        dates=self.payment_dates if self.kind in ('swap','cap') else (self.maturity,)
        weights=self.weights if self.kind in ('swap','cap') else (1.,)
        for T,w in zip(dates,weights):
            if abs(T-t)<1e-9:
                delivery=self.delivery if self.kind=='option' else T
                spot=model.value(self.market,t,x,delivery)
                if self.kind=='option':pay=np.maximum(self.strike-spot if self.put else spot-self.strike,0.)
                elif self.kind=='cap':pay=np.maximum(spot-self.strike,0.)
                else:pay=spot-self.strike
                value+=w*pay
        return value
    def delta(self,model,t,x,bump=1e-5):
        x=np.asarray(x,float);E=np.eye(model.dimension)*bump
        return self.multiplier*np.stack([(self.mark(model,t,x+e)-self.mark(model,t,x-e))/(2*bump) for e in E],axis=-1)
    def gamma(self,model,t,x,bump=1e-4):
        x=np.asarray(x,float)
        if x.ndim!=1:raise ValueError('one state required for a full gamma matrix')
        from .claims import finite_difference_greeks
        return finite_difference_greeks(lambda z:self.multiplier*self.mark(model,t,z),x,bump)['gamma']
    def margin(self,model,t,x):
        # A paid long option has no seller performance obligation. Short-option
        # permissions require explicit nonzero margin parameters from the venue.
        if t>=self.maturity-1e-10:return np.zeros(np.asarray(x).shape[:-1])
        if self.direction in ('short','both') and self.kind in ('option','cap') and self.margin_sigma==0 and self.minimum_margin==0:
            raise ValueError('short nonlinear claims require an explicit margin rule')
        if self.margin_sigma==0:return np.full(np.asarray(x).shape[:-1],self.minimum_margin)
        d=self.delta(model,t,x);_,_,C=model.transition_moments(min(self.margin_days/365,self.maturity-t),'P')
        return np.maximum(self.minimum_margin,self.margin_sigma*np.sqrt(np.maximum(np.einsum('...i,ij,...j->...',d,C,d),0.)))
    def closeout(self,model,t,x,quantity):
        v=quantity*self.multiplier*(self.mark(model,t,x)+self.settlement(model,t,x))
        return np.where(v>0,self.recovery*v,v)


def instrument_from_dict(data):
    return CurveClaim(**data) if 'kind' in data else Instrument(**data)


def diversified_instruments(model,horizon=1.):
    from .risk import default_instruments
    instruments=default_instruments(horizon)
    root=np.zeros(model.dimension)
    for market in ('G1','G2'):
        strike=float(model.value(market,0,root,horizon))
        instruments.append(CurveClaim(f'{market} downside put',market,horizon,1000.,250.,100.,35.,1.5,.06,
            kind='option',strike=strike,expiry=horizon,direction='long',premium_charge_annual=.03))
    T=horizon;K=float(model.forward('power_US',0,root,T))
    instruments.append(CurveClaim('power fixed-strike forward','power_US',T,25.,300.,100.,40.,1.5,.05,
        kind='forward',strike=K,direction='long',minimum_margin=50.,counterparty='supplier',recovery=.4))
    dates=tuple(float(v) for v in np.linspace(horizon/4,horizon,4))
    instruments.append(CurveClaim('compute periodic cap','G1',T,1000.,120.,60.,20.,2.,.08,
        kind='cap',strike=float(model.initial('G1',horizon)),direction='long',payment_dates=dates,
        weights=tuple([.25]*4),premium_charge_annual=.03))
    return instruments
