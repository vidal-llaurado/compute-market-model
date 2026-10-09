"""Financial contracts, node quotes and bilateral collateral terms.

Marks include the contract multiplier and use reporting currency. Futures
marks determine variation settlement. Cash and OTC claims trade at their
funded marks; OTC claims also belong to a collateral netting set.
New zero-value forwards require a new dated vintage."""
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import json
import numpy as np
from .schema import array
from .tree import ScenarioTree

@dataclass
class NettingSet:
    name: str
    posting_threshold: float = 0.0
    receiving_threshold: float = 0.0
    recovery: float = 0.4
    def __post_init__(self):
        if not self.name or not np.isfinite([self.posting_threshold,self.receiving_threshold,self.recovery]).all():
            raise ValueError('finite named CSA required')
        if min(self.posting_threshold,self.receiving_threshold)<0 or not 0<=self.recovery<=1:
            raise ValueError('nonnegative thresholds and recovery in [0,1] required')

@dataclass
class HedgeMarket:
    names: list[str]
    kinds: list[str]
    multiplier: np.ndarray                 # K: units of underlying per position unit
    mark: np.ndarray                       # N,K: pre-trade ex-distribution marks
    distribution: np.ndarray               # N,K: cash per inherited long unit, before multiplier
    lower: np.ndarray                      # N,K: signed post-decision position bounds
    upper: np.ndarray
    trade_limit: np.ndarray                # N,K: gross voluntary turnover
    half_spread: np.ndarray                # N,K: reporting cash per position unit traded
    impact_slope: np.ndarray               # N,K: extra cash/unit past first depth tier
    impact_depth: np.ndarray               # N,K: size of first execution tier
    im_per_unit: np.ndarray                # N,K: cash IM per absolute position
    survives: np.ndarray                  # N,K: 1 if inherited claim continues after incoming settlement
    groups: list[str]                      # K: CSA name for OTC; empty otherwise
    netting_sets: list[NettingSet]
    default: np.ndarray                    # N,G: event on this incoming edge, not persistent status
    settlement_recovery: np.ndarray       # N,K: recovery of positive incoming futures/cash payments
    im_return: np.ndarray                  # N,K: fraction of inherited IM available on this edge
    initial_position: np.ndarray           # K: positions immediately before root
    initial_mark: np.ndarray               # K: previous futures settlement mark / previous asset mark
    initial_im: np.ndarray                 # K: already posted cash, excluded from portfolio cash
    initial_posted_vm: np.ndarray          # G: already posted segregated cash
    initial_received_vm: np.ndarray        # G: received segregated collateral, not spendable
    factor_names: list[str]
    factor_loadings: np.ndarray            # N,R,K: dollar sensitivity per position unit
    exposure_lower: np.ndarray             # N,R: optional hedging mandate on net hedge sensitivities
    exposure_upper: np.ndarray
    schema_version: str = '3.0'

    def validate(self, tree: ScenarioTree):
        n,k,g,r=tree.size,len(self.names),len(self.netting_sets),len(self.factor_names)
        if k==0 or len(set(self.names))!=k: raise ValueError('nonempty unique instrument names required')
        if len(self.kinds)!=k or any(v not in ('futures','cash','otc') for v in self.kinds):
            raise ValueError('instrument kinds: futures, cash, otc')
        if len(self.groups)!=k: raise ValueError('one netting identifier per instrument required')
        self.netting_sets=[NettingSet(**v) if isinstance(v,dict) else v for v in self.netting_sets]
        gn=[v.name for v in self.netting_sets]
        if len(set(gn))!=g: raise ValueError('unique netting sets required')
        for kind,group in zip(self.kinds,self.groups):
            if (kind=='otc' and group not in gn) or (kind!='otc' and group!=''):
                raise ValueError('OTC claims must map to a CSA; other claims cannot use that CSA')
        for name in ('mark','distribution','lower','upper','trade_limit','half_spread','impact_slope',
                     'impact_depth','im_per_unit','survives','settlement_recovery','im_return'):
            signed=name in ('mark','distribution','lower','upper')
            setattr(self,name,array(getattr(self,name),(n,k),name,lower=None if signed else 0))
        for name in ('multiplier','initial_position','initial_mark','initial_im'):
            setattr(self,name,array(getattr(self,name),(k,),name,lower=None if name in ('initial_position','initial_mark') else 0))
        for name in ('initial_posted_vm','initial_received_vm'):
            setattr(self,name,array(getattr(self,name),(g,),name,lower=0))
        self.default=array(self.default,(n,g),'default',lower=0,upper=1)
        self.factor_loadings=array(self.factor_loadings,(n,r,k),'factor_loadings')
        self.exposure_lower=array(self.exposure_lower,(n,r),'exposure_lower')
        self.exposure_upper=array(self.exposure_upper,(n,r),'exposure_upper')
        if len(set(self.factor_names))!=r or np.any(self.exposure_lower>self.exposure_upper):
            raise ValueError('invalid factor mandate')
        if np.any(self.lower>self.upper) or np.any(self.multiplier<=0): raise ValueError('invalid position bounds or multipliers')
        for a in (self.survives,self.default):
            if np.any((a!=0)&(a!=1)): raise ValueError('survival and default flags must be binary')
        if np.any(self.settlement_recovery>1) or np.any(self.im_return>1): raise ValueError('recovery fractions <=1 required')
        for node in range(n):
            for gi,group in enumerate(gn):
                ids=[i for i,v in enumerate(self.groups) if v==group]
                if self.default[node,gi]:
                    if np.any(self.survives[node,ids]!=0) or np.any(self.lower[node,ids]!=0) or np.any(self.upper[node,ids]!=0):
                        raise ValueError('defaulted claims must terminate and cannot reopen')
                    for desc in range(node+1,n):
                        if node in tree.path(desc):
                            if np.any(self.lower[desc,ids]!=0) or np.any(self.upper[desc,ids]!=0) or np.any(self.distribution[desc,ids]!=0) or self.default[desc,gi]:
                                raise ValueError('defaulted netting sets cannot resurrect within the same vintage')
        if np.any(self.lower[tree.leaves]>0) or np.any(self.upper[tree.leaves]<0):raise ValueError('terminal bounds must permit contractual expiry or closing all positions')
        if self.schema_version!='3.0': raise ValueError('unsupported hedge schema')
        return self

    def to_dict(self):
        result={}
        for f in fields(self):
            v=getattr(self,f.name)
            result[f.name]=v.tolist() if isinstance(v,np.ndarray) else [x.__dict__ for x in v] if f.name=='netting_sets' else v
        return result
    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2,allow_nan=False),encoding='utf-8')
    @classmethod
    def load(cls,path,tree):
        return cls(**json.loads(Path(path).read_text())).validate(tree)
    def copy(self,tree,**changes):
        d=self.to_dict();d.update(changes);return HedgeMarket(**d).validate(tree)


def blank_market(tree: ScenarioTree, names: list[str], kinds=None) -> HedgeMarket:
    n,k=tree.size,len(names);z=np.zeros((n,k))
    return HedgeMarket(names,kinds or ['futures']*k,np.ones(k),z.copy(),z.copy(),
        -np.ones((n,k)),np.ones((n,k)),np.full((n,k),2.),z.copy(),z.copy(),np.ones((n,k)),
        z.copy(),np.ones((n,k)),['']*k,[],np.zeros((n,0)),np.ones((n,k)),np.ones((n,k)),
        np.zeros(k),np.zeros(k),np.zeros(k),np.zeros(0),np.zeros(0),[],np.zeros((n,0,k)),
        np.zeros((n,0)),np.zeros((n,0)))
