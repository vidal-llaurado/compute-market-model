"""Complete valuation cases: market, physical book, financing and proposed contract.

Public workflows require the existing book; an empty book must be explicit."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Any
from .schema import Portfolio, Contract
from .hedge_market import HedgeMarket
from .dynamic_hedging import solve_dynamic
from .hedge_pricing import price_and_hedge


@dataclass
class ValuationCase:
    name: str
    portfolio: Portfolio
    market: HedgeMarket
    book: Contract
    offered: Contract | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)
    schema: str = 'compute-market/case/1'

    def __post_init__(self):
        if not self.name or self.schema != 'compute-market/case/1':
            raise ValueError('named case with supported schema required')
        if self.book is None:
            raise ValueError('existing book is required; use Contract.empty(portfolio) deliberately')
        self.book.validate(self.portfolio)
        self.market.validate(self.portfolio.tree)
        if self.offered is not None:self.offered.validate(self.portfolio)

    def to_dict(self):
        return {'schema':self.schema,'name':self.name,'portfolio':self.portfolio.to_dict(),
                'market':self.market.to_dict(),'book':self.book.to_dict(),
                'offered':None if self.offered is None else self.offered.to_dict(),
                'assumptions':self.assumptions}

    @property
    def fingerprint(self):
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def save(self, path):
        out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(self.to_dict(),indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def from_dict(cls, data):
        d=dict(data)
        if 'book' not in d or d['book'] is None:
            raise ValueError('case has no existing book; omission is not an empty-book request')
        d['portfolio']=Portfolio(**d['portfolio'])
        d['market']=HedgeMarket(**d['market'])
        d['book']=Contract(**d['book'])
        if d.get('offered') is not None:d['offered']=Contract(**d['offered'])
        return cls(**d)

    @classmethod
    def load(cls,path):return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def solve(self, mode='dynamic', **kwargs):
        result=solve_dynamic(self.portfolio,self.market,self.book,mode=mode,**kwargs)
        result.provenance['case_sha256']=self.fingerprint
        result.provenance['case_name']=self.name
        return result

    def quote(self, **kwargs):
        if self.offered is None:raise ValueError('case has no offered contract')
        result=price_and_hedge(self.portfolio,self.market,self.offered,self.book,**kwargs)
        result['case_sha256']=self.fingerprint
        result['case_name']=self.name
        return result

    def verify_result(self,result):
        """Reject a different book even when the portfolio happens to be identical."""
        d=result.to_dict() if hasattr(result,'to_dict') else result
        if d.get('provenance',{}).get('case_sha256')!=self.fingerprint:
            raise ValueError('result does not belong to this complete valuation case')
        if d.get('status')!='optimal' or not d.get('policy'):
            raise ValueError('verification requires an optimal solved policy')
        from .hedge_audit import audit_dynamic
        return audit_dynamic(self.portfolio,self.market,self.book,d['policy'],float(d.get('provenance',{}).get('premium',0.)))
