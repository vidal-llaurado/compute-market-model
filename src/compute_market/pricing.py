"""Financing-aware indifference pricing and finite quote-menu optimization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .schema import Portfolio, Contract
from .contracts import combine, node_gpu_hours
from .optimizer import solve, SolveResult
from .risk import certainty_equivalent


@dataclass
class PriceResult:
    amount: float
    kind: str
    base_value: float
    accepted_value: float
    iterations: int
    bracket: tuple[float,float]
    valuation_gap: float

    def to_dict(self):
        return self.__dict__.copy()


def _bracket_price(evaluate: Callable[[float], SolveResult], target: float,
                   kind: str, price_tolerance: float, value_tolerance: float,
                   max_absolute_price: float, initial_step: float) -> PriceResult:
    if price_tolerance <= 0 or value_tolerance < 0 or max_absolute_price <= 0 or initial_step <= 0:
        raise ValueError("invalid price-search limits")
    calls = 0
    def acceptable(x):
        nonlocal calls
        calls += 1
        r = evaluate(float(x))
        return r.status == "optimal" and r.value >= target-value_tolerance, r
    ok, result = acceptable(0.)
    # A negative reservation price is legitimate for risk-reducing obligations.
    if ok:
        hi, best = 0., result
        lo = -min(initial_step,max_absolute_price)
        while True:
            ok_lo, rlo = acceptable(lo)
            if not ok_lo: break
            hi,best = lo,rlo
            if abs(lo) >= max_absolute_price:
                raise ValueError("reservation price below configured search range")
            lo = -min(2*abs(lo),max_absolute_price)
    else:
        lo,hi = 0.,min(initial_step,max_absolute_price)
        while True:
            ok_hi,rhi = acceptable(hi)
            if ok_hi:
                best = rhi; break
            lo = hi
            if hi >= max_absolute_price:
                raise ValueError("no admissible price in search range; capacity or funding may be infeasible")
            hi = min(2*hi,max_absolute_price)
    while hi-lo > price_tolerance:
        mid = (lo+hi)/2
        ok_mid,rmid = acceptable(mid)
        if ok_mid: hi,best = mid,rmid
        else: lo = mid
        if calls > 200:
            raise RuntimeError("price bisection failed to converge")
    return PriceResult(float(hi),kind,float(target),float(best.value),calls,(float(lo),float(hi)),
                       float(best.value-target))


def reservation_price(p: Portfolio, offered: Contract, book: Contract | None = None,
                      payment: str = "upfront", price_tolerance: float = .01,
                      value_tolerance: float = 1e-6, max_absolute_price: float = 1e9,
                      initial_step: float | None = None) -> PriceResult:
    base = book or Contract.empty(p)
    reference = solve(p,base)
    if reference.status != "optimal":
        raise ValueError("baseline book is infeasible: recapitalize or cure it before quoting")
    merged = combine(base,offered,p)
    if payment == "upfront":
        fn = lambda amount: solve(p,merged,premium=amount)
        kind, step = "USD upfront", 100. if initial_step is None else initial_step
    elif payment in ("opening_rate","closing_rate"):
        # A unit rate times actual node obligations: no annuity shortcut.
        volumes = node_gpu_hours(p,offered)
        if volumes.sum() <= 0:
            raise ValueError("running-rate pricing requires positive physical delivery")
        def fn(rate):
            c = Contract(merged.name,merged.required_blocks.copy(),merged.opening_receipts.copy(),
                         merged.closing_receipts.copy(),merged.shortfall_limit.copy(),merged.shortfall_cost.copy())
            if payment == "opening_rate": c.opening_receipts += rate*volumes
            else: c.closing_receipts += rate*volumes
            return solve(p,c)
        kind,step = f"USD/GPU-hour ({payment})",1. if initial_step is None else initial_step
    else:
        raise ValueError("payment must be upfront, opening_rate or closing_rate")
    return _bracket_price(fn,reference.value,kind,price_tolerance,value_tolerance,max_absolute_price,step)


def quote_menu(p: Portfolio, offered: Contract, prices, acceptance,
               book: Contract | None = None) -> dict:
    """Optimize request acceptance followed by the market tree.

    Prices are total upfront amounts. Acceptance probabilities come from a
    separate choice model; conditional on acceptance, future market states follow
    the supplied tree without further dependence on the current choice."""
    prices, probs = np.asarray(prices,float),np.asarray(acceptance,float)
    if prices.ndim != 1 or probs.shape != prices.shape or not prices.size or not np.isfinite(prices).all() or not np.isfinite(probs).all() or np.any((probs < 0)|(probs > 1)):
        raise ValueError("finite prices and matching acceptance probabilities in [0,1] required")
    base = book or Contract.empty(p)
    ref = solve(p,base)
    if ref.status != "optimal": raise ValueError("infeasible baseline")
    merged = combine(base,offered,p)
    rows=[]; best_price=None; best_gain=0.
    for price,prob in zip(prices,probs):
        result=solve(p,merged,premium=float(price))
        gain=None
        if result.status == "optimal":
            gain=certainty_equivalent([ref.value,result.value],[1-prob,prob],p.alpha,p.risk_weight)-ref.value
            if gain > best_gain+1e-8:
                best_price,best_gain=float(price),float(gain)
        rows.append({"price":float(price),"acceptance":float(prob),"gain":gain,"status":result.status})
    return {"quote":best_price,"incremental_value":best_gain,"candidates":rows,
            "decision":"quote" if best_price is not None else "no_quote"}


def quote_catalogue(p: Portfolio, offered: list[Contract], price_grids: list[list[float]],
                    performance, preference, sensitivity: float, book: Contract | None=None) -> dict:
    """Choose a joint price menu for a workload buying at most one configuration.

    Prices are total upfront USD; performance converts equivalent useful work.
    Contracts must already satisfy eligibility and service requirements. Future
    exogenous states are conditionally independent of the current choice.
    Cache repeated conditional-acceptance valuations across menus."""
    from itertools import product
    from .market import logit_choice
    if len(offered) == 0 or len(price_grids) != len(offered) or any(not g for g in price_grids):
        raise ValueError("one nonempty price grid per eligible contract required")
    grids=[np.asarray(g,float) for g in price_grids]
    if any(g.ndim != 1 or not np.isfinite(g).all() for g in grids):
        raise ValueError("finite 1-D price grids required")
    base=book or Contract.empty(p);ref=solve(p,base)
    if ref.status != 'optimal':raise ValueError("infeasible baseline")
    cache={}
    for i,c in enumerate(offered):
        merged=combine(base,c,p)
        for price in grids[i]:
            result=solve(p,merged,premium=float(price))
            cache[i,float(price)]=result.value if result.status=='optimal' else None
    best=None;gain=0.;evaluated=0
    for menu in product(*grids):
        values=[cache[i,float(price)] for i,price in enumerate(menu)]
        if any(v is None for v in values):continue
        choice=logit_choice(menu,performance,preference,sensitivity)
        probabilities=[choice['outside']]+choice['products']
        improvement=certainty_equivalent([ref.value]+values,probabilities,p.alpha,p.risk_weight)-ref.value
        evaluated+=1
        if improvement>gain+1e-8:
            gain=float(improvement)
            best={"prices":[float(x) for x in menu],"choice_probabilities":probabilities}
    return {"decision":"quote" if best else "no_quote","menu":best,
            "incremental_value":gain,"menus_evaluated":evaluated,
            "conditional_valuations":len(cache)}
