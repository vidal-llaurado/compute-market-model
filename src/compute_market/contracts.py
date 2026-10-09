"""Exact contract translations and path-dependent payoff primitives."""
from __future__ import annotations
import numpy as np
from .schema import Portfolio, Contract
from .risk import certainty_equivalent


def combine(a: Contract, b: Contract, p: Portfolio) -> Contract:
    a.validate(p); b.validate(p)
    overlap = (a.shortfall_limit > 0) & (b.shortfall_limit > 0)
    if np.any(overlap & (~np.isclose(a.shortfall_cost,b.shortfall_cost))):
        raise ValueError("different shortfall penalties require separate contract identities; cannot aggregate")
    penalty = np.where(a.shortfall_limit > 0,a.shortfall_cost,b.shortfall_cost)
    return Contract(f"{a.name} + {b.name}", a.required_blocks+b.required_blocks,
                    a.opening_receipts+b.opening_receipts, a.closing_receipts+b.closing_receipts,
                    a.shortfall_limit+b.shortfall_limit, penalty).validate(p)


def firm_strip(p: Portfolio, name: str, pool: str, blocks: float,
               first_stage: int, last_stage: int) -> Contract:
    if pool not in p.pool_names or not np.isfinite(blocks) or blocks <= 0:
        raise ValueError("known pool and positive finite block quantity required")
    if first_stage < 0 or first_stage > last_stage or last_stage > max(p.tree.stage):
        raise ValueError("invalid delivery stages")
    c = Contract.empty(p,name)
    j = p.pool_names.index(pool)
    for n,stage in enumerate(p.tree.stage):
        if first_stage <= stage <= last_stage:
            c.required_blocks[n,j] = blocks
    return c.validate(p)


def node_gpu_hours(p: Portfolio, c: Contract) -> np.ndarray:
    c.validate(p)
    return p.hours * (c.required_blocks @ p.gpus_per_block)


def deterministic_gpu_hours(p: Portfolio, c: Contract) -> float:
    v = node_gpu_hours(p,c)
    totals = np.array([sum(v[n] for n in p.tree.path(leaf)) for leaf in p.tree.leaves])
    if not np.allclose(totals,totals[0],atol=1e-8,rtol=1e-10):
        raise ValueError("path-dependent volume cannot be labeled a deterministic strip")
    return float(totals[0])


def average_cap(prices, weights, quantity: float, strike: float) -> np.ndarray:
    s = np.asarray(prices,float); w = np.asarray(weights,float)
    if s.ndim < 1 or s.shape[-1] != w.size or w.ndim != 1 or not np.isfinite(s).all() or not np.isfinite(w).all():
        raise ValueError("finite prices with last axis matching weights required")
    if np.any(w < 0) or not np.isclose(w.sum(),1) or quantity < 0 or not np.isfinite(quantity) or not np.isfinite(strike):
        raise ValueError("normalized nonnegative weights and valid contract terms required")
    return quantity*np.maximum(s@w-strike,0.)


def margin_floor(compute, power, weights, energy_intensity: float,
                 quantity: float, floor: float) -> np.ndarray:
    a,b = np.asarray(compute,float),np.asarray(power,float)
    if a.shape != b.shape or not np.isfinite(energy_intensity) or energy_intensity < 0:
        raise ValueError("matching compute/power arrays and nonnegative MWh/GPU-hour required")
    # Floor on margin equals a cap on negative margin at negative strike.
    return average_cap(-(a-energy_intensity*b),weights,quantity,-floor)


def terminal_cash_contract(p: Portfolio, name: str, leaf_losses) -> Contract:
    """Seller losses settle at LEAF CLOSE. They cannot fund earlier payments."""
    values = np.asarray(leaf_losses,float)
    if values.shape != (len(p.tree.leaves),) or not np.isfinite(values).all():
        raise ValueError("one finite settlement for each leaf required")
    c = Contract.empty(p,name)
    c.closing_receipts[p.tree.leaves] = -values
    return c


def swing_value(tree, payoff_per_unit, max_per_node: int, min_total: int, max_total: int,
                discounts=None) -> dict:
    """Buyer exercise value on a fixed risk-neutral tree.

    The buyer observes price before choosing an integer nomination. The root has
    no exercise; each later node is an exercise date. Physical delivery and a
    seller's hedge are outside this claim calculation."""
    payoff = np.asarray(payoff_per_unit,float)
    d = np.ones(tree.size) if discounts is None else np.asarray(discounts,float)
    if payoff.shape != (tree.size,) or d.shape != payoff.shape or not np.isfinite(payoff).all() or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("valid node payoff and discounts required")
    if not all(isinstance(x,int) for x in (max_per_node,min_total,max_total)) or not 0 <= min_total <= max_total or max_per_node < 0:
        raise ValueError("invalid integer swing volume limits")
    values, policy = {}, {}
    for n in reversed(range(tree.size)):
        for used in range(max_total+1):
            best, best_q = -np.inf, None
            allowed = [0] if n == 0 else range(min(max_per_node,max_total-used)+1)
            for q in allowed:
                cumulative = used+q
                if tree.children[n]:
                    continuation = sum(tree.probability[m]*d[m]*values[m,cumulative]
                                       for m in tree.children[n])
                else:
                    continuation = 0. if min_total <= cumulative <= max_total else -np.inf
                candidate = q*payoff[n]+continuation
                if candidate > best:
                    best,best_q = candidate,q
            values[n,used],policy[n,used] = best,best_q
    if not np.isfinite(values[0,0]):
        return {"feasible":False,"value":None,"policy":{}}
    return {"feasible":True,"value":float(values[0,0]),
            "policy":{f"{n}:{used}":q for (n,used),q in policy.items() if q is not None}}
