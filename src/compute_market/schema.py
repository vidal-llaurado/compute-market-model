"""Validated, serializable model inputs. Cash units are consistent within one run."""
from __future__ import annotations
from dataclasses import dataclass, fields
import json
from pathlib import Path
import numpy as np
from .tree import ScenarioTree


def array(value, shape, name, lower=None, upper=None):
    a = np.array(value, dtype=float, copy=True)
    if a.size == 0 and np.prod(shape) == 0:
        a = a.reshape(shape)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite shape {shape} required, got {a.shape}")
    if lower is not None and np.any(a < lower):
        raise ValueError(f"{name}: values must be >= {lower}")
    if upper is not None and np.any(a > upper):
        raise ValueError(f"{name}: values must be <= {upper}")
    return a


@dataclass
class Portfolio:
    tree: ScenarioTree
    pool_names: list[str]
    gpus_per_block: np.ndarray                  # [J]
    initial_blocks: np.ndarray                 # [J]
    hours: np.ndarray                          # [N], delivery hours at node
    year_fraction: np.ndarray                  # [N], accrual preceding node
    availability: np.ndarray                   # [N,J], fraction of owned blocks usable
    spot_price: np.ndarray                     # [N,J], USD / GPU-hour
    spot_demand: np.ndarray                    # [N,J], concurrent blocks
    owned_unit_cost: np.ndarray                # [N,J], USD / GPU-hour, energy included
    replacement_price: np.ndarray             # [N,J], USD / GPU-hour, all-in
    replacement_limit: np.ndarray             # [N,J], executable concurrent blocks
    asset_sale_price: np.ndarray               # [N,J], USD / block, net realizable
    asset_eligible_value: np.ndarray           # [N,J], lender mark / block
    terminal_asset_value: np.ndarray           # [N,J], NET exit value, used at leaves only
    sale_limit: np.ndarray                     # [N,J], maximum blocks sold at node
    initial_cash: float
    cash_growth: np.ndarray                    # [N], growth from parent close to node open
    discount: np.ndarray                       # [N], value discount on incoming edge
    opening_cashflow: np.ndarray               # [N], BEFORE operating receipts/costs
    closing_cashflow: np.ndarray               # [N], AFTER operating receipts/costs
    minimum_cash: np.ndarray                   # [N], enforced before and after operations
    terminal_other_liability: np.ndarray       # [N], leaves only; includes unexpired claims
    facility_names: list[str]
    initial_debt: np.ndarray                   # [F]
    commitment: np.ndarray                     # [N,F], zero at maturity forces repayment
    draw_limit: np.ndarray                     # [N,F], can be zero during draw freeze
    repayment_limit: np.ndarray                # [N,F]
    facility_rate: np.ndarray                  # [N,F], annual rate on inherited balance
    advance_rate: np.ndarray                   # [F,J]
    pledge_fraction: np.ndarray                # [F,J], sums across facilities <= 1
    receivable_support: np.ndarray             # [N,F], already haircut, exclusive eligible amount
    debt_transaction_fee: np.ndarray           # [F], per dollar drawn or repaid
    hedge_names: list[str]
    hedge_settlement: np.ndarray               # [N,K], cash received per LONG unit at open
    hedge_initial_price: np.ndarray            # [K], paid per LONG, received per SHORT
    hedge_half_spread: np.ndarray              # [K], cost per gross unit
    hedge_gross_limit: np.ndarray              # [K], long+short bound
    hedge_margin: np.ndarray                   # [N,K], posted cash per gross unit
    hedge_terminal_value: np.ndarray           # [N,K], remaining NET closeout value, leaves
    alpha: float = .9
    risk_weight: float = .25
    integer_blocks: bool = False
    terminal_solvency_floor: float = 0.0
    schema_version: str = "2.0"
    operating_year_fraction: np.ndarray | None = None  # [N], delivery interval accrual, disjoint from year_fraction

    ambiguity_radius: np.ndarray | None = None  # [N], rectangular TV radius; zero gives the specified-probability risk model

    def __post_init__(self):
        if not isinstance(self.integer_blocks,(bool,np.bool_)):
            raise ValueError("integer_blocks must be a boolean")
        if self.schema_version not in ("1.0", "2.0"):
            raise ValueError("unsupported schema_version")
        if not isinstance(self.tree, ScenarioTree):
            self.tree = ScenarioTree(**self.tree)
        n, j, f, k = self.tree.size, len(self.pool_names), len(self.facility_names), len(self.hedge_names)
        if self.ambiguity_radius is None:
            self.ambiguity_radius = np.zeros(n)
        if self.operating_year_fraction is None:
            self.operating_year_fraction = np.zeros(n)
        if j == 0 or len(set(self.pool_names)) != j:
            raise ValueError("at least one uniquely named capacity pool required")
        if len(set(self.facility_names)) != f or len(set(self.hedge_names)) != k:
            raise ValueError("facility and hedge names must be unique")
        shapes = {
            "ambiguity_radius": (n,), "gpus_per_block": (j,), "initial_blocks": (j,), "hours": (n,),
            "year_fraction": (n,), "operating_year_fraction": (n,), "availability": (n,j), "spot_price": (n,j),
            "spot_demand": (n,j), "owned_unit_cost": (n,j),
            "replacement_price": (n,j), "replacement_limit": (n,j),
            "asset_sale_price": (n,j), "asset_eligible_value": (n,j),
            "terminal_asset_value": (n,j), "sale_limit": (n,j), "cash_growth": (n,),
            "discount": (n,), "opening_cashflow": (n,), "closing_cashflow": (n,),
            "minimum_cash": (n,), "terminal_other_liability": (n,),
            "initial_debt": (f,), "commitment": (n,f), "draw_limit": (n,f),
            "repayment_limit": (n,f), "facility_rate": (n,f), "advance_rate": (f,j),
            "pledge_fraction": (f,j), "receivable_support": (n,f),
            "debt_transaction_fee": (f,), "hedge_settlement": (n,k),
            "hedge_initial_price": (k,), "hedge_half_spread": (k,),
            "hedge_gross_limit": (k,), "hedge_margin": (n,k),
            "hedge_terminal_value": (n,k)}
        signed = {"opening_cashflow", "closing_cashflow", "hedge_settlement",
                  "hedge_initial_price", "hedge_terminal_value", "facility_rate"}
        for name, shape in shapes.items():
            setattr(self, name, array(getattr(self, name), shape, name,
                                     lower=None if name in signed else 0))
        for name in ("availability", "advance_rate", "pledge_fraction", "ambiguity_radius"):
            if np.any(getattr(self, name) > 1):
                raise ValueError(f"{name} must not exceed one")
        if np.any(self.pledge_fraction.sum(axis=0) > 1+1e-10):
            raise ValueError("the same hardware cannot be pledged more than once")
        if np.any(self.gpus_per_block <= 0) or np.any(self.cash_growth <= 0) or np.any(self.discount <= 0):
            raise ValueError("block sizes and growth/discount factors must be positive")
        if not np.isfinite(self.initial_cash) or self.initial_cash < 0:
            raise ValueError("initial_cash must be finite and nonnegative")
        if not np.isfinite(self.terminal_solvency_floor) or self.terminal_solvency_floor < 0:
            raise ValueError("terminal_solvency_floor must be finite and nonnegative")
        if not 0 <= self.alpha < 1 or not 0 <= self.risk_weight <= 1:
            raise ValueError("invalid risk parameters")
        if self.integer_blocks and not np.allclose(self.initial_blocks, np.rint(self.initial_blocks)):
            raise ValueError("integer mode requires integer initial block counts")
        if not np.isclose(self.discount[0], 1.0):
            raise ValueError("root discount must be one")
        # A zero-hour node cannot sell free service or carry delivery obligations.
        if np.any(self.spot_demand[self.hours == 0] != 0):
            raise ValueError("zero-hour nodes must have zero spot demand")

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = v.to_dict() if f.name == "tree" else v.tolist() if isinstance(v, np.ndarray) else v
        return d

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self, **changes):
        d = self.to_dict(); d.update(changes)
        return Portfolio(**d)


@dataclass
class Contract:
    name: str
    required_blocks: np.ndarray                # [N,J], exact pool requirements
    opening_receipts: np.ndarray               # [N], can encode cash-settled liabilities (<0)
    closing_receipts: np.ndarray               # [N]
    shortfall_limit: np.ndarray                # [N,J], zero for firm delivery
    shortfall_cost: np.ndarray                 # [N,J], total penalty per block per interval

    @classmethod
    def empty(cls, portfolio: Portfolio, name: str = "empty"):
        n, j = portfolio.tree.size, len(portfolio.pool_names)
        return cls(name, np.zeros((n,j)), np.zeros(n), np.zeros(n), np.zeros((n,j)), np.zeros((n,j)))

    def validate(self, portfolio: Portfolio):
        n, j = portfolio.tree.size, len(portfolio.pool_names)
        for name, shape in [("required_blocks", (n,j)), ("shortfall_limit", (n,j)),
                            ("shortfall_cost", (n,j)), ("opening_receipts", (n,)), ("closing_receipts", (n,))]:
            setattr(self, name, array(getattr(self, name), shape, name,
                                     lower=None if "receipts" in name else 0))
        if np.any(self.shortfall_limit > self.required_blocks):
            raise ValueError("shortfall allowance cannot exceed promised delivery")
        if np.any(self.required_blocks[portfolio.hours == 0] != 0):
            raise ValueError("cannot require service in a zero-hour node")
        if portfolio.integer_blocks and not np.allclose(self.required_blocks, np.rint(self.required_blocks)):
            raise ValueError("integer mode requires integer contract blocks")
        return self

    def to_dict(self):
        return {f.name: getattr(self, f.name).tolist() if isinstance(getattr(self, f.name), np.ndarray)
                else getattr(self, f.name) for f in fields(self)}

    def with_cash(self, at_open: np.ndarray):
        return Contract(self.name, self.required_blocks.copy(), self.opening_receipts+at_open,
                        self.closing_receipts.copy(), self.shortfall_limit.copy(), self.shortfall_cost.copy())

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
