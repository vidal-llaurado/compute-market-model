"""Contractual debt schedules and collateral-cure arithmetic."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .tree import ScenarioTree


@dataclass(frozen=True)
class TermLoan:
    name: str
    principal: float
    annual_rate: float
    principal_payments: dict[int, float]
    floating: bool = False

    def compile(self, tree: ScenarioTree, year_fraction, reference_rate=None):
        """Coupon at node uses inherited principal; amortization then reduces balance.

        For floating loans annual_rate is the contractual spread; otherwise it is
        the fixed coupon. Remaining principal is returned for terminal deduction.
        A principal_payment is keyed by stage, not by simulated path.
        """
        dt = np.asarray(year_fraction,float)
        base = np.zeros(tree.size) if reference_rate is None else np.asarray(reference_rate,float)
        if dt.shape != (tree.size,) or base.shape != dt.shape or not np.isfinite(dt).all() or not np.isfinite(base).all() or np.any(dt < 0):
            raise ValueError("valid node-wise year fractions and reference rates required")
        if not np.isfinite(self.principal) or self.principal < 0 or not np.isfinite(self.annual_rate):
            raise ValueError("invalid loan principal or coupon")
        if any(not isinstance(t,int) or t < 0 or not np.isfinite(a) or a < 0 for t,a in self.principal_payments.items()):
            raise ValueError("principal payments require nonnegative integer stages and amounts")
        if sum(self.principal_payments.values()) > self.principal+1e-8:
            raise ValueError("principal payments exceed original principal")
        payments, balances = np.zeros(tree.size), np.zeros(tree.size)
        for n in range(tree.size):
            parent = tree.parent[n]
            inherited = self.principal if parent < 0 else balances[parent]
            amort = self.principal_payments.get(tree.stage[n],0.)
            if amort > inherited+1e-8:
                raise ValueError("scheduled repayment exceeds outstanding principal")
            coupon = self.annual_rate+(base[n] if self.floating else 0.)
            payments[n] = inherited*coupon*dt[n]+amort
            balances[n] = inherited-amort
        terminal = np.zeros(tree.size)
        terminal[tree.leaves] = balances[tree.leaves]
        return payments, terminal


def collateral_cure(asset_value: float, debt: float, available_cash: float,
                    advance_rate: float, realization_fraction: float) -> dict:
    vals = (asset_value,debt,available_cash,advance_rate,realization_fraction)
    if not all(np.isfinite(vals)) or min(asset_value,debt,available_cash) < 0:
        raise ValueError("finite nonnegative assets, debt and available cash required")
    if not 0 <= advance_rate <= 1 or not 0 < realization_fraction <= 1:
        raise ValueError("advance in [0,1], realization in (0,1] required")
    cash_used = min(available_cash,debt)
    gap = max(0.,debt-cash_used-advance_rate*asset_value)
    if gap <= 1e-12:
        return {"feasible":True,"marked_asset_sale":0.,"cash_used":cash_used,"gap":gap}
    denominator = realization_fraction-advance_rate
    if denominator <= 0:
        return {"feasible":False,"marked_asset_sale":None,"cash_used":cash_used,
                "gap":gap,"reason":"selling collateral cannot improve this borrowing-base gap"}
    sale = gap/denominator
    return {"feasible":sale <= asset_value+1e-8,"marked_asset_sale":sale,
            "cash_used":cash_used,"gap":gap,
            "reason":"sufficient assets" if sale <= asset_value+1e-8 else "required sale exceeds assets"}
