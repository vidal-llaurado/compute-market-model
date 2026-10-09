"""LP/MILP equivalent of the finite-tree model.

One decision per observable node enforces causality. Financial positions are
selected at the root and held; financing, dispatch, replacement and asset
sales adapt at each node."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import platform
import numpy as np
import scipy
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack
from .schema import Portfolio, Contract
from .risk import nested_value


class OptimizationError(RuntimeError):
    """Numerical failure, unbounded model, or solver limit; not economic infeasibility."""


@dataclass
class LinearModel:
    names: list[str] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    integral: list[int] = field(default_factory=list)
    eq: list[dict[int, float]] = field(default_factory=list)
    eq_rhs: list[float] = field(default_factory=list)
    eq_names: list[str] = field(default_factory=list)
    ub: list[dict[int, float]] = field(default_factory=list)
    ub_rhs: list[float] = field(default_factory=list)
    ub_names: list[str] = field(default_factory=list)

    def var(self, name, lower=0., upper=np.inf, integral=False):
        i = len(self.names)
        self.names.append(name); self.lower.append(lower); self.upper.append(upper)
        self.integral.append(int(integral))
        return i

    def equality(self, row, rhs, name):
        self.eq.append(row); self.eq_rhs.append(float(rhs)); self.eq_names.append(name)

    def inequality(self, row, rhs, name):
        self.ub.append(row); self.ub_rhs.append(float(rhs)); self.ub_names.append(name)

    def matrix(self, rows):
        ii, jj, vv = [], [], []
        for i, row in enumerate(rows):
            for j, v in row.items():
                if v != 0:
                    ii.append(i); jj.append(j); vv.append(v)
        return coo_matrix((vv, (ii,jj)), shape=(len(rows),len(self.names))).tocsr()


def add(row, variable, coefficient):
    row[variable] = row.get(variable, 0.) + float(coefficient)


@dataclass
class SolveResult:
    status: str
    value: float | None
    message: str
    policy: dict[str, list] = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    duals: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return {"status": self.status, "value": self.value, "message": self.message,
                "policy": self.policy, "audit": self.audit, "duals": self.duals,
                "provenance": self.provenance}


def compile_model(p: Portfolio, contract: Contract | None = None, premium: float = 0., fixed_hedge=None):
    if not np.isfinite(premium):
        raise ValueError("premium must be finite")
    c = (contract or Contract.empty(p)).validate(p)
    n_count, j_count, f_count, k_count = p.tree.size, len(p.pool_names), len(p.facility_names), len(p.hedge_names)
    lp = LinearModel()
    v = {}
    shapes = {"holdings": (n_count,j_count), "asset_sales": (n_count,j_count),
              "spot_blocks": (n_count,j_count), "owned_delivery": (n_count,j_count),
              "replacement": (n_count,j_count), "shortfall": (n_count,j_count),
              "debt": (n_count,f_count), "draw": (n_count,f_count), "repay": (n_count,f_count),
              "cash_open": (n_count,), "cash_close": (n_count,), "risk_loss": (n_count,),
              "hedge_long": (k_count,), "hedge_short": (k_count,)}
    integral_names = {"holdings", "asset_sales", "spot_blocks", "owned_delivery", "replacement", "shortfall"}
    for name, shape in shapes.items():
        idx = np.empty(shape, dtype=int)
        for ix in np.ndindex(shape):
            lower, upper = 0., np.inf
            if name == "risk_loss":
                lower = -np.inf
            elif name in ("cash_open", "cash_close"):
                lower = p.minimum_cash[ix[0]]
            elif name in ("holdings",):
                upper = p.initial_blocks[ix[1]]
            elif name == "asset_sales": upper = p.sale_limit[ix]
            elif name == "spot_blocks": upper = p.spot_demand[ix]
            elif name == "replacement": upper = p.replacement_limit[ix]
            elif name == "shortfall": upper = c.shortfall_limit[ix]
            elif name == "owned_delivery": upper = c.required_blocks[ix]
            elif name == "debt": upper = p.commitment[ix]
            elif name == "draw": upper = p.draw_limit[ix]
            elif name == "repay": upper = p.repayment_limit[ix]
            elif name in ("hedge_long", "hedge_short"): upper = p.hedge_gross_limit[ix[0]]
            idx[ix] = lp.var(f"{name}{ix}", lower, upper,
                             p.integer_blocks and name in integral_names)
        v[name] = idx
    if fixed_hedge is not None:
        fixed_hedge = np.asarray(fixed_hedge,float)
        if fixed_hedge.shape != (k_count,) or not np.isfinite(fixed_hedge).all() or np.any(np.abs(fixed_hedge)>p.hedge_gross_limit+1e-10):
            raise ValueError("fixed hedge must be a finite vector within gross limits")
        for k in range(k_count):
            lp.equality({v['hedge_long'][k]:1.},max(0.,fixed_hedge[k]),f"fixed_hedge_long[{k}]")
            lp.equality({v['hedge_short'][k]:1.},max(0.,-fixed_hedge[k]),f"fixed_hedge_short[{k}]")
    for k in range(k_count):
        lp.inequality({v['hedge_long'][k]:1, v['hedge_short'][k]:1},
                      p.hedge_gross_limit[k], f"gross_hedge_limit[{k}]")

    for n in range(n_count):
        parent = p.tree.parent[n]
        root = parent < 0
        # Physical asset inventory and simultaneous delivery constraints.
        for j in range(j_count):
            row = {v['holdings'][n,j]:1, v['asset_sales'][n,j]:1}
            if not root: add(row, v['holdings'][parent,j], -1)
            lp.equality(row, p.initial_blocks[j] if root else 0, f"inventory[{n},{j}]")
            lp.equality({v['owned_delivery'][n,j]:1, v['replacement'][n,j]:1,
                         v['shortfall'][n,j]:1}, c.required_blocks[n,j], f"delivery[{n},{j}]")
            lp.inequality({v['spot_blocks'][n,j]:1, v['owned_delivery'][n,j]:1,
                           v['holdings'][n,j]:-p.availability[n,j]}, 0, f"capacity[{n},{j}]")
        # Debt conservation and explicit, non-double-pledged borrowing bases.
        for f in range(f_count):
            row = {v['debt'][n,f]:1, v['draw'][n,f]:-1, v['repay'][n,f]:1}
            if not root: add(row, v['debt'][parent,f], -1)
            lp.equality(row, p.initial_debt[f] if root else 0, f"debt_balance[{n},{f}]")
            row = {v['debt'][n,f]:1}
            for j in range(j_count):
                add(row, v['holdings'][n,j], -p.advance_rate[f,j]*p.pledge_fraction[f,j]*p.asset_eligible_value[n,j])
            lp.inequality(row, p.receivable_support[n,f], f"borrowing_base[{n},{f}]")

        # Opening obligations settle before operating receipts.
        row = {v['cash_open'][n]:1}
        rhs = float(p.opening_cashflow[n] + c.opening_receipts[n])
        if root: rhs += p.initial_cash*p.cash_growth[n] + premium
        else: add(row, v['cash_close'][parent], -p.cash_growth[n])
        for j in range(j_count):
            add(row, v['asset_sales'][n,j], -p.asset_sale_price[n,j])
        for f in range(f_count):
            fee = p.debt_transaction_fee[f]
            add(row, v['draw'][n,f], -1+fee)
            add(row, v['repay'][n,f], 1+fee)
            accrual = p.facility_rate[n,f]*p.year_fraction[n]
            if root: rhs -= accrual*p.initial_debt[f]
            else: add(row, v['debt'][parent,f], accrual)
        for k in range(k_count):
            previous_margin = 0. if root else p.hedge_margin[parent,k]
            margin_change = p.hedge_margin[n,k]-previous_margin
            for name, sign in (("hedge_long",1), ("hedge_short",-1)):
                coefficient = -sign*p.hedge_settlement[n,k] + margin_change
                if root:
                    coefficient += sign*p.hedge_initial_price[k] + p.hedge_half_spread[k]
                add(row, v[name][k], coefficient)
        lp.equality(row, rhs, f"opening_cash[{n}]")

        # All direct service costs pre-funded; conservative within-node chronology.
        cost = {}
        revenue = {}
        for j in range(j_count):
            gpu_hours = p.gpus_per_block[j]*p.hours[n]
            for name in ("spot_blocks", "owned_delivery"):
                add(cost, v[name][n,j], gpu_hours*p.owned_unit_cost[n,j])
            add(cost, v['replacement'][n,j], gpu_hours*p.replacement_price[n,j])
            add(cost, v['shortfall'][n,j], c.shortfall_cost[n,j])
            add(revenue, v['spot_blocks'][n,j], gpu_hours*p.spot_price[n,j])
        row = dict(cost); add(row, v['cash_open'][n], -1)
        lp.inequality(row, -p.minimum_cash[n], f"operating_prefund[{n}]")
        row = {v['cash_close'][n]:1, v['cash_open'][n]:-1}
        for key, val in cost.items(): add(row, key, val)
        for key, val in revenue.items(): add(row, key, -val)
        for f in range(f_count):
            add(row, v['debt'][n,f], p.facility_rate[n,f]*p.operating_year_fraction[n])
        lp.equality(row, p.closing_cashflow[n]+c.closing_receipts[n], f"closing_cash[{n}]")

    # Recurse on terminal net wealth, including debt and collateral.
    for n in reversed(range(n_count)):
        children = p.tree.children[n]
        if not children:
            row = {v['risk_loss'][n]:1, v['cash_close'][n]:1}
            for j in range(j_count): add(row, v['holdings'][n,j], p.terminal_asset_value[n,j])
            for f in range(f_count): add(row, v['debt'][n,f], -1)
            for k in range(k_count):
                add(row, v['hedge_long'][k], p.hedge_margin[n,k]+p.hedge_terminal_value[n,k])
                add(row, v['hedge_short'][k], p.hedge_margin[n,k]-p.hedge_terminal_value[n,k])
            lp.equality(row, p.terminal_other_liability[n], f"terminal_wealth[{n}]")
            lp.inequality({v['risk_loss'][n]:1}, -p.terminal_solvency_floor, f"terminal_solvency[{n}]")
        elif p.ambiguity_radius[n] > 0:
            from .ambiguity import add_risk_epigraph
            add_risk_epigraph(lp, v['risk_loss'][n], [v['risk_loss'][m] for m in children],
                              [p.tree.probability[m] for m in children],
                              p.discount[children], p.alpha, p.risk_weight,
                              p.ambiguity_radius[n], f"robust[{n}]")
        elif p.risk_weight == 0:
            row = {v['risk_loss'][n]:1}
            for m in children:
                add(row, v['risk_loss'][m], -p.tree.probability[m]*p.discount[m])
            lp.equality(row, 0, f"conditional_expectation[{n}]")
        else:
            z = lp.var(f"tail_threshold[{n}]", -np.inf)
            row = {v['risk_loss'][n]:-1, z:p.risk_weight}
            for m in children:
                u = lp.var(f"tail_excess[{n},{m}]")
                lp.inequality({v['risk_loss'][m]:p.discount[m], z:-1, u:-1}, 0,
                              f"tail_excess[{n},{m}]")
                pm = p.tree.probability[m]
                add(row, v['risk_loss'][m], (1-p.risk_weight)*pm*p.discount[m])
                add(row, u, p.risk_weight*pm/(1-p.alpha))
            lp.inequality(row, 0, f"nested_risk[{n}]")
    objective = np.zeros(len(lp.names)); objective[v['risk_loss'][0]] = 1
    return lp, v, objective, c


def solve(p: Portfolio, contract: Contract | None = None, premium: float = 0.,
          time_limit: float = 30., tolerance: float = 1e-6, fixed_hedge=None) -> SolveResult:
    if not np.isfinite(time_limit) or time_limit <= 0 or tolerance <= 0:
        raise ValueError("positive solver time_limit and tolerance required")
    lp, v, objective, c = compile_model(p, contract, premium, fixed_hedge)
    aeq, aub = lp.matrix(lp.eq), lp.matrix(lp.ub)
    beq, bub = np.asarray(lp.eq_rhs), np.asarray(lp.ub_rhs)
    bounds = list(zip(lp.lower,lp.upper))
    if p.integer_blocks:
        matrix = vstack([aeq,aub]).tocsc()
        constraint = LinearConstraint(matrix, np.r_[beq, np.full(len(bub), -np.inf)], np.r_[beq,bub])
        result = milp(objective, integrality=np.array(lp.integral),
                      bounds=Bounds(lp.lower,lp.upper), constraints=constraint,
                      options={"time_limit":time_limit,"mip_rel_gap":1e-9})
    else:
        result = linprog(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq,
                         bounds=bounds, method="highs", options={"time_limit":time_limit,
                         "dual_feasibility_tolerance":1e-8,"primal_feasibility_tolerance":1e-8})
    payload = {"portfolio":p.to_dict(),"contract":c.to_dict(),"premium":premium,
               "fixed_hedge":None if fixed_hedge is None else np.asarray(fixed_hedge,float).tolist()}
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    provenance = {"input_sha256":digest,"python":platform.python_version(),
                  "numpy":np.__version__,"scipy":scipy.__version__,"project":"Compute Market",
                  "solver":"HiGHS MILP" if p.integer_blocks else "HiGHS LP",
                  "units":"USD; blocks; GPU-hours; year fractions"}
    if result.status == 2:
        return SolveResult("infeasible",None,str(result.message),provenance=provenance)
    if not result.success or result.status != 0 or result.x is None:
        raise OptimizationError(f"solver status {result.status}: {result.message}")
    x = result.x
    eq_error = float(np.max(np.abs(aeq@x-beq),initial=0.))
    ub_error = float(np.max(aub@x-bub,initial=0.))
    eq_scaled = float(np.max(np.abs(aeq@x-beq)/np.maximum(1.,np.abs(beq)),initial=0.))
    ub_scaled = float(np.max((aub@x-bub)/np.maximum(1.,np.abs(bub)),initial=0.))
    bound_error = max(float(np.max(np.asarray(lp.lower)-x,initial=0.)),
                      float(np.max(x-np.asarray(lp.upper),initial=0.)))
    integral = np.asarray(lp.integral,dtype=bool)
    int_error = float(np.max(np.abs(x[integral]-np.rint(x[integral])),initial=0.))
    policy = {name:x[idx].tolist() for name,idx in v.items()}
    policy["hedge_net"] = (x[v['hedge_long']]-x[v['hedge_short']]).tolist()
    leaves = p.tree.leaves
    terminal = np.zeros(p.tree.size)
    terminal[leaves] = -x[v['risk_loss'][leaves]]
    if np.any(p.ambiguity_radius > 0):
        from .ambiguity import robust_nested_value
        values = robust_nested_value(p.tree,terminal,p.alpha,p.risk_weight,p.discount,p.ambiguity_radius)
    else:
        values = nested_value(p.tree,terminal,p.alpha,p.risk_weight,p.discount)
    recursion_error = abs(float(values[0])+float(result.fun))
    audit = {"max_equality_residual":eq_error,"max_inequality_violation":ub_error,
             "max_scaled_equality_residual":eq_scaled,"max_scaled_inequality_violation":ub_scaled,
             "max_bound_violation":bound_error,"max_integrality_error":int_error,
             "risk_recursion_error":recursion_error,
             "minimum_open_cash":float(np.min(x[v['cash_open']])),
             "minimum_close_cash":float(np.min(x[v['cash_close']])),
             "minimum_terminal_wealth":float(np.min(terminal[leaves])),
             "expected_terminal_wealth":float(p.tree.unconditional[leaves]@terminal[leaves]),
             "node_count":p.tree.size,"leaf_count":len(leaves),
             "variables":len(lp.names),"constraints":len(lp.eq)+len(lp.ub),
             "max_ambiguity_radius":float(np.max(p.ambiguity_radius)),
             "scope":"survival only in the supplied finite scenario tree"}
    from .validation import independent_policy_audit
    audit.update(independent_policy_audit(p,c,policy,premium))
    if p.integer_blocks: audit["mip_gap"] = float(result.mip_gap)
    scale = max(1.,abs(float(result.fun)))
    physical_error = max(audit["independent_inventory_error"],audit["independent_delivery_error"],audit["independent_capacity_violation"])
    money_error = max(audit["independent_opening_cash_error"],audit["independent_closing_cash_error"],audit["independent_terminal_wealth_error"],audit["independent_debt_balance_error"])
    if max(eq_scaled,ub_scaled) > tolerance or bound_error > tolerance*scale or physical_error > tolerance or money_error > tolerance*scale or int_error > tolerance or recursion_error > tolerance*scale:
        raise OptimizationError(f"post-solve verification failed: {audit}")
    duals = {}
    if not p.integer_blocks:
        # Objective is LOSS; negate RHS sensitivities to obtain VALUE sensitivities.
        duals["value_rhs_equalities"] = dict(zip(lp.eq_names,(-result.eqlin.marginals).tolist()))
        duals["value_rhs_inequalities"] = dict(zip(lp.ub_names,(-result.ineqlin.marginals).tolist()))
        duals["marginal_value_initial_cash"] = float(-result.eqlin.marginals[lp.eq_names.index('opening_cash[0]')]*p.cash_growth[0])
    return SolveResult("optimal",float(-result.fun),str(result.message),policy,audit,duals,provenance)
