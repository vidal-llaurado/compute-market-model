"""Independent arithmetic checks on a solved policy, separate from LP construction."""
from __future__ import annotations
import numpy as np


def independent_policy_audit(p,c,policy,premium=0.):
    # Keep the arithmetic separate from the optimizer's coefficient assembly.
    a={k:np.asarray(v,float) for k,v in policy.items()}
    long=a['hedge_long'];short=a['hedge_short'];net=long-short;gross=long+short
    open_errors=[];close_errors=[];inventory_errors=[];debt_errors=[]
    pre_headroom=[];base_headroom=[];delivery_errors=[];capacity_violations=[]
    for n in range(p.tree.size):
        parent=p.tree.parent[n]
        cash=p.initial_cash if parent<0 else a['cash_close'][parent]
        debt=p.initial_debt if parent<0 else a['debt'][parent]
        assets=p.initial_blocks if parent<0 else a['holdings'][parent]
        margin_before=0. if parent<0 else float(p.hedge_margin[parent]@gross)
        expected_open=(cash*p.cash_growth[n]+p.opening_cashflow[n]+c.opening_receipts[n]
            +a['asset_sales'][n]@p.asset_sale_price[n]
            +a['draw'][n].sum()-a['repay'][n].sum()
            -(p.facility_rate[n]*p.year_fraction[n])@debt
            -p.debt_transaction_fee@(a['draw'][n]+a['repay'][n])
            +p.hedge_settlement[n]@net-p.hedge_margin[n]@gross+margin_before)
        if parent<0:expected_open+=premium-p.hedge_initial_price@net-p.hedge_half_spread@gross
        open_errors.append(abs(a['cash_open'][n]-expected_open))
        volume=p.gpus_per_block*p.hours[n]
        cost=float(volume@(p.owned_unit_cost[n]*(a['spot_blocks'][n]+a['owned_delivery'][n])
                           +p.replacement_price[n]*a['replacement'][n])
                   +c.shortfall_cost[n]@a['shortfall'][n])
        sales=float(volume@(p.spot_price[n]*a['spot_blocks'][n]))
        operating_interest=float((p.facility_rate[n]*p.operating_year_fraction[n])@a['debt'][n])
        expected_close=(a['cash_open'][n]-cost+sales+p.closing_cashflow[n]
                        +c.closing_receipts[n]-operating_interest)
        close_errors.append(abs(a['cash_close'][n]-expected_close))
        pre_headroom.append(float(a['cash_open'][n]-cost-p.minimum_cash[n]))
        inventory_errors.append(float(np.max(np.abs(a['holdings'][n]+a['asset_sales'][n]-assets),initial=0)))
        debt_errors.append(float(np.max(np.abs(a['debt'][n]-debt-a['draw'][n]+a['repay'][n]),initial=0)))
        base=(p.advance_rate*p.pledge_fraction*p.asset_eligible_value[n][None,:])@a['holdings'][n]+p.receivable_support[n]
        if len(base):base_headroom.append(float(np.min(base-a['debt'][n])))
        delivery_errors.append(float(np.max(np.abs(a['owned_delivery'][n]+a['replacement'][n]+a['shortfall'][n]-c.required_blocks[n]),initial=0)))
        capacity_violations.append(float(np.max(a['owned_delivery'][n]+a['spot_blocks'][n]-p.availability[n]*a['holdings'][n],initial=0)))
    terminal_errors=[]
    for n in p.tree.leaves:
        wealth=(a['cash_close'][n]+p.terminal_asset_value[n]@a['holdings'][n]
                -a['debt'][n].sum()-p.terminal_other_liability[n]
                +p.hedge_margin[n]@gross+p.hedge_terminal_value[n]@net)
        terminal_errors.append(abs(wealth+a['risk_loss'][n]))
    return {"independent_opening_cash_error":max(open_errors,default=0.),
            "independent_closing_cash_error":max(close_errors,default=0.),
            "independent_terminal_wealth_error":max(terminal_errors,default=0.),
            "independent_debt_balance_error":max(debt_errors,default=0.),
            "independent_inventory_error":max(inventory_errors,default=0.),
            "independent_delivery_error":max(delivery_errors,default=0.),
            "independent_capacity_violation":max(capacity_violations,default=0.),
            "minimum_operating_prefund_headroom":min(pre_headroom,default=0.),
            "minimum_borrowing_base_headroom":min(base_headroom) if base_headroom else None}
