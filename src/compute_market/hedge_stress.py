"""Joint hedge-market stress, frozen commitments and reverse financing stress."""
from __future__ import annotations
import numpy as np
from .dynamic_hedging import solve_dynamic,without_static_hedges
from .hedge_audit import financial_replay
from .risk import nested_value


def tree_stress(p,m,name,severity=1.):
    if severity<0 or not np.isfinite(severity):raise ValueError('nonnegative stress severity required')
    p=p.copy();m=m.copy(p.tree);stage=np.asarray(p.tree.stage);after=stage>=1
    if name in ('basis','compound'):
        p=p.copy(spot_price=p.spot_price*np.where(after[:,None],max(.05,1-.35*severity),1.))
    if name in ('margin','compound'):m.im_per_unit[after]*=1+3*severity
    if name in ('depth','compound'):m.trade_limit[after]*=max(0,1-severity)
    if name=='spread':m.half_spread[after]*=1+9*severity;m.impact_slope[after]*=1+9*severity
    if name=='limit':m.lower[after]*=max(0,1-.75*severity);m.upper[after]*=max(0,1-.75*severity)
    if name in ('collateral','compound'):
        values=p.asset_eligible_value.copy();values[after]*=max(.05,1-.35*severity)
        draws=p.draw_limit.copy();draws[after]*=max(0,1-severity)
        p=p.copy(asset_eligible_value=values,draw_limit=draws)
    if name=='curve_dislocation':
        # Opposite shocks to near and terminal compute quotes: a calendar-basis event.
        m.mark[after,0]*=1+.30*severity;m.mark[after,1]*=max(.05,1-.20*severity)
    if name=='OTC_default':
        for g in range(len(m.netting_sets)):
            ids=[k for k,v in enumerate(m.groups) if v==m.netting_sets[g].name]
            for node in range(1,p.tree.size):
                if stage[node]==1:m.default[node,g]=1
                m.survives[node,ids]=0;m.lower[node,ids]=0;m.upper[node,ids]=0
                if stage[node]>1:m.distribution[node,ids]=0
    if name=='power_rate':
        p=p.copy(owned_unit_cost=p.owned_unit_cost+np.where(after[:,None],.3*severity,0),
                 facility_rate=p.facility_rate+np.where(after[:,None],.04*severity,0))
        m.mark[after,3]*=1+severity
    if name not in ('basis','margin','depth','spread','limit','collateral','curve_dislocation','OTC_default','power_rate','compound'):
        raise ValueError('unknown tree stress')
    return p,m.validate(p.tree)


def frozen_replay(p,m,c,base,premium=0.):
    """Replay intended controls under new prices and flag infeasible execution.

    Cash follows intended controls to locate violations. A depth violation is a
    diagnostic, not an assumed fill."""
    a={k:np.asarray(v,float).copy() for k,v in base.policy.items()}
    ncount=p.tree.size;financial=financial_replay(p.tree,m,a)
    cash_open=np.zeros(ncount);cash_close=np.zeros(ncount);pre=np.zeros(ncount)
    minbb=np.inf;minprefund=np.inf;position_error=0.
    for n in range(ncount):
        parent=p.tree.parent[n];root=parent<0
        cash=p.initial_cash if root else cash_close[parent]
        olddebt=p.initial_debt if root else a['debt'][parent]
        flow=(p.opening_cashflow[n]+c.opening_receipts[n]+a['asset_sales'][n]@p.asset_sale_price[n]
            +a['draw'][n].sum()-a['repay'][n].sum()-p.debt_transaction_fee@(a['draw'][n]+a['repay'][n])
            -(p.facility_rate[n]*p.year_fraction[n])@olddebt)
        cash_open[n]=cash*p.cash_growth[n]+flow+financial['financial_cash'][n]+(premium if root else 0.)
        pre[n]=cash_open[n]-financial['financial_cash'][n]+financial['pretrade_financial_cash'][n]
        gh=p.gpus_per_block*p.hours[n]
        cost=float(gh@(p.owned_unit_cost[n]*(a['spot_blocks'][n]+a['owned_delivery'][n])+p.replacement_price[n]*a['replacement'][n])+c.shortfall_cost[n]@a['shortfall'][n])
        revenue=float(gh@(p.spot_price[n]*a['spot_blocks'][n]))
        cash_close[n]=cash_open[n]-cost+revenue+p.closing_cashflow[n]+c.closing_receipts[n]-(p.facility_rate[n]*p.operating_year_fraction[n])@a['debt'][n]
        minprefund=min(minprefund,cash_open[n]-cost-p.minimum_cash[n])
        bb=(p.advance_rate*p.pledge_fraction*p.asset_eligible_value[n][None,:])@a['holdings'][n]+p.receivable_support[n]
        if len(bb):minbb=min(minbb,float(np.min(bb-a['debt'][n])))
        position_error=max(position_error,float(np.max(m.lower[n]-a['position'][n],initial=0)),float(np.max(a['position'][n]-m.upper[n],initial=0)))
    residual_position=float(np.max(np.abs(financial['position_balance']),initial=0))
    depth_error=float(np.max(a['buy']+a['sell']-m.trade_limit,initial=0))
    violated=minprefund<-1e-5 or np.min(pre-p.minimum_cash)<-1e-5 or minbb<-1e-5 or max(position_error,depth_error,residual_position)>1e-5
    return {'executable':not violated,'minimum_prefunding_headroom':float(minprefund),
       'minimum_incoming_margin_headroom':float(np.min(pre-p.minimum_cash)),
       'minimum_borrowing_base_headroom':None if np.isinf(minbb) else float(minbb),
       'position_limit_violation':position_error,'turnover_depth_violation':depth_error,'survival_balance_violation':residual_position,
       'interpretation':'Frozen intended controls are tested against stressed equations; failed execution tests are not filled trades.'}


def tree_stress_suite(p,m,book,base=None):
    base=base or solve_dynamic(p,m,book)
    rows=[]
    for name in ('basis','margin','depth','spread','limit','collateral','curve_dislocation','OTC_default','power_rate','compound'):
        ps,ms=tree_stress(p,m,name)
        full=solve_dynamic(ps,ms,book)
        fixed=solve_dynamic(ps,ms,book,fixed_policy=base.policy['position'])
        lo=ms.lower.copy();hi=ms.upper.copy();root=np.asarray(base.policy['position'])[0]
        root_allowed=bool(np.all(root>=lo[0]) and np.all(root<=hi[0]))
        if root_allowed:
            lo[0]=hi[0]=root
            rootlocked=solve_dynamic(ps,ms.copy(ps.tree,lower=lo,upper=hi),book)
            rootresult={'status':rootlocked.status,'value':rootlocked.value}
        else:rootresult={'status':'infeasible','value':None}
        rows.append({'stress':name,'reoptimized':{'status':full.status,'value':full.value},
          'root_locked':rootresult,'financial_policy_locked':{'status':fixed.status,'value':fixed.value},
          'frozen_controls':frozen_replay(ps,ms,book,base)})
    return rows


def reverse_margin_stress(p,m,book,base=None,maximum=20.,tolerance=.05):
    """First infeasible incoming-IM multiplier along a monotone fixed-policy ray."""
    base=base or solve_dynamic(p,m,book);position=base.policy['position'];stage=np.asarray(p.tree.stage)
    def run(mult):
        im=m.im_per_unit.copy();im[stage>0]*=mult
        return solve_dynamic(p,m.copy(p.tree,im_per_unit=im),book,fixed_policy=position,tie_break=False)
    lo,hi=1.,float(maximum);high=run(hi)
    if high.status=='optimal':return {'crossing_found':False,'feasible_through':hi}
    count=1
    while hi-lo>tolerance:
        mid=(lo+hi)/2;r=run(mid);count+=1
        if r.status=='optimal':lo=mid
        else:hi=mid
    return {'crossing_found':True,'last_feasible_multiplier':lo,'first_infeasible_multiplier':hi,
            'evaluations':count,'policy':'fixed node-indexed financial policy; physical and financing recourse allowed'}


def instrument_ablation(p,m,book,base=None):
    base=base or solve_dynamic(p,m,book);result=[]
    for k,name in enumerate(m.names):
        lo=m.lower.copy();hi=m.upper.copy();lo[:,k]=0;hi[:,k]=0
        r=solve_dynamic(p,m.copy(p.tree,lower=lo,upper=hi),book)
        result.append({'instrument':name,'status_without':r.status,'value_without':r.value,
                       'marginal_inclusion_value':None if r.value is None else base.value-r.value})
    return result
