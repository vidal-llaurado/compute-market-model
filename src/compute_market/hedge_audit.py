"""Independent settlement, collateral, turnover and physical-cash replay."""
from __future__ import annotations
import numpy as np
from .validation import independent_policy_audit


def financial_replay(tree,m,policy):
    a={k:np.asarray(v,float) for k,v in policy.items()};q=a['position']
    n,k=q.shape;g=len(m.netting_sets)
    post=np.zeros((n,g));recv=post.copy();im=np.abs(q)*m.im_per_unit
    flow=np.zeros(n);pre=np.zeros(n);income=np.zeros(n);losses=np.zeros(n);cost=np.zeros(n)
    inventory=np.zeros((n,k));vm_expected=np.zeros((n,g))
    for node in range(n):
        parent=tree.parent[node];root=parent<0
        old=m.initial_position if root else q[parent]
        oldim=m.initial_im if root else im[parent]
        oldpost=m.initial_posted_vm if root else post[parent]
        oldrecv=m.initial_received_vm if root else recv[parent]
        returned=float(oldim@m.im_return[node]+oldpost.sum())
        preim=float(np.abs(old*m.survives[node])@m.im_per_unit[node]);prevm=0.
        trade=q[node]-old*m.survives[node]
        inventory[node]=trade-a['buy'][node]+a['sell'][node]
        turnover=a['buy'][node]+a['sell'][node]
        cost[node]=float(turnover@m.half_spread[node]+np.maximum(turnover-m.impact_depth[node],0)@m.impact_slope[node])
        tradecash=0.
        for j in range(k):
            if m.kinds[j]=='futures':
                prev=m.initial_mark[j] if root else m.mark[parent,j]
                value=old[j]*(m.mark[node,j]-prev+m.distribution[node,j])*m.multiplier[j]
            elif m.kinds[j]=='otc' and m.default[node,[x.name for x in m.netting_sets].index(m.groups[j])]:value=0.
            else:value=old[j]*m.distribution[node,j]*m.multiplier[j]
            loss=(1-m.settlement_recovery[node,j])*max(value,0.)
            income[node]+=value-loss;losses[node]+=loss
            if m.kinds[j]!='futures':tradecash-=trade[j]*m.mark[node,j]*m.multiplier[j]
        for gi,group in enumerate(m.netting_sets):
            ids=np.array([j for j in range(k) if m.groups[j]==group.name],int)
            e=float((q[node,ids]*m.mark[node,ids]*m.multiplier[ids]).sum())
            ep=float((old[ids]*m.survives[node,ids]*m.mark[node,ids]*m.multiplier[ids]).sum())
            post[node,gi]=max(-e-group.posting_threshold,0.)
            recv[node,gi]=max(e-group.receiving_threshold,0.)
            prevm+=max(-ep-group.posting_threshold,0.)
            if m.default[node,gi]:
                ed=float((old[ids]*(m.mark[node,ids]+m.distribution[node,ids])*m.multiplier[ids]).sum())
                lgd=(1-group.recovery)*max(ed-oldrecv[gi],0.)
                income[node]+=ed-lgd;losses[node]+=lgd
        flow[node]=income[node]+returned-im[node].sum()-post[node].sum()+tradecash-cost[node]
        pre[node]=income[node]+returned-preim-prevm
    return {'financial_cash':flow,'pretrade_financial_cash':pre,'incoming_settlement':income,'credit_loss':losses,
            'turnover_cost':cost,'initial_margin':im,'posted_vm':post,'received_vm':recv,'position_balance':inventory}


def audit_dynamic(p,m,c,policy,premium=0.):
    a={k:np.asarray(v,float) for k,v in policy.items()};r=financial_replay(p.tree,m,policy)
    ordinary={key:value for key,value in policy.items()}
    ordinary['hedge_long']=[];ordinary['hedge_short']=[]
    adjusted=p.copy(opening_cashflow=p.opening_cashflow+r['financial_cash'])
    report=independent_policy_audit(adjusted,c,ordinary,premium)
    for name in ('financial_cash','incoming_settlement','credit_loss','turnover_cost','initial_margin','posted_vm','received_vm'):
        report['independent_'+name+'_error']=float(np.max(np.abs(a[name]-r[name]),initial=0))
    pred=a['cash_open']-r['financial_cash']+r['pretrade_financial_cash']
    report['independent_pretrade_cash_error']=float(np.max(np.abs(pred-a['cash_pretrade']),initial=0))
    report['independent_position_balance_error']=float(np.max(np.abs(r['position_balance']),initial=0))
    report['market_depth_violation']=float(np.max(a['buy']+a['sell']-m.trade_limit,initial=0))
    report['minimum_pretrade_cash']=float(a['cash_pretrade'].min())
    report['maximum_posted_collateral']=float(np.max(r['initial_margin'].sum(axis=1)+r['posted_vm'].sum(axis=1)))
    report['total_probability_weighted_turnover_cost']=float(p.tree.unconditional@r['turnover_cost'])
    report['terminal_position_error']=float(np.max(np.abs(a['position'][p.tree.leaves]),initial=0))
    return report
