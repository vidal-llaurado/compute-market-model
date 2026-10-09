"""Joint financial, physical and financing control on observable trees.

Inherited positions settle before node trades and margin calls precede
rebalancing. Leaves close or extinguish positions. All controls and the
rectangular risk recursion share one cash ledger."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,platform
import numpy as np
import scipy
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.sparse import vstack
from .optimizer import LinearModel, SolveResult, OptimizationError, add, compile_model
from .schema import Portfolio,Contract
from .hedge_market import HedgeMarket
from .risk import nested_value

@dataclass
class Affine:
    row: dict
    constant: float=0.
    def __add__(self,other):
        if not isinstance(other,Affine):other=Affine({},float(other))
        r=dict(self.row)
        for i,x in other.row.items():add(r,i,x)
        return Affine(r,self.constant+other.constant)
    __radd__=__add__
    def __neg__(self):return self*(-1)
    def __sub__(self,other):return self+-other
    def __rsub__(self,other):return -self+other
    def __mul__(self,x):return Affine({i:v*x for i,v in self.row.items()},self.constant*x)
    __rmul__=__mul__


def var(i):return Affine({int(i):1.})
def equal(lp,expression,rhs,name):lp.equality(expression.row,rhs-expression.constant,name)
def less(lp,expression,rhs,name):lp.inequality(expression.row,rhs-expression.constant,name)
def interval(lp,a):
    lo=hi=a.constant
    for i,c in a.row.items():
        lo+=c*(lp.lower[i] if c>=0 else lp.upper[i]);hi+=c*(lp.upper[i] if c>=0 else lp.lower[i])
    return float(lo),float(hi)

def positive(lp,a,name, *, upper_bound=None):
    """Exact max(a,0), with interval-derived big-M; binary only across the kink."""
    lo,hi=interval(lp,a)
    if upper_bound is not None:
        hi=min(hi,float(upper_bound))
    if lo>=0:return a
    if hi<=0:return Affine({})
    if not np.isfinite([lo,hi]).all():raise ValueError(f'finite bounds needed for {name}')
    z=lp.var(name,0,hi);binary=lp.var(name+'_sign',0,1,True)
    less(lp,a-var(z),0,name+'_lower')
    less(lp,var(z)-var(binary)*hi,0,name+'_zero_branch')
    less(lp,var(z)-a+var(binary)*(-lo),-lo,name+'_active_branch')
    return var(z)

def absolute(lp,a,name):return positive(lp,a,name)*2-a

def without_static_hedges(p):
    n=p.tree.size
    return p.copy(hedge_names=[],hedge_settlement=np.zeros((n,0)),hedge_initial_price=[],
        hedge_half_spread=[],hedge_gross_limit=[],hedge_margin=np.zeros((n,0)),hedge_terminal_value=np.zeros((n,0)))


def compile_dynamic(p,market,contract=None,premium=0.,mode='dynamic',fixed_policy=None,
                    enforce_incoming_margin=True):
    if mode not in ('dynamic','static','unhedged'):raise ValueError('mode must be dynamic, static, unhedged')
    if p.hedge_names:raise ValueError('clear the root-static payoff vector with without_static_hedges before supplying a dated financial market')
    m=market.validate(p.tree);p0=without_static_hedges(p)
    lp,v,_,c=compile_model(p0,contract,premium)
    ncount,kcount=p.tree.size,len(m.names);gcount=len(m.netting_sets)
    shapes={'position':(ncount,kcount),'buy':(ncount,kcount),'sell':(ncount,kcount),
            'initial_margin':(ncount,kcount),'posted_vm':(ncount,gcount),'received_vm':(ncount,gcount),
            'cash_pretrade':(ncount,),'financial_cash':(ncount,), 'turnover_cost':(ncount,),
            'incoming_settlement':(ncount,), 'credit_loss':(ncount,)}
    for name,shape in shapes.items():
        ids=np.empty(shape,int)
        for ix in np.ndindex(shape):
            lo,hi=0.,np.inf
            if name=='position':
                lo,hi=m.lower[ix],m.upper[ix]
                if not p.tree.children[ix[0]] or mode=='unhedged':lo=hi=0.
            elif name in ('buy','sell'):
                n,k=ix; parent=p.tree.parent[n]
                here_lo,here_hi=m.lower[n,k],m.upper[n,k]
                if not p.tree.children[n] or mode=='unhedged':here_lo=here_hi=0.
                if parent<0:old_lo=old_hi=m.initial_position[k]*m.survives[n,k]
                else:
                    old_lo=m.lower[parent,k]*m.survives[n,k]
                    old_hi=m.upper[parent,k]*m.survives[n,k]
                    if mode=='unhedged':old_lo=old_hi=0.
                # Net position changes bound each direction. Round trips have
                # nonnegative cost and no contractual benefit in this model.
                hi=min(m.trade_limit[ix],max(0.,here_hi-old_lo) if name=='buy' else max(0.,old_hi-here_lo))
            elif name in ('cash_pretrade',):lo=p.minimum_cash[ix[0]] if enforce_incoming_margin else -np.inf
            elif name in ('financial_cash','incoming_settlement'):lo=-np.inf
            elif name=='initial_margin':hi=m.im_per_unit[ix]*max(abs(m.lower[ix]),abs(m.upper[ix]))
            elif name in ('posted_vm','received_vm'):
                n,g=ix;group=m.netting_sets[g].name
                hi=sum(max(abs(m.lower[n,k]),abs(m.upper[n,k]))*abs(m.mark[n,k])*m.multiplier[k]
                       for k in range(kcount) if m.groups[k]==group)
            ids[ix]=lp.var(f'{name}{ix}',lo,hi)
        v[name]=ids
    if fixed_policy is not None:
        fixed_policy=np.asarray(fixed_policy,float)
        if fixed_policy.shape!=(ncount,kcount) or not np.isfinite(fixed_policy).all():
            raise ValueError('fixed_policy must be finite node-by-instrument positions')
        for n in range(ncount):
            for k in range(kcount):equal(lp,var(v['position'][n,k]),fixed_policy[n,k],f'fixed_position[{n},{k}]')
    for n in range(ncount):
        parent=p.tree.parent[n];root=parent<0
        qp=[Affine({},m.initial_position[k]) if root else var(v['position'][parent,k]) for k in range(kcount)]
        oldim=[Affine({},m.initial_im[k]) if root else var(v['initial_margin'][parent,k]) for k in range(kcount)]
        oldposted=[Affine({},m.initial_posted_vm[g]) if root else var(v['posted_vm'][parent,g]) for g in range(gcount)]
        oldrecv=[Affine({},m.initial_received_vm[g]) if root else var(v['received_vm'][parent,g]) for g in range(gcount)]
        settlement=Affine({});credit=Affine({});tradecash=Affine({});cost=Affine({})
        returned=sum((oldim[k]*m.im_return[n,k] for k in range(kcount)),Affine({}))+sum(oldposted,Affine({}))
        preim=Affine({});postim=Affine({});prevm=Affine({});postvm=Affine({})
        for k in range(kcount):
            q=var(v['position'][n,k]);inherited=qp[k]*m.survives[n,k]
            equal(lp,q-inherited-var(v['buy'][n,k])+var(v['sell'][n,k]),0,f'position_balance[{n},{k}]')
            turnover=var(v['buy'][n,k])+var(v['sell'][n,k])
            less(lp,turnover,m.trade_limit[n,k],f'market_depth[{n},{k}]')
            if mode=='static' and not root and p.tree.children[n]:
                equal(lp,q-inherited,0,f'static_position[{n},{k}]')
            absq=absolute(lp,q,f'abs_position[{n},{k}]')
            equal(lp,var(v['initial_margin'][n,k])-m.im_per_unit[n,k]*absq,0,f'initial_margin[{n},{k}]')
            postim+=var(v['initial_margin'][n,k])
            if m.im_per_unit[n,k]>0:
                preim+=m.im_per_unit[n,k]*absolute(lp,inherited,f'pre_abs[{n},{k}]')
            spread=turnover*m.half_spread[n,k]
            if m.impact_slope[n,k]>0:
                spread+=m.impact_slope[n,k]*positive(lp,turnover-m.impact_depth[n,k],f'impact[{n},{k}]',upper_bound=m.trade_limit[n,k]-m.impact_depth[n,k])
            cost+=spread
            if m.kinds[k]=='futures':
                previous=m.initial_mark[k] if root else m.mark[parent,k]
                cash=qp[k]*m.multiplier[k]*(m.mark[n,k]-previous+m.distribution[n,k])
            elif m.kinds[k]=='otc' and m.default[n,[g.name for g in m.netting_sets].index(m.groups[k])]:
                cash=Affine({}) # netted closeout is calculated once below
            else:
                cash=qp[k]*m.multiplier[k]*m.distribution[n,k]
            loss=(1-m.settlement_recovery[n,k])*positive(lp,cash,f'settlement_credit[{n},{k}]') if m.settlement_recovery[n,k]<1 else Affine({})
            settlement+=cash-loss;credit+=loss
            if m.kinds[k]!='futures':tradecash-=(q-inherited)*m.mark[n,k]*m.multiplier[k]
        for g,group in enumerate(m.netting_sets):
            ids=[k for k in range(kcount) if m.groups[k]==group.name]
            exposure=sum((var(v['position'][n,k])*m.mark[n,k]*m.multiplier[k] for k in ids),Affine({}))
            before=sum((qp[k]*m.survives[n,k]*m.mark[n,k]*m.multiplier[k] for k in ids),Affine({}))
            posted=positive(lp,-exposure-group.posting_threshold,f'vm_post[{n},{g}]')
            received=positive(lp,exposure-group.receiving_threshold,f'vm_recv[{n},{g}]')
            equal(lp,var(v['posted_vm'][n,g])-posted,0,f'posted_vm[{n},{g}]')
            equal(lp,var(v['received_vm'][n,g])-received,0,f'received_vm[{n},{g}]')
            prevm+=positive(lp,-before-group.posting_threshold,f'pre_vm[{n},{g}]')
            postvm+=posted
            if m.default[n,g]:
                exposure_at_default=sum((qp[k]*(m.mark[n,k]+m.distribution[n,k])*m.multiplier[k] for k in ids),Affine({}))
                loss=(1-group.recovery)*positive(lp,exposure_at_default-oldrecv[g],f'closeout_loss[{n},{g}]')
                settlement+=exposure_at_default-loss;credit+=loss
        effect=settlement+returned-postim-postvm+tradecash-cost
        preeffect=settlement+returned-preim-prevm
        equal(lp,var(v['financial_cash'][n])-effect,0,f'financial_cash[{n}]')
        equal(lp,var(v['turnover_cost'][n])-cost,0,f'turnover_cost[{n}]')
        equal(lp,var(v['incoming_settlement'][n])-settlement,0,f'incoming_settlement[{n}]')
        equal(lp,var(v['credit_loss'][n])-credit,0,f'credit_loss[{n}]')
        # Modify the common physical cash conservation row exactly once.
        row_index=lp.eq_names.index(f'opening_cash[{n}]')
        add(lp.eq[row_index],v['financial_cash'][n],-1)
        # Same permitted opening cures; margin on inherited positions precedes trades.
        equal(lp,var(v['cash_pretrade'][n])-var(v['cash_open'][n])+effect-preeffect,0,f'pretrade_cash[{n}]')
        for r,name in enumerate(m.factor_names):
            e=sum((var(v['position'][n,k])*m.factor_loadings[n,r,k] for k in range(kcount)),Affine({}))
            less(lp,e,m.exposure_upper[n,r],f'mandate_upper[{n},{name}]')
            less(lp,-e,-m.exposure_lower[n,r],f'mandate_lower[{n},{name}]')
    objective=np.zeros(len(lp.names));objective[v['risk_loss'][0]]=1.
    return lp,v,objective,c,p0


def _run(lp,objective,time_limit):
    ae,au=lp.matrix(lp.eq),lp.matrix(lp.ub);be=np.array(lp.eq_rhs);bu=np.array(lp.ub_rhs)
    def optimize(presolve):
        if any(lp.integral):
            cons=LinearConstraint(vstack([ae,au]).tocsc(),np.r_[be,np.full(len(bu),-np.inf)],np.r_[be,bu])
            return milp(objective,integrality=np.array(lp.integral),bounds=Bounds(lp.lower,lp.upper),constraints=cons,
                   options={'time_limit':float(time_limit),'mip_rel_gap':1e-8,'presolve':presolve})
        return linprog(objective,A_eq=ae,b_eq=be,A_ub=au,b_ub=bu,bounds=list(zip(lp.lower,lp.upper)),
                  method='highs',options={'time_limit':float(time_limit),'primal_feasibility_tolerance':1e-8,
                                          'dual_feasibility_tolerance':1e-8,'presolve':presolve})
    r=optimize(True)
    # HiGHS presolve can conflate infeasibility and unboundedness. Obtain a
    # definitive status instead of treating a numerical/solver error as survival failure.
    if r.status==4 and 'unbounded or infeasible' in str(r.message).lower():
        r=optimize(False)
    return r


def solver_certificate(result, objective_name, lp=None, objective=None):
    """Retain the primary certificate and distinguish LP duals from MIP bounds."""
    def number(name):
        value=getattr(result,name,None)
        return None if value is None or not np.isfinite(value) else float(value)
    is_mip=bool(lp is not None and any(lp.integral))
    certificate={'objective_name':objective_name,'status':int(result.status),
        'success':bool(result.success),'message':str(result.message),'objective':number('fun'),
        'mip_dual_bound':number('mip_dual_bound') if is_mip else None,
        'mip_relative_gap':number('mip_gap') if is_mip else None,
        'mip_node_count':number('mip_node_count') if is_mip else None,
        'iterations':number('nit')}
    if lp is not None and not is_mip and result.success and objective is not None:
        eq=np.asarray(result.eqlin.marginals);ub=np.asarray(result.ineqlin.marginals)
        lower=np.asarray(result.lower.marginals);upper=np.asarray(result.upper.marginals)
        bounds_low=np.asarray(lp.lower);bounds_up=np.asarray(lp.upper)
        ml=np.isfinite(bounds_low);mu=np.isfinite(bounds_up)
        dual=float(np.asarray(lp.eq_rhs)@eq+np.asarray(lp.ub_rhs)@ub+
            bounds_low[ml]@lower[ml]+bounds_up[mu]@upper[mu])
        stationarity=np.asarray(objective)-lp.matrix(lp.eq).T@eq-lp.matrix(lp.ub).T@ub-lower-upper
        certificate.update(lp_dual_objective=dual,
            lp_primal_dual_gap=float(result.fun-dual),
            lp_stationarity_error=float(np.max(np.abs(stationarity),initial=0.)))
    return certificate


def compiled_fingerprint(lp, objective):
    """Hash mathematical arrays rather than policy tie-break allocations."""
    h=hashlib.sha256()
    for matrix in (lp.matrix(lp.eq),lp.matrix(lp.ub)):
        a=matrix.tocsr();a.sort_indices()
        for array in (np.asarray(a.shape,dtype='<i8'),a.indptr.astype('<i8'),a.indices.astype('<i8'),a.data.astype('<f8')):
            h.update(array.tobytes())
    for array in (objective,lp.eq_rhs,lp.ub_rhs,lp.lower,lp.upper,lp.integral):
        h.update(np.asarray(array,dtype='<f8').tobytes())
    return h.hexdigest()


def solve_dynamic(p: Portfolio,market: HedgeMarket,contract: Contract|None=None,premium:float=0.,
                  mode='dynamic',fixed_policy=None,time_limit=60.,enforce_incoming_margin=True,
                  tie_break=True) -> SolveResult:
    if not np.isfinite(time_limit) or time_limit<=0:raise ValueError('positive time limit required')
    lp,v,obj,c,p0=compile_dynamic(p,market,contract,premium,mode,fixed_policy,enforce_incoming_margin)
    result=_run(lp,obj,time_limit)
    provenance={'project':'Compute Market','solver':'HiGHS MILP' if any(lp.integral) else 'HiGHS LP',
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'hedge_mode':mode,'premium':float(premium),'incoming_margin_before_trades':enforce_incoming_margin,
        'input_sha256':hashlib.sha256(json.dumps({'portfolio':p.to_dict(),'market':market.to_dict(),
          'contract':c.to_dict(),'premium':premium,'mode':mode,'fixed':None if fixed_policy is None else np.asarray(fixed_policy).tolist()},sort_keys=True).encode()).hexdigest()}
    provenance['primary_certificate']=solver_certificate(result,'minimum nested loss',lp,obj)
    provenance['compiled_sha256']=compiled_fingerprint(lp,obj)
    provenance['compiled_model']={'class':'MILP' if any(lp.integral) else 'LP',
        'physical_blocks_integral':bool(p.integer_blocks),
        'integer_variables':int(sum(lp.integral)),
        'financial_sign_binaries':sum(int(flag) for name,flag in zip(lp.names,lp.integral) if name.endswith('_sign')),
        'variables':len(lp.names),'equalities':len(lp.eq),'inequalities':len(lp.ub)}
    if result.status==2:return SolveResult('infeasible',None,str(result.message),provenance=provenance)
    if not result.success:raise OptimizationError(f'dynamic solve status {result.status}: {result.message}')
    optimum=float(result.fun)
    if tie_break:
        lp.inequality({v['risk_loss'][0]:1},optimum+max(1e-7,abs(optimum)*1e-10),'primary_optimum')
        secondary=np.zeros(len(lp.names))
        for key in ('buy','sell'):
            secondary[v[key].ravel()]=1.
        # Tiny secondary lockup preference resolves redundant trades without changing the objective.
        for key in ('initial_margin','posted_vm'):
            secondary[v[key].ravel()]=1e-7
        second=_run(lp,secondary,time_limit)
        provenance['secondary_certificate']=solver_certificate(second,'minimum turnover and lockup',lp,secondary)
        if second.success:result=second
    x=result.x
    provenance['returned_primary_loss']=float(obj@x)
    provenance['primary_tie_break_slack']=float(obj@x-optimum)
    policy={name:x[idx].tolist() for name,idx in v.items()}
    policy['hedge_net']=policy['position']
    ae,au=lp.matrix(lp.eq),lp.matrix(lp.ub);be=np.array(lp.eq_rhs);bu=np.array(lp.ub_rhs)
    eq=float(np.max(np.abs(ae@x-be),initial=0));ub=float(np.max(au@x-bu,initial=0))
    bound=max(float(np.max(np.asarray(lp.lower)-x,initial=0)),float(np.max(x-np.asarray(lp.upper),initial=0)))
    from .hedge_audit import audit_dynamic
    audit=audit_dynamic(p0,market,c,policy,premium)
    audit.update(max_equality_residual=eq,max_inequality_violation=ub,max_bound_violation=bound,
         variables=len(lp.names),constraints=len(lp.eq)+len(lp.ub),integer_variables=int(sum(lp.integral)),
         node_count=p.tree.size,leaf_count=len(p.tree.leaves),primary_objective_loss=optimum)
    wealth=np.zeros(p.tree.size);wealth[p.tree.leaves]=-x[v['risk_loss'][p.tree.leaves]]
    if np.any(p.ambiguity_radius):
        from .ambiguity import robust_nested_value
        values=robust_nested_value(p.tree,wealth,p.alpha,p.risk_weight,p.discount,p.ambiguity_radius)
    else:values=nested_value(p.tree,wealth,p.alpha,p.risk_weight,p.discount)
    audit['risk_recursion_error']=abs(values[0]+x[v['risk_loss'][0]])
    audit['expected_terminal_wealth']=float(p.tree.unconditional[p.tree.leaves]@wealth[p.tree.leaves])
    audit['minimum_terminal_wealth']=float(wealth[p.tree.leaves].min())
    errors=[val for key,val in audit.items() if ('error' in key or 'violation' in key or 'residual' in key) and val is not None]
    if max(errors,default=0)>max(1e-5,abs(optimum)*3e-7):raise OptimizationError(f'dynamic verification failed: {audit}')
    return SolveResult('optimal',float(values[0]),str(result.message),policy,audit,{},provenance)
