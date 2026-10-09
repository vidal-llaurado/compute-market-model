"""Conditional-tail hedge control under cash, position and depth constraints.

A one-step program reoptimizes against a marked continuation. The default
mandate rewards risk reduction and charges adverse carry. Forecast gains are
centered in the objective; actual cash settlements retain their full value."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr

MULTIPLIER=np.array([1000.,1000.,50.,100000.,10000.,1000.])
INSTRUMENTS=['G1 future','G2 future','power future','rate future','FX future','G1 put']

@dataclass
class ConditionalKernel:
    coefficient: np.ndarray
    residuals: np.ndarray
    scale: np.ndarray
    observations: int
    scenarios: int
    seed: int
    def forecast(self,state):
        x=np.asarray(state,float)
        if x.shape!=(self.coefficient.shape[0]-1,) or not np.isfinite(x).all():raise ValueError('finite observed state required')
        z=np.r_[1.,x/self.scale]@self.coefficient
        return z+self.residuals
    def to_dict(self):
        return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(self).items()}
    @classmethod
    def from_dict(cls,d):
        d=dict(d)
        for k in ('coefficient','residuals','scale'):d[k]=np.asarray(d[k],float)
        return cls(**d)


def fit_kernel(states,scenarios=64,seed=731,ridge=1e-4):
    a=np.asarray(states,float)
    if a.ndim!=3 or min(a.shape[:2])<2 or not np.isfinite(a).all() or scenarios<4:raise ValueError('finite path-by-date-by-state training observations required')
    x=a[:,:-1].reshape(-1,a.shape[2]);y=a[:,1:].reshape(-1,a.shape[2])
    scale=np.maximum(x.std(axis=0),1e-4);design=np.column_stack([np.ones(len(x)),x/scale])
    pen=np.eye(design.shape[1])*ridge;pen[0,0]=0
    coef=np.linalg.solve(design.T@design+pen,design.T@y)
    res=y-design@coef
    rng=np.random.default_rng(seed);sample=res[rng.integers(0,len(res),scenarios)].copy()
    sample-=sample.mean(axis=0) # empirical innovation shape, zero conditional innovation mean
    return ConditionalKernel(coef,sample,scale,len(x),scenarios,seed)


def put_quote(forward, strike, remaining_years):
    """Undiscounted funded put quote on a dated benchmark in the synthetic market."""
    f=np.asarray(forward,float)
    if np.any(f<=0) or not np.isfinite(f).all() or strike<=0 or remaining_years<0:
        raise ValueError('positive forward/strike and nonnegative remaining time required')
    vol=.35*np.sqrt(max((1-np.exp(-1.6*remaining_years))/1.6,0.))
    if vol<1e-12:return np.maximum(strike-f,0.)
    d1=np.log(f/strike)/vol+.5*vol;d2=d1-vol
    return strike*ndtr(-d2)-f*ndtr(-d1)


def quoted_marks(state,remaining,total_steps,dt):
    """Synthetic dated curves and a Black option quote surface.

    Maturity anchors and loadings are specified independently of debt coupons."""
    x=np.asarray(state,float);scalar=x.ndim==1;x=np.atleast_2d(x)
    if x.shape[1]!=11 or not np.isfinite(x).all() or remaining<0:raise ValueError('eleven finite factors and nonnegative remaining dates required')
    out=np.zeros((len(x),6));tau=remaining*dt
    anchor=np.array([2.,3.,70.])*np.exp(np.array([-.08,-.08,0.])*total_steps*dt)
    persistence=np.exp(-np.array([.8,.65,1.6])*tau)
    out[:,:3]=anchor*np.exp(np.clip(x[:,:3]*persistence,-5,5))
    out[:,3]=.04+np.exp(-1.*tau)*(x[:,3]-.04)
    out[:,4]=1.3*np.exp(np.clip(np.exp(-.5*tau)*x[:,4],-5,5))
    out[:,5]=put_quote(out[:,0],anchor[0],tau)
    return out[0] if scalar else out


def physical_observations(state):
    x=np.atleast_2d(np.asarray(state,float))
    g1=2*np.exp(np.clip(x[:,0],-5,5));g2=3*np.exp(np.clip(x[:,1],-5,5))
    local=np.column_stack([g1,g1,g2,g2])*np.exp(np.clip(x[:,6:10],-5,5))
    power=70*np.exp(np.clip(x[:,2],-5,5));fx=1.3*np.exp(np.clip(x[:,4],-5,5))
    hw=np.exp(np.clip(x[:,5],-5,5));demand=np.expm1(np.clip(x[:,10],0,3))
    return local,power,fx,hw,demand


def planned_value(state,holdings,debt,term,remaining,config,availability=None,customer_alive=True):
    """Marked continuation of a feasible fleet plan, recomputed under each state.

    Hardware exit value, floating rentals, contractual receipts, volume caps,
    replacement costs, electricity, FX operating costs and rate-sensitive debt
    service all enter the hedge exposure. This is the local execution target;
    the finite-tree solver separately optimizes the full continuation policy.
    """
    x=np.atleast_2d(np.asarray(state,float));local,power,fx,hw,demand=physical_observations(x)
    h=np.asarray(holdings,float);firm=np.asarray(config.firm_blocks);gh=np.asarray(config.gpus_per_block)*config.hours_per_step
    avail=np.ones(4) if availability is None else np.asarray(availability,float)
    owned=np.minimum(h*avail,firm);replacement=np.maximum(firm-owned,0)
    floating=np.maximum(h*avail-firm,0)[None,:]*np.minimum(demand[:,None],1.)
    cost=np.array(config.nonpower_unit_cost)[None,:]*(fx[:,None]/1.3)+np.array(config.mwh_per_gpu_hour)[None,:]*power[:,None]
    floating=np.where(local>cost,floating,0)
    cash=(floating*(local-cost)-owned[None,:]*cost-replacement[None,:]*local*config.replacement_multiplier)@gh
    cash+=float(gh@(firm*np.array(config.firm_rates)))*float(customer_alive)-config.fixed_overhead_per_step
    cash-=(debt*(np.maximum(x[:,3],-.02)+config.revolver_spread)+term*config.fixed_term_coupon)*config.year_fraction
    asset=(h@np.asarray(config.hardware_values))*hw*np.exp(-config.asset_decline_annual*remaining*config.year_fraction)
    asset*=config.liquidation_fraction-config.liquidation_impact
    # Terminal-wealth units: cash is retained without an invented reinvestment rate.
    return remaining*cash+asset-debt-term


@dataclass
class LinearControls:
    """Physical/financing actions selected in the same LP as hedge positions.

    ``loss_loadings[s, j]`` is the change in scenario loss per action unit.
    ``cash_use[j]`` is current unrestricted cash consumed (negative for a draw).
    All actions are observable-date decisions shared across conditional atoms.
    """
    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    cash_use: np.ndarray
    loss_loadings: np.ndarray


def solve_conditional(losses,gains,old,marks,im,spread,impact,depth,trade_limit,lower,upper,
                      cash_budget,alpha=.9,risk_weight=.65,center_gains=True,funding_rate=.01,
                      funded_prices=None, *, carry_convention=None, controls=None,
                      factor_loadings=None, exposure_lower=None, exposure_upper=None,
                      premium_liquidity_rate=0.):
    """Joint mean/ES LP with execution, cash, carry and optional physical actions.

    Carry choices: ``adverse_only`` centers gains and charges negative means;
    ``centered`` ignores means; ``total_return`` uses uncentered gains; ``symmetric``
    centers gains and charges the absolute expected carry of each position.
    Funding rates are nonnegative *per-interval* cash opportunity charges. Actual
    debt interest is a separate scenario cash flow, not this objective charge.
    """
    from .risk import expected_shortfall
    from scipy.sparse import csr_matrix
    loss=np.asarray(losses,float);raw=np.asarray(gains,float);old=np.asarray(old,float)
    if loss.ndim!=1 or raw.shape!=(len(loss),len(old)) or len(loss)<2:
        raise ValueError('matching scenario loss and gain arrays required')
    k=len(old);s=len(loss)
    vectors=[np.asarray(a,float) for a in (marks,im,spread,impact,depth,trade_limit,lower,upper)]
    marks,im,spread,impact,depth,trade_limit,lower,upper=vectors
    if any(a.shape!=(k,) or not np.isfinite(a).all() for a in vectors) or not np.isfinite(loss).all() or not np.isfinite(raw).all() or not np.isfinite(old).all():
        raise ValueError('finite consistent hedge inputs required')
    if np.any(lower>upper) or np.any(np.minimum.reduce([im,spread,impact,depth,trade_limit])<0) or not np.isfinite(cash_budget) or not 0<=alpha<1 or not 0<=risk_weight<=1:
        raise ValueError('invalid execution constraints')
    carry=carry_convention or ('adverse_only' if center_gains else 'total_return')
    if carry not in ('adverse_only','centered','total_return','symmetric'):
        raise ValueError('unknown carry convention')
    f=np.broadcast_to(np.asarray(funding_rate,float),(k,)).copy()
    pf=np.broadcast_to(np.asarray(premium_liquidity_rate,float),(k,)).copy()
    if not np.isfinite(f).all() or not np.isfinite(pf).all() or np.any(f<0) or np.any(pf<0):
        raise ValueError('finite nonnegative per-interval liquidity charges required')
    funded=np.zeros(k) if funded_prices is None else np.asarray(funded_prices,float)
    if funded.shape!=(k,) or not np.isfinite(funded).all():raise ValueError('finite funded trade prices required')
    if controls is None:
        controls=LinearControls([],np.zeros(0),np.zeros(0),np.zeros(0),np.zeros((s,0)))
    controls=LinearControls(controls.names,*[np.asarray(a,float) for a in
        (controls.lower,controls.upper,controls.cash_use,controls.loss_loadings)])
    j=len(controls.names)
    if len(set(controls.names))!=j or any(a.shape!=(j,) for a in (controls.lower,controls.upper,controls.cash_use)) or controls.loss_loadings.shape!=(s,j):
        raise ValueError('consistent, uniquely named joint controls required')
    if any(not np.isfinite(a).all() for a in (controls.lower,controls.upper,controls.cash_use,controls.loss_loadings)) or np.any(controls.lower>controls.upper):
        raise ValueError('finite bounded joint controls required')
    mu=raw.mean(axis=0);H=raw if carry=='total_return' else raw-mu
    q=np.arange(k);ab=q+k;turn=q+2*k;extra=q+3*k;z=4*k;ex=np.arange(s)+z+1
    mc=np.arange(k)+4*k+1+s;cv=np.arange(j)+5*k+1+s;nv=5*k+1+s+j
    obj=np.zeros(nv);obj[q]=-(1-risk_weight)*H.mean(axis=0)
    lockcharge=f*im+pf*np.maximum(funded,0)
    obj[ab]=lockcharge+1e-9;obj[turn]=spread+1e-9;obj[extra]=impact
    obj[z]=risk_weight;obj[ex]=risk_weight/(s*(1-alpha))
    obj[cv]=(1-risk_weight)*controls.loss_loadings.mean(axis=0)
    if carry in ('adverse_only','symmetric'):obj[mc]=1.
    ii=[];jj=[];vv=[];rhs=[]
    def put(entries,b):
        row=len(rhs)
        for col,value in entries.items():
            if value:ii.append(row);jj.append(int(col));vv.append(float(value))
        rhs.append(float(b))
    for a in range(k):
        put({q[a]:1,ab[a]:-1},0);put({q[a]:-1,ab[a]:-1},0)
        put({q[a]:1,turn[a]:-1},old[a]);put({q[a]:-1,turn[a]:-1},-old[a])
        put({turn[a]:1,extra[a]:-1},depth[a])
        if carry in ('adverse_only','symmetric'):put({q[a]:-mu[a],mc[a]:-1},0)
        if carry=='symmetric':put({q[a]:mu[a],mc[a]:-1},0)
    budget={}
    for ids,values in ((ab,im),(turn,spread),(extra,impact),(q,funded),(cv,controls.cash_use)):
        for index,value in zip(ids,values):budget[int(index)]=float(value)
    put(budget,cash_budget+funded@old)
    for row in range(s):
        entries={int(a):-float(b) for a,b in zip(q,H[row])}
        entries.update({int(a):float(b) for a,b in zip(cv,controls.loss_loadings[row])})
        entries[z]=-1;entries[int(ex[row])]=-1;put(entries,-loss[row])
    if factor_loadings is not None:
        load=np.asarray(factor_loadings,float);elo=np.asarray(exposure_lower,float);ehi=np.asarray(exposure_upper,float)
        if load.ndim!=2 or load.shape[1]!=k or elo.shape!=(len(load),) or ehi.shape!=elo.shape or not all(np.isfinite(x).all() for x in (load,elo,ehi)) or np.any(elo>ehi):
            raise ValueError('finite factor mandate dimensions required')
        for row,lo,hi in zip(load,elo,ehi):
            put({int(a):b for a,b in zip(q,row)},hi)
            put({int(a):-b for a,b in zip(q,row)},-lo)
    bounds=list(zip(lower,upper))+[(0,max(abs(lower[a]),abs(upper[a]))) for a in range(k)]
    bounds += [(0,trade_limit[a]) for a in range(k)]+[(0,trade_limit[a]) for a in range(k)]+[(None,None)]+[(0,None)]*s+[(0,None)]*k+list(zip(controls.lower,controls.upper))
    A=csr_matrix((vv,(ii,jj)),shape=(len(rhs),nv))
    result=linprog(obj,A_ub=A,b_ub=np.asarray(rhs),bounds=bounds,method='highs')
    if result.status==2:return {'status':'infeasible','position':None}
    if not result.success:raise RuntimeError(f'conditional joint solve failed: {result.message}')
    position=result.x[q];actions=result.x[cv];actualturn=np.abs(position-old)
    tc=float(actualturn@spread+np.maximum(actualturn-depth,0)@impact)
    hedge_cash=float(np.abs(position)@im+tc+(position-old)@funded)
    cash_used=hedge_cash+float(controls.cash_use@actions)
    if cash_used>cash_budget+1e-5 or np.any(actualturn>trade_limit+1e-7):raise RuntimeError('execution replay failed')
    residual=loss+controls.loss_loadings@actions-H@position
    carrier=float(np.maximum(-mu*position,0).sum()) if carry=='adverse_only' else float(np.abs(mu*position).sum()) if carry=='symmetric' else 0.
    liquidity=float(lockcharge@np.abs(position))
    value=(1-risk_weight)*residual.mean()+risk_weight*expected_shortfall(residual,np.full(s,1/s),alpha)+tc+liquidity+carrier
    return {'status':'optimal','position':position,'controls':dict(zip(controls.names,actions.tolist())),
        'transaction_cost':tc,'cash_used':float(cash_used),'hedge_cash_used':hedge_cash,
        'conditional_objective':float(value),'solver_objective':float(result.fun+(1-risk_weight)*loss.mean()),
        'centered_financial_gains':carry!='total_return','carry_convention':carry,
        'adverse_expected_carry':float(np.maximum(-mu*position,0).sum()) if carry=='adverse_only' else 0.,
        'carry_charge':carrier,'liquidity_charge':liquidity,'expected_financial_gain':float(mu@position),
        'max_constraint_violation':float(np.max(A@result.x-rhs,initial=0))}
