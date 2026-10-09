"""Causal joint dispatch, debt and hedge execution on stochastic curves."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .portfolio import spot_state,hardware_marks,CurveCase,EconomyTape
from .continuation import ContinuationSurface
from ..simulation import _repair_one,sale_proceeds
from ..risk_execution import LinearControls,solve_conditional
from ..risk import expected_shortfall
from .settlement import reference_model, marks as claim_marks, funded_prices, settle, conditional_gains


@dataclass
class ConditionalInnovations:
    """Realized conditional factor draws shared by the policy comparisons."""
    values: np.ndarray
    model_sha256: str
    step_years: float
    seed: int

    @classmethod
    def generate(cls, model, periods, scenarios, step_years, seed):
        rng = np.random.default_rng(seed)
        values = np.array([model.step(np.zeros((scenarios, model.dimension)), step_years, rng, 'P')[0]
                           for _ in range(periods)])
        return cls(values, model.fingerprint, step_years, seed)

    @property
    def fingerprint(self):
        return hashlib.sha256(np.asarray(self.values, dtype='<f8').tobytes()).hexdigest()

    def save(self, path):
        metadata = {'model_sha256': self.model_sha256, 'step_years': self.step_years,
                    'seed': self.seed, 'values_sha256': self.fingerprint}
        np.savez_compressed(path, values=self.values, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            expected = metadata.pop('values_sha256')
            result = cls(saved['values'], **metadata)
        if result.fingerprint != expected:
            raise ValueError('conditional innovation checksum mismatch')
        return result


@dataclass
class ControlResult:
    policy: str
    terminal_wealth: np.ndarray
    failed: np.ndarray
    failure_step: np.ndarray
    delivery_shortfall: np.ndarray
    maximum_margin: np.ndarray
    minimum_cash: np.ndarray
    maximum_debt: np.ndarray
    asset_sales: np.ndarray
    transaction_cost: np.ndarray
    positions: np.ndarray
    posted_margin: np.ndarray
    cash: np.ndarray
    debt: np.ndarray
    components: np.ndarray
    audit: dict
    def save(self,path):
        import json
        a={k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        a['metadata']=np.array(json.dumps({'policy':self.policy,'audit':self.audit}));np.savez_compressed(path,**a)
    def summary(self,alpha=.95):
        l=-self.terminal_wealth;p=np.full(len(l),1/len(l))
        required=self.audit['required_delivery_gpu_hours']
        return {'policy':self.policy,'paths':len(l),'mean_wealth':float(-l.mean()),'loss_es':float(expected_shortfall(l,p,alpha)),
          'loss_var':float(np.quantile(l,alpha)), 'wealth_variance':float(np.var(l)), 'failure_fraction':float(self.failed.mean()),
          'mean_delivery_shortfall_gpu_hours':float(self.delivery_shortfall.mean()),'margin_p95':float(np.quantile(self.maximum_margin,.95)),
          'mean_delivery_shortfall_fraction':float(self.delivery_shortfall.mean()/required) if required else 0.,
          'mean_margin_cash_years':float(self.posted_margin[:,:-1].sum(axis=1).mean()*self.audit['interval_years']),
          'mean_interest_cost':float(-self.components[:,1].mean()),
          'debt_p95':float(np.quantile(self.maximum_debt,.95)),'mean_turnover_cost':float(self.transaction_cost.mean()),
          'tail_mass_observations':(1-alpha)*len(l),'confidence':alpha,'audit':self.audit}


def _observed_marks(case,t,x,benchmark_basis):
    return claim_marks(case.instruments,reference_model(case.model,benchmark_basis),t,x)


def run_controller(case:CurveCase,tape:EconomyTape,surface:ContinuationSurface,*,policy='adaptive',scenarios=256,
                   alpha=.9,risk_weight=.55,seed=938,carry='adverse_only',policy_model=None,innovations=None):
    """Jointly choose current dispatch, financing and financial positions.

    Settle inherited marks and margin before rebalancing. Conditional innovations
    come from the affine factor law."""
    if policy not in ('unhedged','static','adaptive') or scenarios<16:raise ValueError('unknown policy or insufficient conditional atoms')
    m=case.model;pm=policy_model or m;c=case.fleet;X=tape.curve.factors;dates=tape.curve.times
    if tape.curve.model_sha256!=m.fingerprint or not np.allclose(dates,surface.times):raise ValueError('case, tape and continuation dates/model differ')
    n,T,_=X.shape;T-=1;K=len(case.instruments);dt=c.year_fraction
    conditional=innovations if innovations is not None else ConditionalInnovations.generate(pm,T,scenarios,dt,seed)
    if (conditional.model_sha256!=pm.fingerprint or conditional.step_years!=dt or conditional.seed!=seed
            or conditional.values.shape!=(T,scenarios,m.dimension) or not np.isfinite(conditional.values).all()):
        raise ValueError('conditional innovations must match the model, grid, seed and scenario count')
    A=np.exp(-pm.kappa*dt);gh=np.asarray(c.gpus_per_block)*c.hours_per_step
    h=np.broadcast_to(c.blocks,(n,4)).copy();b=np.full(n,c.initial_cash);d=np.full(n,c.initial_revolver);term=np.full(n,c.initial_term_debt)
    q=np.zeros((n,K));posted=np.zeros(n);interest_due=np.zeros(n);root=np.zeros((n,K));alive=np.ones(n,bool)
    failed=np.zeros(n,bool);failure=np.full(n,-1,int);short=np.zeros(n);maxim=np.zeros(n);mincash=np.full(n,np.inf);maxdebt=d.copy();sales=np.zeros(n);tc=np.zeros(n)
    pos=np.zeros((n,T+1,K));margin_path=np.zeros((n,T+1));cash=np.zeros((n,T+1));debts=np.zeros((n,T+1));components=np.zeros((n,7));wealth=np.zeros(n)
    # components: operating, interest, borrowing principal, asset sales, financial gains, execution, resolution.
    prevmarks=np.zeros((n,K));casherr=0.;budgeterr=0.;lp_calls=0;root_delta=[]
    for u,t in enumerate(dates):
        for i in range(n):
            if not alive[i]:
                cash[i,u]=b[i];debts[i,u]=d[i];continue
            before=b[i];flows=0.;x=X[i,u];marks=_observed_marks(case,t,x,tape.benchmark_basis[i,u]);old=q[i].copy()
            if u:
                cp_alive = tape.counterparty_alive is None or bool(tape.counterparty_alive[i,u])
                event=settle(case.instruments,reference_model(m,tape.benchmark_basis[i,u]),dates[u-1],t,x,old,prevmarks[i],counterparty_alive=cp_alive)
                settled=event['cash'];old=event['positions']
                b[i]+=settled+posted[i];flows+=settled+posted[i];components[i,4]+=settled;posted[i]=0.
            # Interest accrues on the actual post-trade balance and is paid next date.
            # This also charges the final interval before terminal debt repayment.
            due=interest_due[i];b[i]-=due;flows-=due;components[i,1]-=due;interest_due[i]=0.
            if u==T:
                realized=sale_proceeds(float(h[i]@tape.hardware[i,u]),1.,c.liquidation_fraction,c.liquidation_impact)
                closing_claims=float(old@funded_prices(case.instruments,marks))
                b[i]+=closing_claims;flows+=closing_claims;components[i,4]+=closing_claims
                final=realized-d[i]-term[i];b[i]+=final;flows+=final;components[i,3]+=realized;components[i,6]-=d[i]+term[i]
                d[i]=0;term[i]=0;wealth[i]=b[i];alive[i]=False;q[i]=0
            else:
                local,_,unitcost=spot_state(m,t,x,c);local*=tape.local_multiplier[i,u]
                opening=float(gh@(np.asarray(c.firm_blocks)*np.asarray(c.firm_rates)))*c.prepayment_fraction*tape.customer_alive[i,u]
                closing=float(gh@(np.asarray(c.firm_blocks)*np.asarray(c.firm_rates)))*(1-c.prepayment_fraction)*tape.customer_alive[i,u]
                # Coupon terms, not asset valuation yields, determine contractual interest.
                spread=float(m.value('funding',t,x,t));reference=float(m.short_rate(t,x))
                repay_term=min(term[i],term[i] if u+1>=c.term_maturity_step else c.term_amortization)
                b[i]+=opening-repay_term;flows+=opening-repay_term;components[i,0]+=opening;components[i,2]-=repay_term;term[i]-=repay_term
                im=np.array([float(inst.margin(pm,t,x)) for inst in case.instruments])*tape.margin_multiplier[i,u]
                incoming=float(np.abs(old)@im);maxim[i]=max(maxim[i],incoming)
                # Maturing principal is due from current liquidity.
                commitment=0. if u>=c.revolver_maturity_step else c.revolver_commitment
                repair=_repair_one(c,h[i],b[i],d[i],tape.hardware[i,u],tape.eligible[i,u],tape.availability[i,u],tape.replacement[i,u],local,unitcost,commitment,bool(tape.freeze[i,u]),0.,incoming)
                reason=''
                if repair is None:reason='incoming payment or collateral call'
                else:
                    oldh=h[i].copy();h[i]=repair['holdings'];sales[i]+=float((oldh-h[i])@tape.hardware[i,u])
                    net=repair['draw']-repair['repay'];fin=net-c.debt_fee*(repair['draw']+repair['repay'])
                    b[i]+=repair['proceeds']+fin;flows+=repair['proceeds']+fin;d[i]+=net;components[i,2]+=fin;components[i,3]+=repair['proceeds']
                    total_budget=b[i]-repair['cost']-c.reserve
                    limit=np.minimum(np.maximum(h[i]*tape.availability[i,u]-repair['owned'],0),tape.demand[i,u]);limit=np.where(local>unitcost,limit,0.)
                    future=x*A+conditional.values[u]
                    # Revalue at fixed contractual delivery dates.
                    pricing_model=reference_model(pm,tape.benchmark_basis[i,u])
                    gains=conditional_gains(case.instruments,pricing_model,t,x,dates[u+1],future,marks)
                    bounds=[]
                    for inst in case.instruments:
                        lo,hi=inst.bounds(t)
                        if tape.counterparty_alive is not None and not tape.counterparty_alive[i,u] and getattr(inst,'counterparty','cleared')!='cleared':lo=hi=0.
                        bounds.append((lo,hi))
                    loss=-surface.value(u+1,future,h[i],d[i],term[i],c,bool(tape.customer_alive[i,u]))
                    # Borrowing principal offsets its terminal liability. Only interest and fees alter loss.
                    remaining=T-u
                    rate_scen=pm.short_rate(dates[u+1],future)+pm.value('funding',dates[u+1],future,dates[u+1])
                    interest_scen=dt*((reference+spread)+max(remaining-1,0)*rate_scen)
                    load=np.column_stack([-np.broadcast_to(gh*(local-unitcost),(scenarios,4)),interest_scen+c.debt_fee,-interest_scen+c.debt_fee])
                    ceiling=min(commitment,c.advance_rate*float(h[i]@tape.eligible[i,u]))
                    draw_limit=0. if tape.freeze[i,u] else max(0,min(c.max_draw_per_step-repair['draw'],ceiling-d[i]))
                    ctrl=LinearControls([f'spot:{j}' for j in range(4)]+['draw','repay'],np.zeros(6),np.r_[limit,draw_limit,d[i]],
                         np.r_[gh*np.maximum(unitcost,0),-(1-c.debt_fee),1+c.debt_fee],load)
                    lo=np.array([v[0] for v in bounds]);hi=np.array([v[1] for v in bounds])
                    if policy=='unhedged':lo[:]=0.;hi[:]=0.
                    elif policy=='static' and u:
                        lock=np.where(np.array([inst.maturity>t+1e-9 for inst in case.instruments]),root[i],0.)
                        lo=lock.copy();hi=lock.copy()
                    if tape.counterparty_alive is not None and not tape.counterparty_alive[i,u]:
                        for kk,inst in enumerate(case.instruments):
                            if getattr(inst,'counterparty','cleared')!='cleared':lo[kk]=hi[kk]=0.
                    depth=np.array([z.trade_limit for z in case.instruments])*tape.depth_multiplier[i,u]
                    impactdepth=np.array([z.impact_depth for z in case.instruments])*tape.depth_multiplier[i,u]
                    spreads=np.array([z.spread for z in case.instruments])*tape.spread_multiplier[i,u]
                    impacts=np.array([z.impact_slope for z in case.instruments])*tape.spread_multiplier[i,u]
                    result=solve_conditional(loss,gains,old,marks,im,spreads,impacts,impactdepth,depth,lo,hi,total_budget,
                        alpha,risk_weight,controls=ctrl,carry_convention=carry,
                        funded_prices=funded_prices(case.instruments,marks),
                        premium_liquidity_rate=np.array([getattr(z,'premium_charge_annual',0.) for z in case.instruments])*dt,
                        funding_rate=np.array([z.liquidity_charge_annual for z in case.instruments])*dt)
                    lp_calls+=1
                    if result['status']!='optimal':reason='trade or operating cash constraints'
                    else:
                        target=result['position'];act=result['controls'];draw=act['draw'];repay=act['repay'];fin=draw-repay-c.debt_fee*(draw+repay)
                        d[i]+=draw-repay;b[i]+=fin;flows+=fin;components[i,2]+=fin
                        cost=result['transaction_cost'];newim=float(abs(target)@im);spot=np.array([act[f'spot:{j}'] for j in range(4)])
                        premium=float((target-old)@funded_prices(case.instruments,marks))
                        b[i]-=cost+newim+premium;flows-=cost+newim+premium;components[i,4]-=premium;components[i,5]-=cost;posted[i]=newim;tc[i]+=cost
                        required=repair['cost']+float(gh@(spot*np.maximum(unitcost,0)))
                        budgeterr=max(budgeterr,max(0.,c.reserve+required-b[i]));mincash[i]=min(mincash[i],b[i]-required)
                        op=closing+float(gh@(spot*(local-unitcost)))-repair['cost'];b[i]+=op;flows+=op;components[i,0]+=op
                        short[i]+=float(gh@repair['short']);q[i]=target
                        if u==0:root[i]=target
                        maxim[i]=max(maxim[i],newim);maxdebt[i]=max(maxdebt[i],d[i])
                        interest_due[i]=dt*(d[i]*(reference+spread)+term[i]*c.fixed_term_coupon)
                if reason:
                    failed[i]=True;failure[i]=u;alive[i]=False;q[i]=0.
                    financial_close=float(old@funded_prices(case.instruments,marks))
                    b[i]+=financial_close;flows+=financial_close;components[i,4]+=financial_close
                    # Retain lender and delivery losses on failed paths.
                    realized=sale_proceeds(float(h[i]@tape.hardware[i,u]),1.,c.liquidation_fraction,c.liquidation_impact)
                    obligation=(T-u)*float(gh@(np.asarray(c.firm_blocks)*np.maximum(c.replacement_multiplier*local-np.asarray(c.firm_rates)*tape.customer_alive[i,u],0)))
                    fin=realized-d[i]-term[i]-obligation-opening;b[i]+=fin;flows+=fin;components[i,3]+=realized;components[i,6]-=d[i]+term[i]+obligation;components[i,0]-=opening
                    wealth[i]=b[i];d[i]=0;term[i]=0;short[i]+=(T-u)*float(gh@np.asarray(c.firm_blocks))
                prevmarks[i]=marks
            casherr=max(casherr,abs(b[i]-before-flows));pos[i,u]=q[i];margin_path[i,u]=posted[i];cash[i,u]=b[i];debts[i,u]=d[i]
    reconstruct=c.initial_cash+components.sum(axis=1);err=float(np.max(abs(reconstruct-wealth)))
    if max(casherr,err,budgeterr)>1e-4:raise RuntimeError(f'curve cash audit failed: {casherr}, {err}, {budgeterr}')
    audit={'maximum_cash_equation_error':casherr,'maximum_component_error':err,'maximum_prefunding_violation':budgeterr,'lp_calls':lp_calls,
      'conditional_atoms':scenarios,'conditional_tail_mass':(1-alpha)*scenarios,'conditional_confidence':alpha,
      'risk_weight':risk_weight,'conditional_seed':seed,
      'conditional_innovations_sha256':conditional.fingerprint,
      'claim_types':sorted(set(getattr(z,'kind','futures') for z in case.instruments)),
      'case_sha256':case.fingerprint,'market_model_sha256':m.fingerprint,'policy_model_sha256':pm.fingerprint,
      'continuation_training_paths':surface.training_paths,'conditioning':'current factors, physical inventory, debt and cash only','carry':carry,
      'interval_years':dt,'required_delivery_gpu_hours':float(T*(gh@np.asarray(c.firm_blocks))),
      'component_names':['operating','interest','debt principal and fees','asset sales','financial settlements','transaction costs','failure/terminal liabilities']}
    return ControlResult(policy,wealth,failed,failure,short,maxim,np.where(np.isfinite(mincash),mincash,wealth),maxdebt,sales,tc,pos,margin_path,cash,debts,components,audit)
