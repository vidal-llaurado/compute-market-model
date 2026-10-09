"""Financial execution against a financed four-pool fleet.

Policies share an immutable market tape. A trained conditional kernel supplies
prospective shocks; only current observations reach the execution optimizer."""
from __future__ import annotations
from dataclasses import dataclass,replace
import numpy as np
from .simulation import SimulationConfig,MarketTape,generate_tape,_repair_one,sale_proceeds
from .execution import ExecutionMandate,dispatch_controls
from .risk_execution import (MULTIPLIER,INSTRUMENTS,ConditionalKernel,quoted_marks,planned_value,solve_conditional,put_quote)

@dataclass
class FinancialTape:
    physical: MarketTape
    state: np.ndarray
    marks: np.ndarray
    im: np.ndarray
    spread: np.ndarray
    trade_limit: np.ndarray
    position_limit: np.ndarray
    clearing_alive: np.ndarray
    dealer_alive: np.ndarray
    im_recovery: np.ndarray
    name: str='nominal'
    impact_depth: np.ndarray | None = None
    impact_slope: np.ndarray | None = None
    def copy(self):
        return FinancialTape(**{k:v.copy() if isinstance(v,np.ndarray) else v.copy() if k=='physical' else v for k,v in vars(self).items()})
    def validate(self):
        self.physical.validate();c=self.physical.config;p,t=c.paths,c.steps+1
        if self.state.shape!=(p,t,11) or not np.isfinite(self.state).all():raise ValueError('invalid financial state tape')
        if self.impact_depth is None:
            self.impact_depth=np.broadcast_to([120.,96.,28.,16.,24.,100.],(p,t,6)).copy()
        if self.impact_slope is None:
            self.impact_slope=np.broadcast_to([.7,.875,.7,.525,1.05,.7],(p,t,6)).copy()
        for name in ('marks','im','spread','trade_limit','position_limit','impact_depth','impact_slope'):
            a=getattr(self,name)
            if a.shape!=(p,t,6) or not np.isfinite(a).all():raise ValueError(f'invalid {name}')
            if name!='marks' and np.any(a<0):raise ValueError(f'negative {name}')
        for name in ('clearing_alive','dealer_alive','im_recovery'):
            a=getattr(self,name)
            if a.shape!=(p,t) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError(f'invalid {name}')
        if np.any(np.diff(self.clearing_alive.astype(int),axis=1)>0) or np.any(np.diff(self.dealer_alive.astype(int),axis=1)>0):raise ValueError('absorbing default states required')
        return self
    def save(self,path):
        import json
        raw={k:v for k,v in vars(self.physical).items() if isinstance(v,np.ndarray)}
        raw={f'physical_{k}':v for k,v in raw.items()}
        raw.update({k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)})
        raw['configuration']=np.array(json.dumps(self.physical.config.to_dict()))
        raw['name']=np.array(self.name)
        np.savez_compressed(path,**raw)
    @classmethod
    def load(cls,path):
        import json
        with np.load(path,allow_pickle=False) as d:
            c=SimulationConfig(**json.loads(str(d['configuration'])))
            raw={k[len('physical_'):]:d[k] for k in d.files if k.startswith('physical_')}
            physical=MarketTape(**raw,config=c,metadata={'source':'saved financial tape'})
            return cls(physical,**{k:d[k] for k in ('state','marks','im','spread','trade_limit','position_limit','clearing_alive','dealer_alive','im_recovery','impact_depth','impact_slope')},name=str(d['name'])).validate()


def financial_tape(config):
    raw=generate_tape(config);p,t=config.paths,config.steps+1;dt=config.year_fraction
    rng=np.random.default_rng(np.random.SeedSequence([config.seed,9123]))
    generation=np.zeros((p,t));fx=np.zeros((p,t))
    for u in range(1,t):
        generation[:,u]=np.exp(-dt)*generation[:,u-1]+.11*np.sqrt(dt)*rng.normal(size=p)
        common=np.log(raw.index[:,u]/raw.index[:,u-1])
        fx[:,u]=np.exp(-.6*dt)*fx[:,u-1]-.15*common+.12*np.sqrt(dt)*rng.normal(size=p)
    g2=1.5*raw.index*np.exp(generation)
    state=np.zeros((p,t,11));state[:,:,0]=np.log(raw.index/2);state[:,:,1]=np.log(g2/3)
    state[:,:,2]=np.log(raw.power/70);state[:,:,3]=raw.reference_rate;state[:,:,4]=fx
    initial=float(np.asarray(config.blocks)@np.asarray(config.hardware_values))
    state[:,:,5]=np.log((raw.hardware@np.asarray(config.blocks))/initial)
    reference=np.stack([raw.index,raw.index,g2,g2],axis=2)
    state[:,:,6:10]=np.log(raw.local/reference)
    state[:,:,10]=np.log1p(raw.demand_blocks.sum(axis=2)/sum(config.blocks))
    marks=np.stack([quoted_marks(state[:,u],config.steps-u,config.steps,dt) for u in range(t)],axis=1)
    im=np.abs(marks)*MULTIPLIER*np.array([.12,.12,.16,0.,.05,0.]);im[:,:,3]=120.
    im*=raw.margin_multiplier[:,:,None]
    spread=np.broadcast_to([2.,2.5,2.,1.5,3.,2.],(p,t,6)).copy()
    depth=np.broadcast_to([300.,240.,70.,40.,60.,250.],(p,t,6)).copy()
    limit=np.broadcast_to([800.,600.,200.,150.,150.,600.],(p,t,6)).copy()
    return FinancialTape(raw,state,marks,im,spread,depth,limit,np.ones((p,t),bool),
                         raw.counterparty_alive.copy(),np.ones((p,t))).validate()


def stress_tape(tape,name):
    out=tape.copy();c=out.physical.config;start=max(1,c.steps//2);sl=np.s_[:,start:]
    raw=out.physical
    if name in ('basis_widening','compound'):
        raw.local[sl]*=.68;out.state[:,start:,6:10]+=np.log(.68)
    if name=='benchmark_gap':
        out.marks[:,start:,:2]*=1.45
    if name in ('margin_jump','compound'):out.im[sl]*=4
    if name in ('depth_freeze','compound'):out.trade_limit[sl]=0
    if name=='position_limit_cut':out.position_limit[sl]*=.2
    if name=='spread_widening':out.spread[sl]*=15
    if name=='option_dealer_default':out.dealer_alive[sl]=False
    if name=='clearing_failure':out.clearing_alive[sl]=False;out.im_recovery[:,start]=.6
    if name in ('lender_haircut','compound'):
        raw.eligible[sl]*=.55;raw.freeze[sl]=True
    if name=='power_spike':
        raw.power[sl]*=2.2;out.state[:,start:,2]+=np.log(2.2);out.marks[:,start:,2]*=2.2
    if name=='rate_jump':
        raw.reference_rate[sl]+=.055;out.state[:,start:,3]+=.055;out.marks[:,start:,3]+=.055
    if name=='FX_jump':out.state[:,start:,4]+=np.log(1.25);out.marks[:,start:,4]*=1.25
    if name=='volume_outage':
        raw.availability[:,start:,[0,1]]*=.35;raw.demand_blocks[sl]*=.5
        out.state[:,:,10]=np.log1p(raw.demand_blocks.sum(axis=2)/sum(c.blocks))
    if name=='correlation_break':
        anchor=out.marks[:,start-1,:2].copy()
        out.marks[:,start:,:2]=anchor[:,None,:]**2/np.maximum(out.marks[:,start:,:2],1e-8)
    if name in ('benchmark_gap','correlation_break'):
        # The option and future share the same dated settlement benchmark.
        # Revalue the option at every affected date, including exact expiry intrinsic.
        strike=2*np.exp(-.08*c.steps*c.year_fraction)
        for u in range(start,c.steps+1):
            out.marks[:,u,5]=put_quote(out.marks[:,u,0],strike,(c.steps-u)*c.year_fraction)
    supported=['basis_widening','benchmark_gap','margin_jump','depth_freeze','position_limit_cut','spread_widening',
               'option_dealer_default','clearing_failure','lender_haircut','power_spike','rate_jump','FX_jump','volume_outage','correlation_break','compound']
    if name not in supported:raise ValueError('unknown named stress')
    out.name=name;return out.validate()


@dataclass
class HedgePathResult:
    policy: str
    terminal_wealth: np.ndarray
    financing_failure: np.ndarray
    delivery_breach: np.ndarray
    failure_step: np.ndarray
    failure_reason: list[str]
    minimum_cash_headroom: np.ndarray
    maximum_margin: np.ndarray
    maximum_funding_need: np.ndarray
    total_turnover: np.ndarray
    transaction_cost: np.ndarray
    hedge_cash_pnl: np.ndarray
    credit_loss: np.ndarray
    total_asset_sales: np.ndarray
    positions: np.ndarray
    cash_trace: np.ndarray
    pnl_components: np.ndarray
    audit: dict
    def save(self,path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        arrays={k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        arrays['failure_reason']=np.asarray(self.failure_reason);arrays['policy']=np.array(self.policy)
        np.savez_compressed(path,**arrays)


def simulate_hedges(tape,kernel:ConditionalKernel,policy='joint',alpha=.9,risk_weight=.65,margin_funding=None,mandate=None):
    if policy not in ('unhedged','static','dynamic','adaptive','joint','joint_unhedged','joint_static'):
        raise ValueError('policy must be unhedged, static, adaptive or joint')
    mandate=mandate or ExecutionMandate()
    # 'dynamic' is a low-level alias for the conditional financial-only comparator.
    joint=policy in ('joint','joint_unhedged','joint_static')
    tape.validate();raw=tape.physical;c=raw.config;n,T=c.paths,c.steps;dt=c.year_fraction
    holdings=np.broadcast_to(c.blocks,(n,4)).copy();cash=np.full(n,c.initial_cash,dtype=float);debt=np.full(n,c.initial_revolver,dtype=float);term=np.full(n,c.initial_term_debt,dtype=float)
    active=np.ones(n,bool);failed=np.zeros(n,bool);breach=np.zeros(n,bool);failure=np.full(n,-1,int);reasons=['']*n
    wealth=np.zeros(n);mincash=np.full(n,np.inf);maxmargin=np.zeros(n);fundingneed=np.zeros(n);turnsum=np.zeros(n);tc=np.zeros(n);hpnl=np.zeros(n);credit=np.zeros(n);sales=np.zeros(n)
    q=np.zeros((n,6));rootq=np.zeros((n,6));posted=np.zeros(n);commitment=np.full(n,c.revolver_commitment,dtype=float)
    positions=np.zeros((n,T+1,6));cashtrace=np.zeros((n,T+1));components=np.zeros((n,7))
    joint_draws=0.;joint_repays=0.;joint_spot_blocks=0.;joint_optimizations=0
    # Operating; debt interest; terminal asset proceeds; hedge gain incl premiums; execution; credit; other principal/closeout.
    casherr=0.;debt_error=0.;inventory_error=0.;funding_error=0.;calls=0
    gh=np.array(c.gpus_per_block)*c.hours_per_step;firm=np.asarray(c.firm_blocks)
    for t in range(T+1):
        for i in np.flatnonzero(active):
            before=cash[i];cashflows=0.;old=q[i].copy();marks=tape.marks[i,t]
            previous_marks=tape.marks[i,t-1] if t else marks
            received=0.;credit_now=0.
            if t:
                vm=old[:5]*(marks[:5]-previous_marks[:5])*MULTIPLIER[:5]
                clearing_event=tape.clearing_alive[i,t-1] and not tape.clearing_alive[i,t]
                if clearing_event:
                    credit_now+=(1-c.counterparty_recovery)*float(np.maximum(vm,0).sum())
                received+=float(vm.sum())-credit_now
                dealer_event=tape.dealer_alive[i,t-1] and not tape.dealer_alive[i,t]
                if dealer_event:
                    value=old[5]*marks[5]*MULTIPLIER[5]
                    received+=c.counterparty_recovery*value;credit_now+=(1-c.counterparty_recovery)*value;old[5]=0.
                if t==T and tape.dealer_alive[i,t]:
                    received+=old[5]*marks[5]*MULTIPLIER[5];old[5]=0.
                interest=debt[i]*(c.revolver_spread+raw.reference_rate[i,t-1])*dt+term[i]*c.fixed_term_coupon*dt
                amort=term[i] if t==c.term_maturity_step else min(term[i],c.term_amortization)
                term[i]-=amort;cash[i]-=interest+amort;cashflows-=interest+amort
                components[i,1]-=interest;components[i,6]-=amort
                if t==c.revolver_maturity_step:
                    probability=c.refinance_probability/(1+.25*(raw.margin_multiplier[i,t]-1))
                    if raw.refinance_u[i,t]>probability:commitment[i]=0.
            collateral_return=posted[i]*tape.im_recovery[i,t]
            collateral_loss=posted[i]-collateral_return;credit_now+=collateral_loss
            cash[i]+=received+collateral_return;cashflows+=received+collateral_return
            hpnl[i]+=received;credit[i]+=credit_now
            components[i,3]+=received+credit_now-collateral_loss;components[i,5]-=credit_now
            posted[i]=0.
            if not tape.clearing_alive[i,t]:old[:5]=0.
            if t==T:
                proceeds=sale_proceeds(float(holdings[i]@raw.hardware[i,t]),1.,c.liquidation_fraction,c.liquidation_impact)
                cash[i]+=proceeds-debt[i]-term[i];cashflows+=proceeds-debt[i]-term[i]
                components[i,2]+=proceeds;components[i,6]-=debt[i]+term[i]
                wealth[i]=cash[i];failed[i]|=cash[i]<-1e-7
                if failed[i] and failure[i]<0:failure[i]=t;reasons[i]='terminal insolvency'
                active[i]=False;q[i]*=0
            else:
                fixed=float(gh@(firm*np.asarray(c.firm_rates)))*float(raw.customer_alive[i,t])
                opening=fixed*c.prepayment_fraction;closing=fixed-opening
                cash[i]+=opening;cashflows+=opening;components[i,0]+=opening
                fx=np.exp(tape.state[i,t,4]);unitcost=np.asarray(c.nonpower_unit_cost)*fx+np.asarray(c.mwh_per_gpu_hour)*raw.power[i,t]
                preim=float(np.abs(old)@tape.im[i,t]);maxmargin[i]=max(maxmargin[i],preim)
                support=c.receivable_advance*closing
                beforeowned=np.minimum(firm,holdings[i]*raw.availability[i,t])
                beforeext=np.minimum(firm-beforeowned,raw.replacement[i,t])
                beforeshort=np.maximum(0,firm-beforeowned-beforeext)
                beforecost=float(gh@(beforeowned*unitcost+beforeext*c.replacement_multiplier*raw.local[i,t]+beforeshort*c.shortfall_penalty_per_gpu_hour))+c.fixed_overhead_per_step
                bb=min(commitment[i],c.advance_rate*float(holdings[i]@raw.eligible[i,t])+support)
                fundingneed[i]=max(fundingneed[i],preim+beforecost+c.reserve+max(0,debt[i]-bb)*(1+c.debt_fee)-cash[i])
                repaired=_repair_one(c,holdings[i],cash[i],debt[i],raw.hardware[i,t],raw.eligible[i,t],raw.availability[i,t],raw.replacement[i,t],raw.local[i,t],unitcost,commitment[i],bool(raw.freeze[i,t]),support,preim)
                failreason=''
                if repaired is None:failreason='incoming margin / delivery prefunding / borrowing base'
                else:
                    r=repaired;oldassets=holdings[i].copy();olddebt=debt[i]
                    holdings[i]=r['holdings'];sold=oldassets-holdings[i];sales[i]+=float(sold.sum())
                    debt[i]+=r['draw']-r['repay'];fee=c.debt_fee*(r['draw']+r['repay'])
                    financing=r['proceeds']+r['draw']-r['repay']-fee
                    cash[i]+=financing;cashflows+=financing
                    components[i,2]+=r['proceeds'];components[i,6]+=r['draw']-r['repay']-fee
                    debt_error=max(debt_error,abs(debt[i]-olddebt-r['draw']+r['repay']))
                    inventory_error=max(inventory_error,float(np.max(np.abs(holdings[i]+sold-oldassets))))
                    mincash[i]=min(mincash[i],cash[i]-preim-r['cost']-c.reserve)
                    total_budget=max(0.,cash[i]-r['cost']-c.reserve)
                    planned_spot=np.minimum(np.maximum(0,holdings[i]*raw.availability[i,t]-r['owned']),raw.demand_blocks[i,t])
                    planned_spot=np.where(raw.local[i,t]>unitcost,planned_spot,0.)
                    spot_prefund=min(total_budget,float(gh@(planned_spot*unitcost)))
                    budget=total_budget-spot_prefund
                    limit=tape.position_limit[i,t].copy();lower,upper=mandate.bounds(limit)
                    if not tape.clearing_alive[i,t]:lower[:5]=upper[:5]=0.
                    if not tape.dealer_alive[i,t]:lower[5]=upper[5]=0.
                    if policy=='joint_unhedged':lower[:]=upper[:]=0.
                    if policy=='static' and t:budget=total_budget
                    if policy=='unhedged':target=np.zeros(6);result={'status':'optimal','position':target}
                    elif policy=='static' and t:
                        target=rootq[i].copy();target[:5]*=tape.clearing_alive[i,t];target[5]*=tape.dealer_alive[i,t]
                        result={'status':'optimal','position':target}
                    else:
                        predicted=kernel.forecast(tape.state[i,t])
                        futuremarks=quoted_marks(predicted,T-t-1,T,dt)
                        gains=(futuremarks-marks)*MULTIPLIER
                        due=term[i] if t+1==c.term_maturity_step else min(term[i],c.term_amortization)
                        futurevalue=planned_value(predicted,holdings[i],debt[i],term[i]-due,T-t-1,c,raw.availability[i,t],raw.customer_alive[i,t])
                        futurevalue-=due+(debt[i]*(raw.reference_rate[i,t]+c.revolver_spread)+term[i]*c.fixed_term_coupon)*dt
                        losses=-futurevalue
                        local_args=mandate.local_arguments(dt)
                        if margin_funding is not None:local_args['funding_rate']=margin_funding
                        controls=None
                        if joint:
                            budget=total_budget
                            available_base=min(commitment[i],c.advance_rate*float(holdings[i]@raw.eligible[i,t])+support)
                            additional_draw=0. if raw.freeze[i,t] else max(0.,min(c.max_draw_per_step-r['draw'],available_base-debt[i]))
                            controls=dispatch_controls(c,planned_spot,unitcost,raw.local[i,t],predicted[:,3],
                                T-t,debt[i],additional_draw,debt[i],raw.reference_rate[i,t])
                            joint_optimizations+=1
                        def local_solve(available):
                            lo,hi=lower.copy(),upper.copy()
                            if policy=='joint_static' and t:
                                locked=rootq[i].copy();locked[:5]*=tape.clearing_alive[i,t];locked[5]*=tape.dealer_alive[i,t]
                                if np.any(locked<lo-1e-8) or np.any(locked>hi+1e-8):return {'status':'infeasible','position':None}
                                lo=hi=locked
                            return solve_conditional(losses,gains,old,marks,tape.im[i,t],tape.spread[i,t],
                                tape.impact_slope[i,t],tape.impact_depth[i,t],tape.trade_limit[i,t],lo,hi,available,
                                alpha,risk_weight,funded_prices=np.r_[np.zeros(5),marks[5]*MULTIPLIER[5]],
                                controls=controls,**local_args)
                        result=local_solve(budget);calls+=1
                        if result['status']=='infeasible' and total_budget>budget+1e-7:
                            budget=total_budget;result=local_solve(budget);calls+=1
                    if result['status']!='optimal':failreason='hedge execution or position mandate'
                    else:
                        target=np.asarray(result['position']);turn=np.abs(target-old)
                        cost=float(turn@tape.spread[i,t]+np.maximum(turn-tape.impact_depth[i,t],0)@tape.impact_slope[i,t])
                        premium=(target[5]-old[5])*marks[5]*MULTIPLIER[5]
                        newim=float(np.abs(target)@tape.im[i,t])
                        selected_spot=None
                        if joint:
                            decisions=result['controls'];dplus=decisions['draw'];dminus=decisions['repay']
                            financing=dplus-dminus-c.debt_fee*(dplus+dminus)
                            previous_debt=debt[i];debt[i]+=dplus-dminus
                            cash[i]+=financing;cashflows+=financing;components[i,6]+=financing
                            budget+=financing
                            debt_error=max(debt_error,abs(debt[i]-previous_debt-dplus+dminus))
                            joint_draws+=dplus;joint_repays+=dminus
                            selected_spot=np.array([decisions[f'spot:{j}'] for j in range(4)])
                            joint_spot_blocks+=float(selected_spot.sum())
                        if np.any(target<lower-1e-5) or np.any(target>upper+1e-5) or np.any(turn>tape.trade_limit[i,t]+1e-5) or newim+cost+premium>budget+1e-4:
                            failreason='static hedge violates current execution or cash limits'
                        else:
                            if t==0:rootq[i]=target
                            q[i]=target;posted[i]=newim;maxmargin[i]=max(maxmargin[i],newim)
                            cash[i]-=newim+cost+premium;cashflows-=newim+cost+premium
                            tc[i]+=cost;turnsum[i]+=float(turn.sum());hpnl[i]-=premium+cost
                            components[i,3]-=premium;components[i,4]-=cost
                            capacity=np.maximum(0,holdings[i]*raw.availability[i,t]-r['owned'])
                            spot=np.minimum(capacity,raw.demand_blocks[i,t]);spot=np.where(raw.local[i,t]>unitcost,spot,0)
                            if selected_spot is not None:spot=selected_spot
                            spotcost=float(gh@(spot*unitcost));available=max(0.,cash[i]-r['cost']-c.reserve)
                            if spotcost>available and spotcost>0:spot*=available/spotcost
                            spotcost=float(gh@(spot*unitcost));revenue=float(gh@(spot*raw.local[i,t]))
                            mincash[i]=min(mincash[i],cash[i]-r['cost']-spotcost-c.reserve)
                            operating=-r['cost']-spotcost+revenue+closing
                            cash[i]+=operating;cashflows+=operating;components[i,0]+=operating
                            breach[i]|=float(gh@r['short'])>1e-7
                if failreason:
                    failed[i]=True;failure[i]=t;reasons[i]=failreason;active[i]=False;breach[i]|=bool(np.any(firm>0))
                    # Close out the marked surviving option; futures were cash-settled above.
                    option=old[5]*marks[5]*MULTIPLIER[5]
                    proceeds=sale_proceeds(float(holdings[i]@raw.hardware[i,t]),1.,c.liquidation_fraction,c.liquidation_impact)
                    closeout=(T-t)*float(gh@(firm*np.maximum(0,c.replacement_multiplier*raw.local[i,t]-np.asarray(c.firm_rates)*float(raw.customer_alive[i,t]))))
                    resolution=option+proceeds-debt[i]-term[i]-closeout-opening
                    cash[i]+=resolution;cashflows+=resolution;wealth[i]=cash[i];hpnl[i]+=option
                    components[i,3]+=option;components[i,2]+=proceeds;components[i,6]-=debt[i]+term[i]+closeout;components[i,0]-=opening
                    mincash[i]=min(mincash[i],cash[i]-preim-c.reserve);q[i]*=0
            casherr=max(casherr,abs(cash[i]-before-cashflows));positions[i,t]=q[i];cashtrace[i,t]=cash[i]
    reconstruction=c.initial_cash+components.sum(axis=1)
    component_error=float(np.max(np.abs(reconstruction-wealth)))
    if max(casherr,debt_error,inventory_error,component_error)>1e-5:raise RuntimeError(f'path ledger error {casherr,debt_error,inventory_error,component_error}')
    mincash=np.where(np.isfinite(mincash),mincash,wealth-c.reserve)
    audit={'max_cash_conservation_error':casherr,'max_debt_balance_error':debt_error,'max_inventory_error':inventory_error,
      'max_pnl_component_error':component_error,'conditional_optimizations':calls,'all_paths_terminated':bool(not active.any()),
      'training_observations':kernel.observations,'conditional_scenarios':kernel.scenarios,
      'execution_mandate':mandate.to_dict(),'joint_optimizations':joint_optimizations,
      'joint_incremental_draws':joint_draws,'joint_incremental_repays':joint_repays,
      'joint_spot_block_intervals':joint_spot_blocks,
      'policy_class':'joint conditional dispatch-financing-hedging' if joint else 'conditional hedge' if policy in ('dynamic','adaptive') else policy,
      'cash_opportunity_rate_override':margin_funding,
      'component_names':['operating_cash','debt_interest','asset_sales','gross_financial_cash','execution_cost','credit_losses','principal_and_resolution']}
    return HedgePathResult(policy,wealth,failed,breach,failure,reasons,mincash,maxmargin,fundingneed,turnsum,tc,hpnl,credit,sales,positions,cashtrace,components,audit)
