"""Joint reservation pricing and incremental financial-policy attribution."""
import numpy as np
from .dynamic_hedging import solve_dynamic, compile_dynamic, _run, solver_certificate, compiled_fingerprint
from .schema import Contract
from .contracts import combine,node_gpu_hours
from .pricing import _bracket_price, PriceResult
from .optimizer import OptimizationError
from .risk import expected_shortfall


def price_and_hedge(p,m,offered,book=None,payment='upfront',price_tolerance=.05,mode='dynamic',tie_break=True):
    book=book or Contract.empty(p);base=solve_dynamic(p,m,book,mode=mode,tie_break=tie_break)
    if base.status!='optimal':raise ValueError('existing book requires a financing or capacity cure')
    merged=combine(book,offered,p);volumes=node_gpu_hours(p,offered)
    def valued(amount, tie_break=False):
        if payment=='upfront':return solve_dynamic(p,m,merged,premium=amount,mode=mode,tie_break=tie_break)
        if payment not in ('opening_rate','closing_rate'):raise ValueError('invalid payment convention')
        paid=Contract(**merged.to_dict()).validate(p)
        if payment=='opening_rate':paid.opening_receipts+=volumes*amount
        else:paid.closing_receipts+=volumes*amount
        return solve_dynamic(p,m,paid,mode=mode,tie_break=tie_break)
    if payment!='upfront' and not np.any(volumes):raise ValueError('rate requires nonzero service volume')
    if price_tolerance<=0:raise ValueError('positive price tolerance required')
    # In a continuous case the payment is an affine cash-flow control. Minimize
    # it directly under the reference risk-value constraint; no scalar search.
    lp,v,obj,_,_=compile_dynamic(p,m,merged,mode=mode)
    pricing_certificate=None
    if not any(lp.integral):
        if payment not in ('upfront','opening_rate','closing_rate'):raise ValueError('invalid payment convention')
        pay=lp.var('contract_payment',-1e9,1e9)
        if payment=='upfront':lp.eq[lp.eq_names.index('opening_cash[0]')][pay]=-1.
        else:
            tag='opening_cash' if payment=='opening_rate' else 'closing_cash'
            for n,volume in enumerate(volumes):
                if volume:lp.eq[lp.eq_names.index(f'{tag}[{n}]')][pay]=-float(volume)
        lp.inequality({int(v['risk_loss'][0]):1.},-base.value+1e-5,'reservation_value_floor')
        price_objective=np.zeros(len(lp.names));price_objective[pay]=1.
        solved=_run(lp,price_objective,120.)
        if solved.status==2:raise ValueError('no admissible payment within configured range')
        if not solved.success:raise OptimizationError(f'direct reservation LP: {solved.message}')
        pricing_certificate=solver_certificate(solved,'minimum contract payment',lp,price_objective)
        pricing_certificate['compiled_sha256']=compiled_fingerprint(lp,price_objective)
        optimum=float(solved.x[pay])
        if abs(optimum)>=1e9-1:raise ValueError('reservation outside configured search range')
        lower=min(optimum,pricing_certificate['lp_dual_objective'])-1e-7
        amount=optimum+max(1e-7,price_tolerance/4)
        accepted=valued(amount,tie_break=tie_break)
        if accepted.status!='optimal' or accepted.value<base.value-1e-5:
            raise OptimizationError('direct reservation payment failed independent valuation')
        price=PriceResult(amount,'USD upfront' if payment=='upfront' else 'USD/GPU-hour',base.value,
                          accepted.value,2,(lower,amount),accepted.value-base.value)
        pricing_method='direct LP with dual lower bound and separately valued upper payment'
    else:
        price=_bracket_price(valued,base.value,'USD upfront' if payment=='upfront' else 'USD/GPU-hour',
                             price_tolerance,1e-5,1e9,1000 if payment=='upfront' else 1.)
        accepted=valued(price.amount,tie_break=tie_break)
        pricing_method='bracketed financed valuation; secondary tie-break only on final policies'
    qb=np.asarray(base.policy['position']);qa=np.asarray(accepted.policy['position'])
    imb=np.asarray(base.policy['initial_margin']).sum(axis=1)+np.asarray(base.policy['posted_vm']).sum(axis=1)
    ima=np.asarray(accepted.policy['initial_margin']).sum(axis=1)+np.asarray(accepted.policy['posted_vm']).sum(axis=1)
    lb=np.asarray(base.policy['risk_loss'])[p.tree.leaves];la=np.asarray(accepted.policy['risk_loss'])[p.tree.leaves]
    prob=p.tree.unconditional[p.tree.leaves]
    return {'reservation':price.to_dict(),'payment':payment,'pricing_method':pricing_method,'pricing_certificate':pricing_certificate,'instrument_names':m.names,
      'base_allocation':qb.tolist(),'accepted_allocation':qa.tolist(),'incremental_hedge':(qa-qb).tolist(),
      'root_incremental_hedge':(qa[0]-qb[0]).tolist(),'incremental_posted_collateral':(ima-imb).tolist(),
      'incremental_terminal_ES':expected_shortfall(la,prob,p.alpha)-expected_shortfall(lb,prob,p.alpha),
      'incremental_minimum_pretrade_cash':accepted.audit['minimum_pretrade_cash']-base.audit['minimum_pretrade_cash'],
      'base':base.to_dict(),'accepted':accepted.to_dict(),
      'interpretation':'The incremental hedge is accepted-book allocation minus existing-book allocation, on the same observed history.'}
