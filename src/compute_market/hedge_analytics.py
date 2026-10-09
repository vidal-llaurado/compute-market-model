"""Risk and liquidity diagnostics using common paths and explicit units."""
import numpy as np
from .statistics import mean_interval,tail_summary,wilson_interval,paired_comparison
from .risk import expected_shortfall


def summarize_paths(result,bootstrap=200,seed=31):
    wealth=result.terminal_wealth;loss=-wealth;p=np.full(len(wealth),1/len(wealth))
    return {'policy':result.policy,'wealth':mean_interval(wealth),'loss_tail':tail_summary(loss,.95,bootstrap,seed),
      'tail_deviation':expected_shortfall(loss-loss.mean(),p,.95),'variance':float(loss.var(ddof=1)),
      'financing_failure':wilson_interval(result.financing_failure),'delivery_breach':wilson_interval(result.delivery_breach),
      'mean_execution_cost':float(result.transaction_cost.mean()),'mean_financial_cash_pnl':float(result.hedge_cash_pnl.mean()),
      'mean_credit_loss':float(result.credit_loss.mean()),'mean_preterminal_blocks_sold':float(result.total_asset_sales.mean()),
      'peak_margin_quantiles':np.quantile(result.maximum_margin,[.5,.95,.99]).tolist(),
      'funding_need_quantiles':np.quantile(result.maximum_funding_need,[.5,.95,.99]).tolist(),
      'cash_headroom_quantiles':np.quantile(result.minimum_cash_headroom,[.01,.05,.5]).tolist(),
      'audit':result.audit}


def effectiveness(unhedged,hedged,bootstrap=200,seed=32):
    if len(unhedged.terminal_wealth)!=len(hedged.terminal_wealth):raise ValueError('matched paths required')
    a=-unhedged.terminal_wealth;b=-hedged.terminal_wealth;n=len(a);p=np.full(n,1/n)
    avar=float(a.var(ddof=1));bvar=float(b.var(ddof=1))
    improvement=expected_shortfall(a,p,.95)-expected_shortfall(b,p,.95)
    rng=np.random.default_rng(seed);sample=[]
    for _ in range(bootstrap):
        ix=rng.integers(0,n,n);sample.append(expected_shortfall(a[ix],p,.95)-expected_shortfall(b[ix],p,.95))
    return {'policy':hedged.policy,'paired_wealth_change':paired_comparison(unhedged.terminal_wealth,hedged.terminal_wealth),
      'ES_reduction_dollars':improvement,'ES_reduction_interval':np.quantile(sample,[.025,.975]).tolist() if sample else None,
      'variance_reduction_fraction':None if avar<=1e-12 else 1-bvar/avar,
      'tail_deviation_reduction_dollars':expected_shortfall(a-a.mean(),p,.95)-expected_shortfall(b-b.mean(),p,.95),
      'funding_need_95_change':float(np.quantile(hedged.maximum_funding_need,.95)-np.quantile(unhedged.maximum_funding_need,.95)),
      'peak_margin_95_change':float(np.quantile(hedged.maximum_margin,.95)-np.quantile(unhedged.maximum_margin,.95)),
      'failure_probability_change':float(hedged.financing_failure.mean()-unhedged.financing_failure.mean()),
      'direction':'Positive ES reduction and variance reduction are improvements; positive funding-need change consumes more liquidity.'}


def centered_tail_contributions(components,alpha=.95):
    """Additive ES contributions from one consistent fractional empirical tail.

    Components are signed P&L. This attribution is of the supplied cash ledger,
    not a unique causal split among interacting economic risk factors.
    """
    a=np.asarray(components,float)
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all() or not 0<=alpha<1:raise ValueError('finite component matrix and valid alpha required')
    losses=-(a-a.mean(axis=0));total=losses.sum(axis=1);order=np.argsort(-total,kind='stable')
    weight=np.zeros(len(a));remaining=1-alpha
    for i in order:
        used=min(1/len(a),remaining);weight[i]=used/(1-alpha);remaining-=used
        if remaining<=1e-14:break
    contribution=weight@losses
    return {'contributions':contribution.tolist(),'total_centered_ES':float(contribution.sum()),
            'sum_error':abs(float(contribution.sum())-expected_shortfall(total,np.full(len(a),1/len(a)),alpha))}
