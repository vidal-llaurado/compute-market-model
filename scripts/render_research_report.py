#!/usr/bin/env python3
"""Render figures and a research report from the saved study results."""
from pathlib import Path
import sys,json,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compute_market.curves.portfolio import CurveCase,EconomyTape


PALETTE = ['#285884', '#258b91', '#5b9b82']
FIGURE_WIDTH = 210 / 25.4 - 2 * .82


def plot_style():
    if shutil.which('kpsewhich'):
        font = subprocess.run(['kpsewhich', 'LinLibertine_R.otf'], capture_output=True, text=True).stdout.strip()
        if font:
            font_manager.fontManager.addfont(font)
    try:
        font_manager.findfont('Linux Libertine O', fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError('Install Linux Libertine or the TeX Live fonts used by build_pdf.sh.') from error
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'text.color': '#292d32',
        'axes.labelcolor': '#292d32',
        'xtick.color': '#292d32',
        'ytick.color': '#292d32',
        'axes.facecolor': '#eaeaf2',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.prop_cycle': plt.cycler(color=PALETTE),
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'white',
        'grid.linewidth': .9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': False,
        'axes.spines.left': False,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'axes.labelpad': 7,
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        'lines.markerfacecolor': 'none',
        'legend.frameon': True,
        'legend.facecolor': '#eaeaf2',
        'legend.edgecolor': 'white',
        'legend.framealpha': .95,
        'legend.fancybox': False,
        'figure.constrained_layout.h_pad': .06,
        'figure.constrained_layout.w_pad': .06,
    })


def render(root=ROOT):
    plot_style()
    root=Path(root);src=root/'results/study';docs=root/'docs';figdir=docs/'figures';figdir.mkdir(exist_ok=True)
    load=lambda n:json.loads((src/(n+'.json')).read_text())
    s=load('summary');g=load('curve_greeks');ident=load('pricing_identities');filt=load('filtering');fit=load('initial_curve_fit');p=load('policy_comparison');rs=load('reservation_surface');options=load('options');swing=load('swing')
    case=CurveCase.load(src/'case.json');tape=EconomyTape.load(src/'holdout_tape.npz');names=['unhedged','static','adaptive'];titles=['Unhedged','Static financial inventory','Adaptive joint policy']
    def finish(fig,name,legend=None,columns=1):
        if legend:
            location='outside right center' if legend=='right' else 'outside lower center'
            fig.legend(loc=location,ncols=columns,borderaxespad=.5)
        fig.savefig(figdir/f'{name}.png',dpi=300);plt.close(fig)
    dates=np.linspace(.5,3,100);F=case.model.surface('G1',.5,tape.curve.factors[:,4],dates);qs=np.quantile(F,[.1,.5,.9],axis=0)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.65),layout="constrained");ax.fill_between(dates,qs[0],qs[2],color=PALETTE[0],alpha=.18,label='10–90% interval')
    ax.plot(dates,qs[1],color=PALETTE[0],label='Median');ax.plot(dates,[case.model.initial('G1',t) for t in dates],color=PALETTE[1],linestyle='--',label='Initial')
    ax.set(xlabel='Delivery date (years from origin)',ylabel='USD per GPU-hour');finish(fig,'curve_fan',legend='right',columns=1)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.8),layout="constrained");ax.plot(filt['times'],np.array(filt['truth'])[:,0],label='Simulated');ax.plot(filt['times'],np.array(filt['estimates'])[:,0],linestyle='--',label='Filtered')
    ax.set(xlabel='Observation time (years)',ylabel='Factor state');finish(fig,'filter',legend='right',columns=1)
    C=np.array(g['covariance']);sd=np.sqrt(np.diag(C));a=np.array(g['factor_delta'])*sd/1000;b=np.array(g['constrained_projection']['residual_delta'])*sd/1000;z=np.arange(len(a))
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,3.4),layout="constrained");ax.bar(z-.18,a,width=.36,label='Portfolio');ax.bar(z+.18,b,width=.36,label='After cash-constrained local hedge')
    ax.set_xticks(z,labels=['Level','Front','Back','GPU basis','Region','Power','Rates','Funding','Service'],rotation=30,ha='right');ax.set(ylabel='Exposure (USD thousands)\nper factor standard deviation');finish(fig,'greeks',legend='below',columns=2)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.8),layout="constrained")
    for n,title,style in zip(names,titles,['-', '--', ':']):
        with np.load(src/f'holdout_{n}.npz',allow_pickle=False) as z:w=np.sort(z['terminal_wealth']/1e6)
        ax.plot(w,np.arange(1,len(w)+1)/len(w),linestyle=style,label=title.replace('Static financial inventory','Static inventory'))
    ax.set(xlabel='Terminal wealth (USD millions)',ylabel='Cumulative probability');finish(fig,'holdout',legend='below',columns=3)
    stresses=[k for k in s['paths'] if k!='nominal'];diff=[(s['paths'][k]['unhedged']['loss_es']-s['paths'][k]['adaptive']['loss_es'])/1e6 for k in stresses]
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,4.1),layout="constrained");ax.barh([k.replace('_',' ').capitalize() for k in stresses],diff,color=[PALETTE[0] if v < 0 else PALETTE[1] for v in diff]);ax.axvline(0,color='#53616d',linewidth=.8);ax.invert_yaxis()
    ax.set(xlabel='Adaptive gain in worst-5% mean wealth (USD millions)');finish(fig,'stresses')
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.65),layout="constrained")
    for seed in sorted(set(r['seed'] for r in rs)):
        rr=[r for r in rs if r['seed']==seed];ax.plot([r['branches'] for r in rr],[r['reservation']['amount']/1000 for r in rr],marker='o',label=f'Seed {seed}')
    ax.set(xlabel='Conditional branches at each date',ylabel='Upfront payment (USD thousands)');finish(fig,'pricing_support',legend='right',columns=1)
    sy=load('system_summary');cal=load('calibration/summary');asset=load('assets/summary');bucket=load('risk/bucket_risk');ab=load('instrument_ablation');profile=load('calibration/latent_profile')
    vals=np.array(cal['estimates']);truth=np.array(cal['truth']);bands=np.array(cal['bootstrap']['quantiles_025_50_975']);z=np.arange(len(vals))
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.65),layout="constrained");ax.errorbar(z,vals,yerr=np.vstack([vals-bands[0],bands[2]-vals]),fmt='o',capsize=4,label='Estimate and bootstrap interval');ax.scatter(z,truth,color=PALETTE[1],marker='x',s=30,linewidths=1.3,label='Generating value')
    ax.set_xticks(z,labels=['Mean reversion','Diffusion','Physical drift','Pricing jump rate']);ax.set(ylabel='Parameter value\n(coordinate-specific units)');finish(fig,'parameter_recovery',legend='below',columns=2)
    with np.load(src/'assets/policy_outcomes.npz') as z:pv=z['cashflows'];upper=z['upper']
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.8),layout="constrained");ax.plot(np.sort(pv/1e6),np.arange(1,len(pv)+1)/len(pv),label='Frozen operating policy');ax.plot(np.sort(upper/1e6),np.arange(1,len(upper)+1)/len(upper),linestyle='--',label='Perfect-information bound')
    ax.set(xlabel='Discounted cohort value (USD millions)',ylabel='Cumulative probability');finish(fig,'asset_option',legend='below',columns=2)
    labels=[f"{b['market'].replace('_US','').replace('_SG',' SG')} {b['center']:g}y" for b in bucket['buckets']]
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,4.3),layout="constrained");ax.barh(labels,np.array(bucket['unit_pnl'])/1000);ax.invert_yaxis();ax.axvline(0,color='#53616d',linewidth=.8);ax.set(xlabel='USD thousands per bucket bump');finish(fig,'bucket_risk')
    with np.load(src/'liquidity.npz') as z:L=z['state']
    dates=tape.curve.times;lm=np.quantile(L,[.1,.5,.9],axis=0)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,2.8),layout="constrained");ax.fill_between(dates,lm[0],lm[2],color=PALETTE[0],alpha=.18,label='10–90% interval');ax.plot(dates,lm[1],color=PALETTE[0],label='Median');ax.set(xlabel='Time (years)',ylabel='Execution stress state');finish(fig,'liquidity',legend='right',columns=1)
    nominal=s['paths']['nominal'];cp=g['constrained_projection'];up=g['projection'];maxz=max(abs(r['z_score']) for r in ident['futures_martingales'])
    rows='\n'.join(f"| {title} | {nominal[n]['mean_wealth']/1e6:.3f} | {-nominal[n]['loss_es']/1e6:.3f} | {np.sqrt(nominal[n]['wealth_variance'])/1e6:.3f} |" for n,title in zip(names,titles))
    horizon=case.fleet.steps*case.fleet.year_fraction
    fundingrows='\n'.join(f"| {title} | {nominal[n]['mean_margin_cash_years']/horizon/1e6:.3f} | {nominal[n]['margin_p95']/1e6:.3f} | {round(nominal[n]['failure_fraction']*s['protocol']['holdout_paths'])} / {s['protocol']['holdout_paths']:,} |" for n,title in zip(names,titles))
    servicerows='\n'.join(f"| {title} | {nominal[n]['mean_delivery_shortfall_gpu_hours']:.2f} | {nominal[n]['mean_delivery_shortfall_fraction']:.4%} |" for n,title in zip(names,titles))
    promised=nominal['adaptive']['audit']['required_delivery_gpu_hours']
    stressrows='\n'.join(f"| {k.replace('_',' ').capitalize()} | {(s['paths'][k]['unhedged']['loss_es']-s['paths'][k]['adaptive']['loss_es'])/1e6:+.3f} | {round(s['paths'][k]['adaptive']['failure_fraction']*s['protocol']['stress_paths_per_case'])} |" for k in stresses)
    pr='\n'.join(f"| {b} | {1+b+b*b:,} | {min(r['reservation']['amount'] for r in rs if r['branches']==b):,.0f}–{max(r['reservation']['amount'] for r in rs if r['branches']==b):,.0f} | {(1-.9)*b:.1f} |" for b in sorted(set(r['branches'] for r in rs)))
    cr='\n'.join(f"| {name} | {tr:.5f} | {fitv:.5f} | {lo:.5f}–{hi:.5f} |" for name,tr,fitv,lo,hi in zip(cal['names'],truth,vals,bands[0],bands[2]))
    book=load('risk/financed_book_buckets');br='\n'.join(f"| {b['center']:g} years | {v:,.2f} | {err:.4f} |" for b,v,err in zip(book['buckets'],book['unit_pnl'],book['bump_consistency']))
    maxledger=max(v['audit']['maximum_cash_equation_error'] for a in s['paths'].values() for v in a.values());maxrecon=max(v['audit']['maximum_component_error'] for a in s['paths'].values() for v in a.values())
    maxpre=max(v['audit']['maximum_prefunding_violation'] for a in s['paths'].values() for v in a.values())
    failuretotal=sum(round(v['adaptive']['failure_fraction']*s['protocol']['stress_paths_per_case']) for k,v in s['paths'].items() if k!='nominal')
    mixed=ab['mixed_vs_futures_only'];outside=sy['risk']['out_of_span'];risk=sy['risk']
    funded_mean_pct=mixed['mean_wealth_change']/ab['futures_only']['mean_wealth']
    funded_tail_pct=mixed['es_reduction']/-ab['futures_only']['loss_es']
    funded_margin_change=nominal['adaptive']['margin_p95']-ab['futures_only']['margin_p95']
    funded_average_change=(nominal['adaptive']['mean_margin_cash_years']-ab['futures_only']['mean_margin_cash_years'])/horizon
    sd_reduction=1-np.sqrt(nominal['adaptive']['wealth_variance']/nominal['unhedged']['wealth_variance'])
    quad=max(abs(q['quadrature_change']) for q in sy['risk']['option_quadrature'])
    text=rf'''---
title: "Pricing Compute Obligations"
subtitle: "Numerical experiments"
date: ""
---

# Research design

A compute provider accepting a new contract commits capacity while taking on delivery, power and financing obligations. Its reservation price is the payment that preserves the optimized value of its existing book after these obligations are added. The experiments examine how operating flexibility and financial hedging affect this value, and how the cash needed to carry a hedge limits the protection it provides.

A shared curve model links compute benchmarks and basis, power, rates and funding. Through this common state, a market shock can change physical delivery costs, hardware appraisal and the value of financial claims, while also changing the operator's preferred dispatch and financing decisions. Execution liquidity responds to both idiosyncratic shocks and shared funding shocks.

All inputs are synthetic. The portfolio experiment trains on {s['protocol']['training_paths']:,} paths, evaluates {s['protocol']['holdout_paths']:,} independent paths, and uses {s['protocol']['conditional_atoms']} conditional innovations per decision. Thirteen named stresses use {s['protocol']['stress_paths_per_case']} paths each. The horizon is one year with eight operating intervals. Each policy faces the same market, physical, liquidity and counterparty events on each evaluation path.

The financial universe has eleven dated futures, two paid puts, a transferable fixed-strike power forward and a periodic compute cap. The adaptive policy observes the simulated factors in full and jointly chooses dispatch, financing and financial positions using a fitted continuation value for a specified future operating plan. Separate experiments examine what can be learned from noisy prices, how delivery exposure extends beyond the factor model, and how operating choices and financing enter asset values and reservation prices on finite scenario trees.

![The G1 delivery curve at six months. The band contains the pointwise 10th to 90th percentiles under the physical law.](figures/curve_fan.png)

# Pricing consistency

The portfolio calculations require consistent futures marks, discounting and option values under the shared pricing law. Its cumulant normalizes the positive futures curves; the electricity reference is additive. Across {ident['paths']:,} pricing-law paths, the twelve market/date checks depart from the futures martingale identity by at most {maxz:.2f} Monte Carlo standard errors. The two-year analytical discount factor is {ident['discount']['analytic']:.8f}, versus {ident['discount']['monte_carlo']:.8f} in simulation with standard error {ident['discount']['standard_error']:.8f}.

The two-year G1 futures reference is {ident['terminal_forward']['futures']:.6f}; the corresponding terminal-forward strike is {ident['terminal_forward']['analytic']:.6f}. Rate–compute covariance generates this difference; setting the rate loading to zero removes it. Splitting and recombining an arithmetic delivery strip changes its mark by {ident['delivery_overlap_error']:.2g}.

Option prices are calculated from the discounted affine transform. Doubling Fourier quadrature from 128 to 256 nodes changes the three tested put prices by at most {quad:.2g}. Independent pricing-law Monte Carlo, Gaussian-limit formulas, put–call parity and expiry settlement provide separate checks. Deterministic option prices allow the operating controller to compare funded protection with futures without simulation noise from option valuation at each decision.

# Estimating the curve and its dynamics

Calibration separates the delivery curve observed today from the parameters governing its evolution. The initial-curve fit recovers eight knots from dated and strip observations. Its data Jacobian has rank {fit['data_rank']}, with weighted quote RMSE {fit['weighted_quote_rmse']:.3f} in measurement-standard-error units.

To examine identification of the dynamics, the structural recovery experiment uses a two-factor affine model, {cal['independent_paths']:,} independent physical histories, 24 transitions per history and nine noisy option prices. Four parameters are estimated while nuisance jump parameters and loading anchors remain fixed. Six option prices at excluded expiries are reserved for checking the fitted model. This controlled identification test is separate from the nine-factor operating study, whose physical drift is fitted from its own training sample.

| Free coordinate | Generating value | Estimate | Bootstrap 2.5–97.5% |
|---|---:|---:|---:|
{cr}

All {cal['bootstrap']['converged']} of {cal['bootstrap']['replicates']} cluster-bootstrap refits converged, and the data-only Jacobian has full rank {cal['report']['data_rank']}. Held-out option-price RMSE is {cal['heldout_price_rmse']:.6f} per quoted unit. In a separate example, estimating physical drift from a time-zero futures reference alone gives data rank {cal['unidentified_data_rank']}, even with a prior penalty.

![Parameter recovery with factor normalization and nuisance parameters fixed. Bars show bootstrap percentiles across 24 refits.](figures/parameter_recovery.png)

Process variation, averaged across bootstrap fits, contributes {cal['predictive_uncertainty']['process_variance']:.6f} to estimated predictive variance; variation in the predicted means contributes {cal['predictive_uncertainty']['parameter_variance']:.6f}. This balance reflects the large synthetic sample; a short market history can leave substantially more parameter uncertainty.

# What a short price history identifies

The preceding recovery experiment pools many independent histories. To examine what a single short history reveals, a separate experiment observes three clean prices, each with measurement noise, at 145 dates over three years. It estimates one physical drift coordinate with the remaining law fixed. The fitted drift is {cal['latent_report']['estimates'][0]:.4f}, against the generating value 0.0300, while the two factor-state RMSEs are {cal['latent_factor_rmse'][0]:.4f} and {cal['latent_factor_rmse'][1]:.4f}.

The quasi-likelihood profile is nearly flat: over the entire tested range from −0.20 to +0.20, twice the objective change stays below {max(profile['twice_objective_above_grid_minimum']):.2f}. The short history gives more precise estimates of the factor states than of long-run drift. This profile measures sensitivity with nuisance coordinates fixed; finite-sample coverage would require a design-specific bootstrap.

Filtering is also examined in the nine-factor model, using fifteen prices per date to predict two withheld local prices. The held-out quote RMSE is {filt['held_out_quote_rmse']:.4f} per GPU-hour and the minimum observation rank is {filt['minimum_observed_rank']} of nine. These results describe inference from noisy observations; the portfolio policies below use fully observed simulated factors.

![The simulated level factor and its filtered estimate in the nine-factor noisy-price experiment.](figures/filter.png)

# Delivery exposure and risk outside the factor model

A delivery curve can move in ways that the chosen factors do not represent. To measure this exposure, a clean physical-margin strip is perturbed in {risk['bucket_count']} calendar-delivery buckets across two compute references and one power reference. Compute buckets use log coordinates; power buckets use price coordinates. Their Jacobian with respect to the factors has rank {risk['factor_span_rank']} of nine. For claims supported on the chosen delivery nodes, sensitivities mapped from delivery buckets differ from analytic factor Greeks by at most {risk['factor_bridge_max_error']:.4f} currency per factor unit, against sensitivities on the scale of millions of dollars. Half-size bumps change a bucket delta by at most {risk['bucket_half_step_max_change']:.5f}.

A constructed bucket shock has factor-projection norm {outside['factor_projection_norm']:.2g}, yet produces P&L of {outside['full_revaluation_pnl']:,.2f} when the portfolio is revalued. The local bucket estimate is {outside['linear_pnl']:,.2f}. This exposure arises from a curve movement outside the affine factor span. The shock is a sensitivity test with no assigned occurrence probability.

![Currency P&L per bucket bump. Compute bumps are 0.0001 in log price; power bumps are 0.01 currency/MWh.](figures/bucket_risk.png)

For a financed portfolio, the response to a curve change also depends on the decisions it induces. We therefore reoptimize after perturbing the G1 US initial curve, using the same conditional innovations for every perturbation. This calculation has three branches and two intervals, so the reported policy response is specific to that finite scenario tree.

| Delivery center | P&L for a 0.0001 log bump | Delta change after half bump |
|---|---:|---:|
{br}

These sensitivities serve different purposes in hedging. A covariance-weighted projection selects a hedge against factor exposure, while the bucket-shape projection measures how much delivery-curve risk lies outside the factor model. The full financial universe spans {up['span_rank']} local covariance directions. The constrained hedge respects signed position limits, turnover limits and a {cp['liquidity_budget']:,.0f} cash budget. Full conditional revaluation also includes nonlinear jump and funding risks.

# The value of operating flexibility

Hardware value depends on the owner's ability to change how it is used as margins evolve. The asset experiment values this flexibility for a 64-GPU cohort over three years, using 24 operating intervals. It includes startup and shutdown costs, standby expenses, two-interval minimum runs and an independent retirement-sale function. The interval rental rate is fixed when operating capacity is committed. The policy trains on {asset['training_paths']:,} pricing-law paths and is evaluated on {asset['paths']:,} independent paths.

\Needspace{{12\baselineskip}}

| Operating rule | Discounted value (USD) | Standard error (USD) |
|---|---:|---:|
| Switching and retirement | {asset['value']:,.0f} | {asset['standard_error']:,.0f} |
| Always operate | {asset['always_run_value']:,.0f} | {asset['always_run_standard_error']:,.0f} |
| Sell immediately | {asset['immediate_sale_value']:,.0f} | — |
| Perfect-information upper comparison | {asset['perfect_information_upper_mean']:,.0f} | {asset['upper_standard_error']:,.0f} |

The switching and retirement policy has a higher estimated value than either continuous operation or immediate sale. It operates for an average {asset['mean_run_intervals']:.2f} intervals and retires on {asset['retirement_fraction']:.1%} of paths. Its mean gap to the information upper bound is {asset['information_gap']:,.0f}, with paired standard error {asset['paired_gap_standard_error']:,.0f}. Every policy path satisfies its operating constraints and remains below the corresponding upper payoff up to rounding.

![Discounted operating cash flows on independent paths, compared with the upper bound from perfect information.](figures/asset_option.png)

Operating value informs investment decisions. The separate financed-fleet experiment uses different quantities and horizons from the 64-GPU experiment, with borrowing tied to lender appraisal and sales valued at executable proceeds.

# Hedging benefits and funding needs

The portfolio comparison measures protection after accounting for the cost of funding the hedge and delivering the committed service. All policies start with the same capacity, commitments, debt and cash. The worst-5% column averages terminal wealth over exactly 5% of probability mass, with fractional weight at the boundary observation. Higher mean and tail wealth are preferable; lower standard deviation indicates less dispersion.

| Policy | Mean wealth (USD m) | Worst-5% wealth (USD m) | SD (USD m) |
|---|---:|---:|---:|
{rows}

Relative to the unhedged policy, adaptive control raises worst-5% mean wealth by {p['adaptive_vs_unhedged']['es_reduction']:,.0f}, with a paired bootstrap interval of {p['adaptive_vs_unhedged']['paired_es_bootstrap_interval_95'][0]:,.0f}–{p['adaptive_vs_unhedged']['paired_es_bootstrap_interval_95'][1]:,.0f}. Mean wealth rises by {p['adaptive_vs_unhedged']['mean_wealth_change']:,.0f} and standard deviation falls by {sd_reduction:.1%}. It also raises worst-5% mean wealth by {p['adaptive_vs_static']['es_reduction']:,.0f} relative to static inventory. The bootstrap intervals measure sampling uncertainty under this synthetic market law.

The operator must deliver {promised:,.0f} GPU-hours of firm service on each path over the year. Delivery shortfall includes service left unfulfilled after financing failure.

| Policy | Mean shortfall (GPU-hours) | Share of promised service |
|---|---:|---:|
{servicerows}

The policies have the same mean delivery shortfall in this experiment, but their cash requirements differ. Initial margin is recorded after every trade and returned at settlement. Average posted margin weights each balance by the time it remains posted. Peak margin is the maximum requirement along a path; the table reports its 95th percentile across paths. Borrowing interest, debt fees, premiums and trading costs enter terminal wealth.

| Policy | Average posted IM (USD m) | Peak IM, p95 (USD m) | Failed paths |
|---|---:|---:|---:|
{fundingrows}

The adaptive policy's reduction in wealth dispersion comes with more posted margin than either comparison policy. The protection therefore depends on the operator's ability to keep cash available for collateral throughout the life of the hedge.

Instrument choice also changes the balance between protection and funding. To measure what the additional instruments contribute, we evaluate the adaptive policy with only the eleven futures. The full universe adds two paid puts, a transferable power forward and a periodic cap. Both versions use the same training sample, simulated paths and conditional innovations.

Adding these four instruments raises mean wealth by {mixed['mean_wealth_change']:,.0f} ({funded_mean_pct:.2%}) and worst-5% mean wealth by {mixed['es_reduction']:,.0f} ({funded_tail_pct:.2%}). The paired interval for the tail improvement is {mixed['paired_es_bootstrap_interval_95'][0]:,.0f}–{mixed['paired_es_bootstrap_interval_95'][1]:,.0f}. Peak margin at p95 changes by {funded_margin_change:+,.0f}, and average posted margin by {funded_average_change:+,.0f}. Here the additional instruments modestly improve tail wealth and reduce posted margin. Practical instrument selection also depends on access, funding terms and implementation costs.

![Terminal wealth under three policies on the same independent paths.](figures/holdout.png)

# Liquidity, credit and stress

The funding requirements above are measured on paths that already include stochastic spreads, executable depth and collateral multipliers linked to funding conditions and market declines. Supplier defaults become known when they occur, and the cash ledger records funded positions, segregated margin, payable liabilities and default recovery. The additional stress interventions examine how the hedge performs when particular sources of market, funding or delivery risk become more severe.

![The simulated execution state before the additional stress interventions.](figures/liquidity.png)

\Needspace{{22\baselineskip}}

| Named intervention | Adaptive gain in worst-5% wealth (USD m) | Adaptive failures / 512 |
|---|---:|---:|
{stressrows}

Under benchmark dislocation, the adaptive hedge reduces worst-5% mean wealth relative to the unhedged portfolio as the hedge loses alignment with the physical book. Margin, execution and liquidity interventions examine whether the operator can finance its positions through adverse cash demands. Margin calls and liquidity spirals produce financing failures even when hedging raises worst-5% mean wealth. These failures enter both the wealth and service outcomes.

![A negative bar means the hedge underperforms the unhedged physical book under that intervention.](figures/stresses.png)

# Reservation pricing and numerical resolution

The reservation calculation prices a new obligation by comparing optimized financed-portfolio values before and after acceptance under the same curve law. It fixes two decision intervals over one year and varies the number of branches and the simulation seed to examine sensitivity to numerical resolution. Futures, European options and the fixed-strike forward have cash flows compatible with this grid. The quarterly cap is excluded because its payment dates fall between the grid dates.

| Branches | History nodes | Payment range, three seeds (USD) | Local 90% tail mass |
|---|---:|---:|---:|
{pr}

Across these sampled trees, variation in reservation payments is larger than the certified interval for any one tree. With independent draws at each branch count, increasing the number of branches need not move the price monotonically toward a limit. The results also report conditional tail weights and whether the risk evaluation concentrates on the worst state.

![Reservation payments from independent scenario trees over the same horizon.](figures/pricing_support.png)

# Accounting checks and reproducibility

The wealth and funding comparisons depend on recording each cash flow at the appropriate date. Across all portfolio cases, the largest cash-equation residual is {maxledger:.2g} currency units and the largest independent terminal-wealth reconstruction error is {maxrecon:.2g}. The largest numerical prefunding residual is {maxpre:.2g}. Distributions, premium exchanges, debt principal, interest, transaction costs and hardware sales remain separate accounting components.

`results/study/` contains complete valuation cases and all training, holdout and stress tapes. `assets/` stores the operating rule, independent outcomes and upper comparison. `risk/` stores bucket shocks, factor bridges, financed-book bumps and transform checks. `calibration/` stores raw synthetic histories, noisy quotes, anchors, parameter draws, held-out repricing and the latent profile. `pricing/` retains complete finite programs and solver certificates.

Verification commands and the original release's validation record are in `docs/VALIDATION.md`.
'''
    (docs/'RESEARCH_REPORT.md').write_text(text)
    print(f'Saved ten figures and research report to {docs}.')

if __name__=='__main__':render()
