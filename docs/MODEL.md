---
title: "Pricing Compute Obligations"
subtitle: "Stochastic term structures, portfolio valuation and control"
date: ""
---

# Market and balance sheet

A compute operator commits capacity today against uncertain rental prices, power costs and financing conditions. A new contract can provide cash, consume scarce capacity and alter the cost of hedging the existing book. We study the payment required to preserve the book's optimized value after acceptance, under the operator's physical and financing constraints. This reservation price depends on how the contract changes the operator's business.

Shared stochastic factors describe compute benchmarks, GPU and regional basis, electricity, interest rates and commercial funding spreads at every delivery date. Contracts turn these curves into payments and delivery obligations; physical capacity, debt and cash determine which choices are feasible. The numerical examples use synthetic inputs to study the resulting contract prices and operating decisions.

A clean futures reference represents a continuously settled financial exposure, while a terminal forward exchanges a price difference at a specified future settlement date. A physical reservation price compensates the operator for committing capacity, credit and liquidity. Their values depend on the same market factors, with differences arising from settlement and delivery rights and, for the physical obligation, the operator's existing book.

The complete state is

$$
S_t=(X_t,Z_t^{liq},K_t,\mathcal B_t,b_t,D_t,M_t,\theta_{t-},H_t^{cp}),
$$

where $X_t$ contains market factors and $Z_t^{liq}$ describes execution liquidity. The physical book consists of inventory and availability $K_t$ and customer commitments $\mathcal B_t$. The balance sheet records unrestricted cash $b_t$, debt $D_t$ and posted collateral $M_t$, together with the inherited financial position $\theta_{t-}$. Counterparty survival is recorded in $H_t^{cp}$.

# Delivery conventions and information

## Calendar time, tenor and units

The calendar delivery date is $T$, and $x=T-t$ is time to delivery. Financial time is in years; operating quantities use the actual number of GPU-hours or MWh in each interval. A one-year example uses eight delivery intervals of 1,095 hours. Contractual expiries remain fixed as the valuation date advances.

A compute service identifies generation, region and service eligibility. Positive compute curves have units of currency per contracted GPU-hour. Electricity curves have units of currency per MWh. Interest and funding rates are annual decimals. A contract multiplier converts each quoted price change into cash.

Hardware-time reservations and useful-work commitments require different quantities. For workload $w$, configuration $j$ delivers $\nu_{wj}$ useful units per GPU-hour, subject to quality, memory, topology and latency eligibility. Only eligible configurations enter the substitution set. The workload simulator tracks remaining work, checkpoint losses, cluster placement and deadlines. Benchmark methodology must fix these conditions before measuring $\nu_{wj}$ [R12].

## Observable state

Let $\mathcal H_t$ contain the full history of the model and its observations, and let $\mathcal G_t\subseteq\mathcal H_t$ be the information available when a decision is made. A quote or policy is $\mathcal G_t$-adapted. The explicit pricing and forecasting identities below condition on $\mathcal H_t$. The operating control study uses full state information, $\mathcal G_t=\mathcal H_t$. The separate filtering experiment observes noisy quotes:

$$
Y_i=\mathcal O_i(t_i,X_{t_i};\Theta)+\epsilon_i.
$$

Each observation operator reflects the instrument and its settlement terms: a dated futures reference, a terminal forward, an arithmetic delivery strip or a physical quote with supplied contractual adjustments. Observation records distinguish indicative references from executable quotes and specify their sources and uncertainty. Adjusting physical quotes for delivery and cancellation rights prevents contractual price differences from being fitted as market factors.

When factors are hidden, the statistical state is the conditional distribution of $X_t$ given $\mathcal G_t$. For a fixed payoff, its conditional discounted expectation under reduced information is the posterior average of the corresponding full-state expectation under the relevant valuation measure. The included extended Kalman filter propagates exact conditional transition means and covariances, linearizes each nonlinear observation and uses a Joseph-form covariance update. Its Gaussian posterior approximation is local because the observations are nonlinear and the factor law has jumps. The noisy-quote experiment measures filtering performance [R21]; it does not implement control on a posterior distribution.

# Coupled stochastic curves

## Function-valued market state

Contracts, options and equipment appraisal all depend on the delivery curves. We specify their dynamics directly, with physical constraints governing delivery. A structural spot model would derive the curves from assumptions about supply and demand. The construction below uses a finite-dimensional family of curves closed under time shifts [R25].

For each market $a$, define the Musiela curve $\mathcal F_t^a(x)=F^a(t,t+x)$. Calendar roll contributes the transport operator $\partial_x$. A generic log-curve representation is

$$
df_t^a(x)=\{\partial_x f_t^a(x)+\mu_t^a(x)\}\,dt
+\sigma_t^a(x)\,dW_t+dJ_t^a(x),
\qquad f_t^a=\log\mathcal F_t^a.
$$

The drift depends on the interpretation of $F$. A clean futures curve is a martingale under its pricing measure. The affine construction below imposes this restriction through its compensator, following the forward-curve organization of HJM and its commodity extensions [R15–R18].

## Finite-dimensional realization

Shared factors make it possible to simulate the full set of delivery curves with a finite state. Their decay rates determine how strongly each factor affects nearby and distant deliveries.

Let $K=\operatorname{diag}(\kappa_1,\ldots,\kappa_d)$, with $\kappa_i\geq0$. Under measure $m\in\{P,Q\}$,

$$
dX_t=(-KX_t+\eta_m)\,dt+L\,dW_t^m+dJ_t^m.
$$

Here $J^m$ is a raw compound-Poisson process with intensity $\lambda_m$ and Gaussian marks $Z_m\sim N(m_m,V_m)$. The pricing law uses $\eta_Q=0$. Calibration to the supplied initial curves takes $X_0=0$. Simulation may start from another factor state, representing a displacement from those reference curves. With $C=LL^\top$, the cumulant of the driving increment is

$$
\psi_m(z)=\eta_m^\top z+\frac12z^\top Cz
+\lambda_m\left[\exp\!\left(z^\top m_m+\frac12z^\top V_mz\right)-1\right].
$$

Each curve has a loading vector $c_a$ and maturity shape

$$
\ell_a(x)=e^{-Kx}c_a.
$$

These shapes are closed under differentiation and time shift: $\partial_x\ell_a=-K\ell_a$. The finite state therefore represents the chosen curve family exactly as contracts age. An empirical principal-component basis whose span changes under time shift would require a projection rule and a measured closure error [R16, R25].

The common factors express level, near-delivery scarcity, long-end slope, generation basis, regional basis, power, interest, funding liquidity and service basis. Shared Brownian covariance and jump marks transmit an event through all affected curves. Fast decay concentrates its effect near delivery; zero decay supplies a persistent level component. The factor law and loading vectors determine covariance between curves. A long-end compute shock affects both rental margins and the appraisal supported by those margins, while separate resale and execution factors describe risks outside that relationship.

## Positive compute references

For a calibrated initial curve $F_0^a(T)>0$, define

$$
A_a(t,T)=\int_{T-t}^{T}\psi_Q(\ell_a(v))\,dv,
$$

$$
\boxed{
F^a(t,T)=F_0^a(T)
\exp\!\left[\ell_a(T-t)^\top X_t-A_a(t,T)\right].
}
$$

For fixed $T$, the state drift cancels the derivative of $\ell_a(T-t)$. The remaining exponential compensator is exactly $\psi_Q(\ell_a(T-t))$. The driving process has independent increments, so

$$
E_Q[F^a(u,T)\mid\mathcal H_t]=F^a(t,T),\qquad t\leq u\leq T.
$$

This identity prices a continuously settled futures reference. It requires no storage relation of the form spot times $e^{r(T-t)}$ for compute.

In log-Musiela coordinates, away from initial interpolation knots,

$$
df_t^a(x)=\{\partial_x f_t^a(x)-\psi_Q(\ell_a(x))\}\,dt
+\ell_a(x)^\top L\,dW_t^Q+\ell_a(x)^\top dJ_t^Q.
$$

The cumulant includes the raw-jump compensation. Omitting it would create drift in the clean futures reference.

## GPU, regional and service basis

For local service $j$ and its reference benchmark $0$,

$$
\log\frac{F^j(t,T)}{F^0(t,T)}=
\log\frac{F_0^j(T)}{F_0^0(T)}
+[\ell_j(T-t)-\ell_0(T-t)]^\top X_t
-[A_j(t,T)-A_0(t,T)].
$$

This basis curve inherits its maturity shape from the difference in loadings. Some basis factors have no traded hedge. Inferred local reference curves support valuation and observation filtering; the instrument registry separately states where an executable financial claim exists.

## Normal electricity curves

Electricity can be negative, so its reference uses an additive specification. Let $\bar j_Q=\lambda_Qm_Q$. Then

$$
F^a(t,T)=F_0^a(T)+\ell_a(T-t)^\top X_t
-\int_{T-t}^{T}\ell_a(v)^\top\bar j_Q\,dv.
$$

The same factors and pricing law apply. The additive jump compensation makes the fixed-delivery reference a $Q$-martingale while permitting negative prices [R17, R18]. Clipping those prices would change expectations and option values.

## Forecasting and pricing laws

The forecasting and pricing laws share factor loadings, decay and diffusion covariance. Their drift and jump laws may differ. For a positive reference,

$$
E_P[S_T^a\mid\mathcal H_t]
=F^a(t,T)
\exp\!\left\{\int_0^{T-t}
[\psi_P(\ell_a(v))-\psi_Q(\ell_a(v))]dv\right\}.
$$

The futures reference equals expected spot when this risk-premium adjustment vanishes. Statistical estimation uses the physical law; clean derivative valuation uses the pricing law. The reference study fixes the structural coefficients when fitting physical drift. Market data must identify the pricing-law parameters that affect tradable claims. Sparse observations can leave those parameters unidentified even when the fitted curves are smooth.

The change of measure restricts Brownian drift adjustments to the diffusion span and requires compatible support for the two Gaussian jump laws. These conditions exclude singular changes of measure within the constant-coefficient family. Shared jump laws may still generate nonlinear payoff risk that finitely many traded futures cannot span [R19, R20].

# Interest, discounting and commercial funding

## Discounting with shared factors

Discounting uses the same factors as compute and power prices, retaining the dependence between future payments and the rates used to value them.

Let $a_r$ be the rate-loading vector and

$$
B(x)=\int_0^x e^{-Kv}a_r\,dv.
$$

For a calibrated initial instantaneous forward rate $r_0(t)$, the short rate is

$$
\phi(t)=r_0(t)+\psi_Q(-B(t)),\qquad
r_t=\phi(t)+a_r^\top X_t.
$$

The zero-coupon bond is

$$
\boxed{
P(t,T)=\exp\!\left[
-\int_t^T\phi(u)du-B(T-t)^\top X_t
+\int_0^{T-t}\psi_Q(-B(v))dv
\right].
}
$$

At $X_0=0$, this reproduces $P(0,T)=\exp[-\int_0^T r_0(u)du]$. The endpoint and integrated factor process are simulated jointly, so pathwise discounting retains covariance with compute prices. The affine rate specification permits negative rates [R19].

## Futures versus terminal forwards

A terminally settled forward delivers $S_T-K$ at $T$. Its zero-value strike is

$$
K^{\mathrm{fwd}}(t,T)
=\frac{E_Q[e^{-\int_t^T r_u du}S_T\mid\mathcal H_t]}{P(t,T)}.
$$

For the positive compute curve, write $\tau=T-t$. Then

$$
K^{\mathrm{fwd}}(t,T)=F(t,T)e^{\chi(t,T)},
$$

$$
\chi(t,T)=\int_0^\tau
\left[\psi_Q(\ell(v)-B(v))-\psi_Q(-B(v))-\psi_Q(\ell(v))\right]dv.
$$

This is the rate–compute convexity adjustment. When the rate loading is zero, $\chi=0$. The normal-curve implementation uses the corresponding cumulant-gradient identity. Continuously margined futures, terminal forwards and periodic physical invoices consequently have separate settlement operators.

## Commercial funding

A commercial spread is modeled as

$$
s_t(x)=s_0(t+x)\exp[\ell_s(x)^\top X_t].
$$

It shares the market factors but is not constrained to be a clean traded-futures martingale. Debt contracts determine which reference rate, spread, reset dates and coupon conventions apply. Changes in the funding curve affect new and refinanced borrowing; outstanding fixed-coupon debt keeps its coupon.

# Delivery claims and exercise rights

## Arithmetic delivery aggregation

For delivery quantities $q_i$ at dates $T_i$, a futures strip reference is

$$
\overline F_t=\frac{\sum_iq_iF(t,T_i)}{\sum_iq_i}.
$$

A fixed-price strip with cash settlement at each delivery date has strike

$$
K_t=\frac{\sum_iq_iP(t,T_i)K^{\mathrm{fwd}}(t,T_i)}
{\sum_iq_iP(t,T_i)}.
$$

Delivery quantities weight prices arithmetically. Continuous strips integrate against delivery weights, with quadrature split at initial-curve knots. Partitioning a delivery interval and recombining its quantities preserves the original strip value [R16, R18]. A contract with one later settlement date requires discounting to that date.

Physical service rights also specify supplier availability, performance, termination, credit and volume terms. These terms enter the portfolio valuation below. A clean cap settles in cash, leaving the operator to procure physical capacity.

## European claims

A claim paying $g(F(e,T))$ at expiry $e\leq T$ has value

$$
V_t=P(t,e)E_{Q^e}[g(F(e,T))\mid\mathcal H_t].
$$

In the diffusion-only model, the reference is lognormal under the expiry-forward measure. The Gaussian option formula includes the covariance adjustment from the random discount factor. With jumps, prices follow from inversion of the discounted affine transform. An independent Monte Carlo calculation jointly simulates factors and integrated rates to value the same payoff; its standard error measures sampling uncertainty. Put–call parity and the limit of zero rate convexity provide further checks.

## Flexible exercise

The swing example allows at most one exercise per specified date and at most $N$ exercises overall. Exercise pays $(S_{t_i}-K)^+$ per unit. A backward least-squares recursion estimates continuation values for each possible number of remaining rights. The fitted regressions are held fixed during evaluation on independent paths. Exercise decisions depend on current observations and remaining rights [R22].

Other payoff functions, including monthly-average compute caps and compute–power margin floors, use the same simulated curves. The main control study includes funded puts, a fixed-strike power forward and a periodic cap alongside dated futures. The swing exercise policy is trained separately on the same curve state.

# Physical assets and financing

The market curves determine prices, but the operator must also deliver the promised service and finance the equipment that supplies it. These requirements constrain dispatch, borrowing and sales of hardware.

## Operating feasibility

Let $y_t$ denote dispatch and procurement decisions. Physical feasibility requires

$$
A_ty_t\leq C_t^{\mathrm{usable}},\qquad y_t\in\mathcal Y_t.
$$

The set $\mathcal Y_t$ includes cluster placement, eligibility, concurrency and indivisibility. Customer commitments consume capacity before discretionary sales. Outages reduce usable resources, and replacement procurement is subject to specified prices and limits. Job-level placement matters when an aggregate pool would conceal fragmented capacity.

The curve-control experiment operates four compatible fleet pools at block level. Separate placement and workload calculations retain the constraints of individual jobs. Financial hedges change the cash available to meet these obligations; delivery requires capacity or replacement procurement.

## Equipment values

A hardware cohort has three values with different uses:

$$
A_t^{econ},\qquad A_t^{lender},\qquad A_t^{liq}(q).
$$

These values answer different questions about the same equipment. Economic value reflects future operating choices and supports investment decisions. Lender appraisal determines the borrowing base under the contract's eligibility rules. Liquidation value measures the cash obtainable from selling a specified quantity under prevailing execution conditions.

## Operating, idle and retired states

On decision dates $t_n$, let $s_n$ denote idle, operating, or the number of mandatory operating intervals remaining. Retirement is absorbing. The available actions are run, idle and retire. While a minimum run is incomplete, the cohort must continue operating. The action determines the next operating state $\Gamma(s_n,a_n)$.

For the reference single-cohort problem, the current interval's capacity is sold at its observed opening rental rate. Its interval cash receipt is

$$
R_n^{run}=N\,8760\,\Delta t_n\,u
\{F^{compute}(t_n,t_n)-eF^{power}(t_n,t_n)-o\}
-\mathrm{startup}(s_n).
$$

Idle capacity pays standby and any shutdown cost. Retirement receives independently specified resale proceeds, less shutdown cost. These cash flows are known at the decision date. A contract with intraperiod price resets would require additional settlement dates.

Under a specified valuation law $M$, the operating Bellman equation is

$$
A_n(x,s)=\max_{a\in\mathcal A(s)}
\left\{R_n(x,s,a)+
E_M[D_{n,n+1}A_{n+1}(X_{n+1},\Gamma(s,a))\mid X_n=x]\right\}.
$$

Under a selected pricing law $Q$, this equation gives the equipment's unencumbered economic value. Under $P$, it gives expected discounted operating cash flow under the forecasting law. Debt and the owner's risk preferences enter the separate financed-portfolio problem, so debt costs are counted there once.

Backward least-squares simulation estimates each action's continuation value on training paths [R22, R24]. The resulting policy is held fixed during evaluation on independent paths. A second dynamic program uses the entire future of each path to give an upper bound under perfect information. For every evaluated path,

$$
PV(\text{admissible policy})\leq PV(\text{perfect information}).
$$

The fitted policy's expected value is a lower bound on the optimum for this finite grid; the expected value under perfect information is an upper bound. Monte Carlo estimates of both expectations have sampling error. The gap combines the error from approximating the optimal policy with the value of future information.

## Lender appraisal from the curves

The fleet appraisal integrates margins to the fixed calendar end of economic life, $T_g^{\mathrm{life}}$. Its forward-margin proxy is

$$
\widetilde A_t^g=
\int_t^{T_g^{\mathrm{life}}} P(t,u)\,u_g\,h_g
\left[K_g^{\mathrm{fwd}}(t,u)-e_gK_P^{\mathrm{fwd}}(t,u)-o_g\right]^+du
+P(t,T_g^{\mathrm{life}})R_g.
$$

Here $h_g$ converts years to deliverable GPU-hours, $u_g$ is appraisal utilization, $e_g$ is MWh per GPU-hour and $o_g$ is other marginal cost. A fixed initial scale matches the supplied equipment mark. An independent appraisal/liquidity basis captures resale conditions outside rental margins; secondary-market observations can inform this component [R06, R07].

The appraisal values forward margins with fixed utilization, while the operating problem values the right to change dispatch. Lender marks may lag the appraisal and apply eligibility reductions.

Cash flows through the operating horizon enter operating wealth. The terminal hardware mark covers life after that horizon, so rental receipts are counted once.

## Liquidation

The lender's appraisal follows its contractual update dates and eligibility rules. A sale converts equipment into cash at an execution-dependent price. Its proceeds can be represented by

$$
A^{liq}(q)=a\,q\left(1-h-\zeta q/d\right)^+,
$$

where $a$ is the reference resale mark, $h$ an immediate-sale haircut and $d$ market depth. Liquidation losses depend on the quantity sold and enter the cash and collateral ledger. The asset experiment reports economic value per cohort; the financed-book experiment measures collateral for the whole fleet. Each specifies its own quantity and horizon.

## Debt and borrowing capacity

For an eligible collateral pool,

$$
BB_t=\sum_g\alpha_g A_{g,t}^{\mathrm{eligible}}
+\sum_c\beta_c R_{c,t}^{\mathrm{eligible}}.
$$

Receivables and equipment must satisfy the contract's eligibility rules, and each asset can be pledged only once. The synthetic curve case sets the receivable advance to zero. The amount available to draw depends on the facility commitment, drawn balance, borrowing base, draw limits and suspension conditions.

Borrowing and principal repayment change cash and debt together. Interest on the post-action balance accrues over the interval and is paid at the next date, including the final interval before terminal repayment. Scheduled term amortization is paid when due. Refinancing requires a specified financing arrangement even when asset value is positive.

With collateral value $A$, debt $D$, advance rate $\alpha$, available cure cash $b$, and sale proceeds $\chi x$ from selling marked collateral $x$, compliance requires

$$
D-b-\chi x\leq\alpha(A-x).
$$

When $\chi>\alpha$, the minimal sale in this linear example is

$$
x_{\min}=\frac{(D-b-\alpha A)^+}{\chi-\alpha}.
$$

The numerical repair routine includes liquidation impact and physical breakpoints. Selling more pledged equipment need not improve the borrowing-base gap. A cure may instead require cash, new capital or revised financing [R09].

# Portfolio value and reservation pricing

## Value of the financed book

Capacity and borrowing constraints determine which policies the operator can follow. Its risk preferences determine how it values the resulting cash flows. Under full state information, let $V_t(S_t)$ be the optimized value of the financed operating book. A recursive risk mapping is

$$
\mathcal U_t(Y)=(1-\omega)E_P[Y\mid\mathcal G_t]
-\omega ES_{\alpha,t}(-Y).
$$

With distributions $d_t$, injections $e_t$ and admissible controls $a_t$,

$$
V_t(s)=\sup_{a_t\in\mathcal A_t(s)}
\left\{d_t-e_t+\mathcal U_t[\delta_{t,t+1}V_{t+1}(S_{t+1})]\right\}.
$$

Each cash flow enters once, through either the state transition or the immediate objective. Terminal value includes remaining hardware, debt, financial closeout and unfulfilled obligations. Clean derivative valuation uses $Q$; private risk management uses $P$ and the selected risk mandate. Both use the same curve state [R01–R03, R07, R08].

The finite-control program uses nested conditional risk at its stated decision dates, with probabilities generated from the curve transition law. Repeating a fixed conditional tail penalty at more frequent dates changes the preference. The path controller instead applies conditional expected shortfall to its fitted continuation problem; terminal loss and financing outcomes are reported separately. Optional rectangular probability ambiguity changes the risk evaluation while preserving the specified clean pricing law [R04].

## Contract price

Let $\Phi(s;c,A)$ be the state after accepting obligations $c$ and receiving upfront payment $A$. The reservation payment is

$$
\boxed{
A^{\min}(c\mid s)=\inf\{A:V(\Phi(s;c,A))\geq V(s),\;
\Phi(s;c,A)\text{ has a feasible policy}\}.
}
$$

Receiving payment can make a previously infeasible financing plan feasible. The payment therefore enters the state before the accepted book is reoptimized. Opening and closing periodic rates solve the corresponding equation with receipts on their contractual dates. In continuous linear cases, the payment is a decision variable: the solver supplies a lower bound, and a separate valuation checks the upper payment. Integer cases use bracketed valuation.

A clean strip mark provides a market reference for the obligation. The operator's reservation price also reflects delivery rights, optionality, financing and risk. Payment can relax a cash constraint, and delivery can consume capacity or offset an existing exposure. Different books can therefore have different reservation prices under the same forecast. A sum of separately valued contract features is sufficient only when these effects separate.

## Marginal values of capacity and cash

Private capacity value is the change in $V$ after adding eligible inventory at a delivery interval. Market scarcity affects reference prices, while placement, financing and existing commitments determine the value of an additional block to this book. This value can jump at constraint boundaries.

For a local sensitivity, suppose a zero-size contract leaves the state unchanged and requires zero payment. In a neighborhood of this base point, assume $V$ and the contract state map are continuously differentiable, the reservation solution is interior to the feasible payment range, and the value constraint binds locally. Payment adds one unit to unrestricted cash, with $m_b(s)=\partial_bV(s)>0$. If $\dot s_c$ is the state derivative of the contract at zero size and zero payment, the implicit-function theorem gives

$$
\frac{dA^{\min}}{d\epsilon}\bigg|_0
=-\frac{\nabla V(s)^\top\dot s_c}{m_b(s)}.
$$

This expresses the capacity and risk burden in units of available cash. A local commercial quote additionally incorporates customer acceptance and cross-configuration substitution. In the one-product exponential-arrival limit, the price is marginal cash cost plus capacity opportunity value divided by $m_b$, plus the inverse price-elasticity parameter [R05].

# Curve Greeks and incomplete-market hedging

## From curve perturbations to factors

For positive curve $a$,

$$
\partial_XF^a(t,T)=F^a(t,T)\ell_a(T-t),\qquad
\partial^2_{XX}F^a(t,T)=F^a(t,T)\ell_a(T-t)\ell_a(T-t)^\top.
$$

A delivery-strip Greek is the same arithmetic weighted sum of its point Greeks. Portfolio Greeks include changes in the optimal policy, financing and collateral. The experiment estimates them by revaluing the finite control problem with the same random draws after each perturbation. Holding the policy fixed would measure sensitivity under that policy alone.

A functional curve derivative $\delta V/\delta f^a(x)$ projects into factor coordinates as

$$
g_k=\sum_a\int \frac{\delta V}{\delta f^a(x)}\ell_{a,k}(x)\,dx.
$$

Operating contracts, equipment appraisal and financial instruments all express risk in the same factor coordinates. Greeks measure local sensitivity; separate jump and funding stresses capture nonlinear changes [R16, R20, R23].

## Bucketed delivery risk and shape risk

Dated curve buckets express risk in the delivery periods a commercial desk trades. For each curve, piecewise-linear tent functions $h_i(T)$ partition its delivery grid. Positive compute and funding curves use log-price coordinates; electricity and instantaneous forward rates use absolute-price coordinates. A bucket perturbation changes the initial calibrated curve:

$$
f_0^a(T)\mapsto f_0^a(T)+\sum_i z_i h_i(T).
$$

The affine compensator and pricing law remain unchanged, so the bumped model still satisfies its pricing identities. Revaluation supplies

$$
d_i=\partial_{z_i}V,\qquad G_{ij}=\partial^2_{z_i z_j}V.
$$

Central differences use the same random draws for both perturbations. Repeating the calculation with a half-size bump measures sensitivity to the bump size. Reoptimizing the book includes policy changes, while valuing a fixed claim measures its clean price sensitivity.

Let $J_{ik}$ be the factor loading at bucket node $i$. The bucket sensitivities then give the factor exposure

$$
g^{bucket}=J^\top d.
$$

For linear claims supported on the bucket nodes, this relation is exact up to differentiation error. Delivery integrals extending between nodes and nonlinear values obtained by reoptimization can differ from directly computed factor Greeks because of interpolation. The software reports this residual.

The projector $P_J=JJ^+$ acts on bucket-shock coordinates. The vector $(I-P_J)^\top d$ identifies sensitivity to curve shapes outside the model's factor span. It is distinct from unhedgeable risk inside the factor model. Such off-span shocks are deterministic model-risk tests; the affine simulation does not assign probabilities to them.

## Hedge projection

Let $g$ be the portfolio value delta, $H$ the matrix whose rows are instrument value deltas, and $C_h$ the conditional factor covariance over the hedge horizon. The unconstrained local projection solves

$$
\min_q(g+H^\top q)^\top C_h(g+H^\top q).
$$

We solve this problem with a pseudoinverse after whitening by the covariance. The residual measures the factor exposure left after projection. Duplicate instruments add no hedge dimensions, and exposure outside the traded span remains even with unrestricted notional.

For a complete delta span, a finite jump can still leave

$$
V(X+z)-V(X)+q^\top[H_{\mathrm{price}}(X+z)-H_{\mathrm{price}}(X)]\neq0.
$$

Spanning the factor deltas can therefore leave nonlinear jump risk. The constrained controller revalues financial claims and evaluates its fitted continuation for the operating book in each conditional scenario.

## Execution and risk mandate

For positions held over an interval, futures variation settlement is

$$
\sum_k q_{k,t-1}m_k[F_k(t)-F_k(t-1)].
$$

Execution pays spreads and tiered impact on turnover. Position bounds, turnover caps and impact thresholds are independent inputs. Margin locks cash, with funding consequences for liquidity and borrowing. The curve example uses a synthetic margin rule based on a factor-covariance stress horizon plus a per-contract floor.

\Needspace{6\baselineskip}

For conditional future gains $G$ with mean $\mu$, the default hedge mandate solves a joint operating and financial problem containing

$$
\rho\{L-(G-\mu)q+B a\}+TC(q-q^-)
+\sum_k f_kIM_k(q)+\sum_k(-\mu_kq_k)^+.
$$

Here $a$ contains current dispatch and financing actions. Centering removes the reward for positive expected trading gains, while adverse carry remains a cost. Centered-risk and total-return objectives express alternative mandates; the latter includes signed expected carry. Each claim's settlement terms determine its cash flows. Changes in the mark of a funded claim enter its economic gain, but become cash only through sale or distribution.

\Needspace{5\baselineskip}

The additional financial policy induced by a customer contract is

$$
\Delta q^*(c\mid\mathcal B)=q^*(\mathcal B+c)-q^*(\mathcal B).
$$

This separates contract-driven hedging from any financial allocation the original book would already select.

# Tradable claims, settlement and execution state

## Marks and cash flows

A tradable claim is specified by its ex-distribution mark, cash events, factor sensitivities, maturity, multiplier, permitted positions, execution terms and collateral rule. The model includes futures, European options, fixed-strike transferable forwards, coupon swaps and periodic caps. Every cash-event date must appear on the control grid so that settlement and financing use the same chronology.

For futures, inherited inventory generates variation settlement. For a funded claim, changes in its mark remain unrealized until sale; coupon and exercise amounts alone are distributions. If $v_n$ is its ex-distribution mark and $d_{n+1}$ the next distribution, its economic interval gain is

$$
G_{n+1}=m\{v_{n+1}+d_{n+1}-v_n\}.
$$

Trading from $q^-$ to $q$ exchanges $m v_n(q-q^-)$ in actual cash. Holding inventory receives $m q^-d_{n+1}$, not $m q^-(v_{n+1}-v_n)$. At maturity the ex-distribution mark is zero and the claim's remaining payoff is paid exactly once.

A fixed-strike forward retains its original strike. The periodic cap pays $\sum_n w_n(F_n-K)^+$, applying the strike separately at each coupon date. An option on an arithmetic average pays $(\sum_nw_nF_n-K)^+$, applying it after averaging. The main control study uses the periodic cap; average-payoff claims are valued separately.

## Option prices from the affine transform

For an option expiring at $E$ on a reference delivered at $T\geq E$, define

$$
\mathcal M(z)=E_Q\left[D(t,E)\{F(E,T)\}^{z}\mid X_t=x\right].
$$

The discounted moment transform is affine in $x$ with complex argument $z$ [R19]. Fourier inversion under the discount-weighted and asset-weighted distributions gives a call price

$$
C=\mathcal M(1)P_1(F>K)-K\mathcal M(0)P_0(F>K),
$$

and a put follows by put–call parity. The transform retains dependence between the payoff, stochastic rates and jumps. Prices are checked against probability and price bounds and compared at different quadrature resolutions. The no-jump case has a Gaussian closed form. The transform quadrature requires nonzero projected diffusion; independent Monte Carlo valuation also supports degenerate cases.

## Funding and counterparty events

Paid long options require their premium in cash and have no initial margin under the supplied paid-option convention. Short options require an explicit collateral rule. Gross linear margin is used by the path controller; the finite-control adapter also supports bilateral netting and CSA thresholds through its netting-set schema. Initial margin is a segregated asset and returns when inventory is closed.

Positive unsecured exposure at counterparty default receives its specified recovery, while negative amounts remain payable. Default extinguishes affected claims and blocks reopening with that counterparty. The reference path study has one named supplier counterparty and a survival process correlated with funding stress. This settlement rule leaves clean reference marks unchanged. Any negotiated spread for credit risk enters the executable quote as a separate input.

The control objective separates expected trading carry, premium cash consumption, margin opportunity charges and debt interest. Instrument-specific liquidity-charge rates are annual decimals multiplied by the actual interval. Cash debt interest accrues on actual borrowing.

## Stochastic execution liquidity

Execution liquidity determines market access. Its reference realization combines an idiosyncratic mean-reverting jump process with shared funding and downside factors:

$$
dZ_t=-\kappa_ZZ_tdt+\sigma_ZdW_t^{liq}+dJ_t^{liq},
\qquad \zeta_t=Z_t+\beta_fX_t^{fund}+\beta_d(-X_t^{level})^+.
$$

Spreads and margin multipliers increase exponentially with $\zeta_t$, while depth decreases, subject to specified bounds. Above a specified stress threshold, executable depth is zero. This state allows funding stress to coincide with expensive hedging and reduced market access; its coefficients are synthetic [R09].

Position limits, per-period turnover limits and impact-tier thresholds remain distinct instrument inputs. The same sampled liquidity history is used for every policy on a comparison tape. Financial cash already due is settled before the controller responds to a new liquidity state.

# Identification, estimation and parameter uncertainty

Calibration combines observations of initial curves, physical transitions, option prices and observation noise. It specifies free parameter coordinates, bounds, scale and sign conventions, fixed loadings and optional priors. The free coordinates may include loadings, decay rates, diffusion, physical drift and parameters of either jump law. Jump covariance uses Cholesky parameters to preserve positive semidefiniteness.

For factor observations, $X_{n+1}-e^{-K\Delta}X_n$ has conditional characteristic function

$$
\varphi_\Delta(u)=\exp\left\{
\int_0^\Delta\psi_P(i e^{-Kv}u)dv\right\}.
$$

The estimator matches this function to its empirical counterpart at specified frequencies, together with conditional cross-moments that identify persistence. Market observations enter as normalized repricing errors. These blocks form a bounded weighted least-squares problem [R26]. The weights are chosen for this design rather than to minimize the estimator's asymptotic variance.

For a history of noisy quotes, the estimator maximizes a Gaussian quasi-likelihood. It uses exact first and second transition moments, including jumps, and linearizes the nonlinear observation functions [R21]. The synthetic recovery experiment observes factor states. A full jump likelihood would require a richer density and filtering approximation.

The scaled Jacobian of the data moments and repricing errors assesses local identification before prior penalties are added. Its singular values distinguish information supplied by the observations from uniqueness induced by regularization: a prior penalty can produce a unique optimizer even when the data leave a direction unidentified.

The synthetic cluster bootstrap resamples complete independent paths and perturbs quotes according to their stated measurement errors. The model is refitted for every draw, with failed fits reported. Predictive variance decomposes as

$$
\operatorname{Var}(Y\mid\mathcal D)=
E_\Theta[\operatorname{Var}(Y\mid\Theta,\mathcal D)]
+\operatorname{Var}_\Theta(E[Y\mid\Theta,\mathcal D]).
$$

The first component is process variation; the second is variation across parameter draws. More simulation paths improve estimates conditional on a fitted model, while additional observations are needed to identify uncertain parameters. A single dependent market history requires block resampling or an appropriate parametric observation model. The recovery study fixes nuisance coordinates and evaluates held-out derivative prices under synthetic observations. Its bootstrap measures variability under the specified resampling design; confidence coverage would require a separate statistical argument.

# Numerical methods

The reference economy has nine factors, six compute curves, two electricity curves, one commercial funding curve, an affine interest-rate model, four physical fleet pools and fifteen dated financial instruments: eleven futures, two funded puts, a transferable power forward and a periodic compute cap. Its numerical parameters are synthetic. The experiments retain their case definitions and random seeds.

## Simulation and control

The affine factor transition is sampled from its exact endpoint law. The simulation draws Brownian endpoint and integrated-factor noise jointly and samples compound-Poisson event counts, ages and marks. Smooth cumulant and covariance integrals use fixed Gaussian quadrature. For this constant-coefficient model, the factor endpoints have no Euler time-step bias. Operations and settlement still require a choice of discrete dates.

The path controller estimates continuation values by quadratic regression on factor states from independent training paths. At each decision date, a conditional LP uses these estimates to jointly choose current discretionary dispatch, borrowing, repayment, dated futures and funded claims. These decisions share a cash budget. The continuation values a specified future physical plan, so only current decisions are optimized at each step. This is receding-horizon control [R11, R22].

Incoming variation settlement, accrued interest, scheduled payments and inherited-margin requirements are handled before rebalancing. Mandatory financing repair may sell pledged capacity. New trades, margin and discretionary delivery costs must then fit current cash. Closing receipts arrive after prefunding. This chronology makes financing failure a path event rather than a penalty added at maturity.

## Finite control programs

Each node of the scenario tree represents an observable history. The adapter samples the factor transition at that node and uses the resulting curve values in the finite LP/MILP control problem. It retains the existing book, offered contract, model identity, factor states and date grid. Instrument positions and physical decisions depend only on the history available at each node. The pricing study fixes the horizon and varies branching and random seed.

The compiled model records objective, dual bound, gap, residuals and integrality separately from any secondary turnover tie-break. Finite-support expected shortfall is reported with conditional tail mass and worst-child saturation, which bound the precision available from the conditional support [R10, R13, R14]. Increasing tree size addresses sampling error in the finite program; continuation approximation and factor-transition accuracy require separate checks.

## Reproducibility and interpretation

A complete case specifies the curve model, fleet, debt terms, existing commitments, offered contract and financial instruments. Saved simulations contain factors, integrated rates and physical events; saved policies include the training law, continuation fit, conditional seed and mandate. Hashes and dependency records identify the inputs and code used for each result. Training and evaluation use different seeds, while policy comparisons share evaluation shocks. File formats and software interfaces are described in the [implementation notes](IMPLEMENTATION.md).

Numerical checks cover martingales, initial discount fit, futures–forward convexity, strip partitioning, option parity, analytic Greeks, filter observability, jump hedge residuals, conditional tail support, independent cash reconstruction and replay. Replay compares recorded numerical outputs without refitting. Failed paths and cash shortfalls remain in the evaluations.

Pricing identities establish consistency within the chosen curve family. Recovery experiments assess estimation under synthetic observations, and independent evaluation measures the performance of a fixed policy under the supplied law. Stress experiments measure consequences of specified changes in prices, availability, collateral eligibility, margin or market depth. They assign no occurrence probabilities to those interventions.

The construction links market dynamics to the operator's decisions through

$$
\boxed{\text{coupled curves}\;\longrightarrow\;
\text{claims and factor exposures}\;\longrightarrow\;
\text{financed portfolio value and control}.}
$$

Payment timing, collateral terms and delivery commitments enter the reservation price through their effect on the operator's feasible choices. A quote therefore depends on the business accepting the obligation as well as on market prices.

Full bibliographic records and the use of each source appear in the accompanying [bibliography](REFERENCES.md), also included at the end of the PDF.
