# Input format

## Complete curve case

The root JSON uses schema `compute-market/curve-case` and contains `model`, `fleet`, `instruments`, `existing_book`, `offered` and `assumptions`. An empty existing book must be supplied explicitly. The case is identified by a SHA-256 hash of its canonical serialization.

The executable example is `examples/curve_case.json`. Field validation is implemented by `CurveModel`, `CurveSpec`, `CurveCase`, `Instrument` and the fleet configuration. Array dimensions must agree with the named factors and pools.

## Curve model

`factors` lists the common coordinates. `kappa` contains nonnegative annual decay rates. `diffusion` maps independent Brownian shocks into those coordinates. `physical_drift` is the annual P drift vector; Q drift is normalized to zero. The P and Q specifications each include a Poisson intensity, mark mean and mark covariance. Validation requires equivalent jump supports and a Brownian drift change inside the diffusion span.

Each curve specifies `name`, `kind`, `dates`, `values`, `loading`, `unit` and `observable`. Dates begin at zero and increase. Values of `log` and `funding` curves are positive; `normal` prices can be negative. `observable` distinguishes an observed benchmark from an inferred reference. Initial positive curves interpolate logarithmically; normal prices interpolate linearly. Requests past the calibrated horizon are rejected.

`rate_loading` links the short rate to factors. `rate_dates` and `rate_forwards` specify the initial instantaneous forward-rate curve. `quadrature` selects deterministic integration nodes. The initial discount curve is fitted by the affine shift, using the same factor state.

## Quote observations

A `Quote` records the market, calendar start/end dates, value, standard deviation, source and instrument form. Point futures and terminal forwards have separate forms. For interval observations, the observation operator uses arithmetic delivery weights, with discount weighting for periodically settled forwards.

`normalized_physical` observations require a supplied rights adjustment. Record the adjustment’s source and uncertainty. Executed volume, provider, region, GPU form factor, topology, service guarantee, payment and cancellation rights belong in the upstream normalization record. Public listings, executable quotes and completed transactions are separate observation classes.

The initial fitting routine estimates knot values and reports observation rank and singular values. Decay, covariance and jump-law estimation are separate steps. The operating study estimates physical drift with the other parameters fixed. The structural calibration module can also estimate selected decay, covariance, loading and jump-law parameters; the recovery experiment specifies which nuisance parameters are fixed.

## Units and timing

Time is in years, with explicit delivery intervals. The example uses 8,760 hours per year; real calendars should supply their actual intervals. GPU-hours and MWh measure delivered quantities. Each instrument specifies the cash multiplier that converts its contract count to exposure.

Interest, funding spread and liquidity opportunity charges are annual decimal rates. Electricity intensity is MWh per GPU-hour. Borrowing balances, collateral, premiums, margins, execution costs and cash reserves are currency amounts. Hardware marks are currency per physical block. Rates can be negative where the contract and model permit them.

Financial maturity is a fixed calendar date and must coincide with a settlement date in the path experiment. A positive futures position earns `position * multiplier * price_change`. Compute producer bounds in the example are short-only; power and rate protection are long-only. Position directions are specified in the input fields.

## Instrument execution

Each dated futures instrument has a separate position limit, turnover limit and impact depth. Spread and impact slope are cash costs per contract. Margin uses a specified covariance stress horizon, scale and per-contract floor. Opportunity charges on posted margin are annual. Deployment requires executable spread, depth and margin terms in the registry.

The shared claim registry also provides option premiums, fixed-strike OTC marks, cash events and recovery. The finite backend supports bilateral collateral sets and thresholds; the path study uses gross margin for each instrument.

## Physical portfolio and financing

Specify compatible pool size, GPUs per block, fixed commitments, rates, prepayment fraction, availability, demand, replacement limits, energy intensity and operating costs. Workload-level eligibility and topology are inputs to the dedicated workload/placement operators.

Specify unrestricted cash and reserve; revolving and term debt; advance rates, facility commitment, draw limits, fees, fixed coupon, amortization and maturity; hardware market marks and liquidation terms. Lender marks may lag market appraisal. The synthetic case does not pledge receivables. Actual facility eligibility and exclusive collateral assignments must be preserved.

The hardware's economic life ends at a fixed calendar date. The appraisal integrates over its remaining life, and an independent liquidity basis accounts for resale conditions outside the rental margin. Debt principal remains a liability in terminal wealth.

## Simulation tapes and policy artifacts

`CurveTape` contains `times`, factor paths, integrated short rates and model identity. `EconomyTape` adds appraisal and eligible marks, availability, demand, replacement capacity, customer survival, draw freezes, benchmark/local basis and margin/spread/depth multipliers. Arrays include the root and terminal dates.

Each policy result records positions, unrestricted cash, debt and posted initial margin at every decision date. Margin histories are in currency units; their time integrals are in currency-years. Service shortfall is reported in GPU-hours and as a fraction of total promised GPU-hours. Terminal wealth includes operating receipts, financing costs, claim settlements and resolution cash flows. Reported tail wealth averages the lowest 5% of the wealth distribution, including fractional weight at a boundary observation.

`conditional_innovations.npz` stores the controller's realized factor draws by decision date, scenario and factor. Its metadata records the fitted model, interval length, seed and array checksum. The study, instrument comparison and replay load the same draws.

A policy result saves the complete cash, debt and position paths, terminal wealth, failures, delivery shortfall, peak margin, sales, execution costs and component ledger. Fitted continuations and P-model parameters are saved separately and held fixed during evaluation. Training and holdout seeds differ; paired policies use the same evaluation tape.

Named stresses apply deterministic interventions to saved paths. Their results are conditional on those interventions. Statistical estimation of event rates requires observed histories, including quote vintages, source corrections, censored availability and defaults.

## Claim registry

A futures record contains `name`, `market`, `delivery`, `multiplier`, `direction`, `position_limit`, `trade_limit`, `impact_depth`, `spread`, `impact_slope`, `margin_sigma`, `margin_days`, `minimum_margin`, and `liquidity_charge_annual`. Prices are per quoted unit; the multiplier converts them to currency per position. The position limit bounds inventory, the trade limit bounds turnover at each date, and depth sets the threshold for the second impact tier.

Funded claims additionally specify `kind` (`option`, `forward`, `swap`, `cap`), `strike`, `expiry` when relevant, `put`, `payment_dates`, `weights`, `premium_charge_annual`, `counterparty` and `recovery`. All distribution dates must appear in the execution grid. A cap pays at each specified coupon date. Incoming distributions, purchase/sale cash and remaining ex-distribution value are recorded separately.

## Liquidity and credit

`assumptions.liquidity_spec` contains the named funding factor, common-factor coefficients, idiosyncratic mean reversion/volatility/jump law, spread/depth/margin responses, bounds and outage threshold. Rates and intensities are annual; execution multipliers are dimensionless. `counterparty_hazard_annual` and the documented funding/liquidity covariates generate the supplier survival tape. The tape saves realized counterparty states; policies observe them only as they occur.

## Bucket risk

Each `CurveBucket` identifies market, left support, center, right support and bump size. Compute/funding coordinates are logarithmic; electricity is currency/MWh; rates are annual decimal changes. Bucket calculations record the valuation function's scope, support grid, base model, full/diagonal gamma layout, half-bump error and factor-bridge residual. A 1% log-price move and a one-basis-point rate move are different numerical coordinates.

## Hardware cohorts

`AssetSpec` specifies compute and power references, GPU count, MWh per GPU-hour, other hourly cost, utilization, annual standby costs, startup/shutdown charges, minimum run intervals, terminal residual and a separate resale function. Sale loadings use the same named factor order as the market model. `FleetPolicy` records its training law, source model identity, dates, regression scales and coefficients. Independent evaluation records discounted cash and actions at every date.

Cohort values cover the specified GPU count, lender marks are per physical block, and liquidation proceeds depend on quantity. Each output records its scope.

## Calibration case

`inputs.json` contains a complete model, `parameters`, `anchors`, `quotes`, optional `transition_file`, optional `frequency_grid`, observation-block weights and initial free-parameter values. Every free parameter has a unique name/path, finite bounds and a positive numerical scale. A prior needs both a mean and standard error. Covariance coordinates use `jump_cholesky_p` or `jump_cholesky_q` to keep jump covariance positive semidefinite.

A quote records kind, market, observation time, delivery date or strip, value, standard error, source, optional expiry/strike and factor state when observed. When factor histories are unobserved, the latent estimator accounts for uncertainty about those states. Quote standard errors quantify measurement uncertainty; executable bid–ask spreads are separate inputs.

A physical transition tape must carry measure `P`. The synthetic cluster bootstrap resamples complete independent paths and perturbs quotes at their stated uncertainty. A single market history with dependent observations needs a different resampling design. Calibration outputs retain raw inputs, data/model hashes, free-coordinate estimates, active bounds, data rank, prior declarations, convergence status and uncertainty draws. The package includes synthetic data only.
