# Implementation

The package uses one market state to price claims, value hardware and make operating decisions. `MarketModel` defines dated curves, discounting, sensitivities and conditional transitions. `TradableClaim` defines a claim's value, cash distributions, margin and execution terms. A `CurveCase` combines these with capacity, financing, existing commitments and the proposed contract.

The pricing law Q values financial claims; the physical law P supplies operating forecasts. An operator's reservation price also depends on its inventory, cash, debt and risk preferences. Commercial funding spreads enter the financing terms but are not tradable. Execution liquidity responds to both shared market shocks and idiosyncratic shocks.

## Curve and claim modules

| Module under `compute_market/curves/` | Responsibility |
|---|---|
| `model.py` | Curve specifications; P/Q cumulants; affine Musiela realization; normal and positive curves; discounting; terminal forwards; joint exact-law endpoint/integrated-factor transitions; serialized tapes |
| `defaults.py` | Fully specified synthetic nine-factor economy and calibrated initial reference curves |
| `claims.py` | Arithmetic strips, forecasting transform, Gaussian/Monte Carlo European claims, finite-rights swing exercise, generic finite-difference Greeks |
| `observations.py` | Point/strip observations, physical normalization, analytic/numerical Jacobians, extended Kalman filter, initial-curve fitting, physical-drift estimation |
| `risk.py` | Dated futures registry, independent position/turnover/impact limits, covariance-based margin, covariance-weighted hedge projection, Euler factor attribution |
| `portfolio.py` | Complete curve case, four fleet pools, curve-based remaining-life appraisal, lender mark lag, causal physical events, named stress transformations |
| `continuation.py` | Quadratic factor-state regression for future physical-plan components; inventory, debt and commitments remain explicit coefficients |
| `control.py` | Incoming settlement and repair; joint current dispatch, borrowing/repayment and financial LP; cash, debt and posted-margin histories; failure and terminal accounting |
| `adapter.py` | Samples the curve transition into a complete finite control case; private inventory-value revaluation |
| `study.py` | Independent experiments, training/holdout protocol, frozen policies, saved tapes, stress pairs, pricing and filtering diagnostics |
| `cli.py` | Complete-case validation, surface construction, private quote, study and replay commands |

`CurveModel` is an alias for `AffineCurveModel`, the implementation with OU–Lévy factors and exponential maturity loadings. The example has nine factors; the model accepts other dimensions. `CurveTape` stores factors and integrated short rates, and `EconomyTape` adds physical events and market frictions. Another model can implement the same market protocol with its own claim-pricing methods; the current Fourier pricer uses affine transforms.

## Numerical and settlement backend

`dynamic_hedging.py`, `optimizer.py` and `hedge_pricing.py` compile finite control programs whose decisions depend on observable histories. `case.py` stores the inputs and their hashes. The curve adapter supplies the histories, prices, rates and equipment marks.

`risk_execution.py` solves the conditional linear program for hedging and operations. It supports any number of instruments, signed position limits, tiered impact, margin and linear operating controls. `simulation.py` handles mandatory financing adjustments and liquidation. When these adjustments succeed, the controller allocates the remaining liquidity from a feasible cash balance. The curve controller supplies prices from the shared curve state.

`hedge_market.py` and `hedge_audit.py` define and check futures, funded options, transferable fixed-strike OTC claims, collateral netting and recovery. The claim registry supplies these instruments to the finite-control backend. The main path experiment uses fifteen instruments with specified margin rules.

`allocation.py` and `workloads.py` enforce cluster placement and workload constraints. The research fleet uses compatible pools; detailed cluster scheduling is a separate physical problem. `risk.py`, `tails.py` and `ambiguity.py` calculate expected shortfall, diagnose scenario support and handle probability ambiguity. `pricing.py` provides point estimates and demand menus for commercial quotes.

## What each numerical method computes

**Clean claims.** Affine transforms price terminal forwards and bonds; Gaussian formulas or deterministic affine-transform inversion price European claims, with independent discounted Monte Carlo checks. Swing exercise is a least-squares stopping policy evaluated on new paths.

**Local risk.** Analytic reference deltas and gammas use shared factor coordinates. Portfolio deltas come from reoptimizing a small finite control problem after bumping each factor, with common random numbers across valuations. The hedge projection uses a local covariance metric; constraints, gamma and jumps are evaluated separately.

**Private reservation.** A finite control program sampled from the curve model values the existing book and the book after acceptance, including their financing. In continuous linear cases, it solves directly for the payment and returns a numerical certificate. Integer cases use bracketed revaluation. Sampling determines this method’s approximation error.

**Path policy.** The adaptive policy jointly chooses current dispatch, debt, futures and funded claims using a fitted continuation value for a specified future operating plan. Mandatory incoming calls and the adjustments needed to meet them precede these choices. The static policy chooses a financial position at the initial date and holds it to expiry; its physical and financing decisions remain adaptive. The unhedged policy uses the same operating and financing controls with no financial positions.

**Operating assets.** A separate switching and retirement policy approximates a Bellman recursion by regression and is evaluated on independent paths. A perfect-information calculation bounds the remaining improvement. Economic operating value, lender appraisal and liquidation proceeds are separate valuations of the same hardware.

## Chronology and interest

At each date, the ledger settles inherited futures gains and funded-claim distributions, applies observed counterparty events, releases previously posted initial margin, pays accrued interest, receives opening invoices and pays scheduled term principal. It tests incoming margin on inherited positions before rebalancing. Mandatory repair may borrow, repay or sell capacity. The state records the resulting changes in capacity and collateral.

The conditional LP then selects new positions, additional financing and discretionary service. Execution costs, signed premium exchanges, new margin and delivery costs must fit cash above the operating reserve. Closing customer receipts and discretionary revenue arrive after that test. Interest accrues on the resulting debt balances for payment at the next date. The terminal date pays that last accrual, settles surviving instruments, sells remaining hardware and repays residual debt.

The ledger records economic cash flows in seven component groups. Margin movements cancel when released. The validation script reconstructs final wealth from these groups and checks it against the cash path.

## Approximation boundaries

The curve realization and its conditional endpoint law are exact for the specified constant-coefficient affine family; deterministic Gaussian quadrature evaluates the integrals. Jumps have constant intensity and Gaussian marks. The filter uses a local Gaussian approximation. The fitted continuation estimates the value of a specified future operating plan; optimizing future decisions would require a Bellman value. Lender appraisal approximates future operating margins.

The path experiment observes simulated factor states. Separate experiments evaluate filtering and calibration from noisy quote histories; the path controller does not use their posteriors.

On failure, the model values contractual replacement losses and liquidates the entity without resolving legal creditor priority. The example has one currency, unrestricted cash and four compatible fleet pools. Restrictions across legal entities, legal recovery and actual execution schedules require additional inputs and rules.

A finite-control grid includes only instruments whose cash events coincide with its dates. The adapter records exclusions in the sampled case. The eight-interval path study includes all four quarterly cap coupons; the two-interval pricing study excludes the cap and includes aligned options and the terminal forward.

The path experiment trades transferable claims at their clean marks and applies specified recovery after default. Executable prices can also include negotiated credit spreads. The path controller uses gross instrument-level margin; the finite-control backend additionally supports bilateral collateral netting.

## Risk, asset and estimation modules

| Module | Economic/mathematical role | Executed checks |
|---|---|---|
| `curves/interfaces.py` | Stable market-law and tradable-claim protocols | Runtime conformance and exported API |
| `curves/buckets.py` | Tent-function delivery shocks; delta/gamma; factor bridge; off-span shapes | Node-supported analytic bridge, half bumps, cross-gamma symmetry |
| `curves/option_pricing.py` | Batched discounted affine transform; Gaussian limit | Independent Monte Carlo, put–call parity, expiry, price bounds and quadrature refinement |
| `curves/instruments.py` | Ex-distribution futures/option/forward/swap/cap registry | Fixed strikes, coupon dates, serialization, explicit margin |
| `curves/settlement.py` | VM, premiums, distributions, maturity and recovery | Separate marks and cash, single distribution at each event, recovery signs and persistent default |
| `curves/registry.py` | Shared claims exported to finite LP/MILP market data | Event-grid alignment, identical clean marks and cash events |
| `curves/liquidity.py` | Shared and idiosyncratic execution risk | Seed replay, bounded multipliers, wrong-way direction, missing-factor errors |
| `curves/fleet_options.py` | Operating switching, minimum runs, retirement, liquidation | Policy feasibility, independent cash replay, pathwise information bound |
| `curves/calibration.py` | Declared-coordinate estimation, latent history, rank, bootstrap | Recovery, rank excluding priors, PSD covariance, serialization, total variance |
| `curves/system_study.py` | Reproducible risk, assets, calibration and hedge-universe experiments | Retained inputs, trained models, independent tapes and reports |

## Reproducibility

`results/study/protocol.json` gives the sample sizes, dates, seeds and controller settings. `case.json` includes the economic book. `policy_model.json` stores the estimated P law, `continuation.npz` the continuation fit, and `conditional_innovations.npz` the controller's simulated draws. Holdout and stress tapes store every simulated factor, integrated rate and physical event. Policy files record wealth, failure, shortfall, positions, posted margin, cash, debt and component cash flows.

`results/study/pricing/` contains the sampled cases and quote certificates. `results/validation/` contains records from the original tests and replays. These describe the source and environment used for those runs. New checks and experiments can write to the ignored `work/` directory:

```bash
python scripts/validate_study.py --output work/validation
python scripts/validate_study.py --replay --output work/replay
python scripts/reproduce.py --output work/reproduction
python scripts/validate_study.py --study work/reproduction --output work/reproduction-checks
```

The first command checks accounting identities and statistics against the saved outcomes. The optional replay re-executes all three holdout policies, the futures-only comparison and the asset policy using their saved paths and draws. A full reproduction generates a new set of experiments; `--quick` reduces the sample sizes. The reproduction script sets one linear-algebra thread per process before loading NumPy.

To match the original environment, use Python 3.13.5 and install the pinned versions in a fresh virtual environment:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
```

## Reproduce individual layers

```bash
python -m compute_market curves analyze --output work/analysis
python -m compute_market curves calibrate \
  results/study/calibration/inputs.json --output work/refit.json
```

`curves analyze` runs the bucket-risk, operating-policy and synthetic-calibration experiments. `curves calibrate` fits the supplied data; history paths resolve relative to the input JSON. A full reproduction includes these experiments and the instrument comparison. To repeat just that comparison on a completed reproduction, use `python scripts/instrument_ablation.py work/reproduction`.

The top-level `research` command runs the separate finite-kernel control study. The `curves study` command, also called by `scripts/reproduce.py`, runs the curve-based study reported here. Their output formats differ, so each study needs its own output directory.

## Documents and package builds

The report renderer reads saved study results and writes the report and figures. These commands require a Git checkout, which includes the saved study. The source distribution contains the library, documentation, examples and tests; the wheel contains the library. Figures use Linux Libertine, the same font as the PDFs. PDF generation also requires Pandoc, XeLaTeX and Latin Modern. With the pinned environment installed, the Python distribution uses the standard build frontend:

```bash
python scripts/render_references.py
python scripts/render_research_report.py
bash scripts/build_pdf.sh
python -m build --no-isolation --outdir ../compute-market-packages
```
