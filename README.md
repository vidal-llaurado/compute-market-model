# Pricing Compute Obligations

A model for pricing compute contracts and managing a financed hardware fleet, built on the premise that compute is not a commodity and contracts cannot be priced according to market rates alone. 

Instead, an operator's reservation price depends on its existing commitments, available capacity and (importantly) financing. Shared stochastic factors describe compute prices, electricity costs, interest rates and funding conditions. The model calculates the payment that preserves the optimized value of the existing book after taking on a new obligation.

The purpose of the project is less about developing a production-ready pricing engine and more about providing a conceptual bench in the emerging compute economics. 

- [Model](docs/MODEL.md) ([PDF](docs/MODEL.pdf)): equations, assumptions and modeling choices.
- [Research report](docs/RESEARCH_REPORT.md) ([PDF](docs/RESEARCH_REPORT.pdf)): experiments and their results.
- [Implementation](docs/IMPLEMENTATION.md), [input format](docs/DATA_CONTRACT.md) and [worked examples](examples/README.md).

## Run an example

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
python -m pytest -q
python -m compute_market curves quote examples/curve_case.json \
  --branches 8 --periods 2 --seed 611 --output work/quote.json
```

The quote includes the reservation-price interval, incremental hedge and solver certificate. The sampled case is saved alongside it. The original research environment is pinned in [requirements-tested.txt](requirements-tested.txt); [reproduction instructions](docs/IMPLEMENTATION.md#reproducibility) cover that environment and the full study.

## Check the results

```bash
python scripts/validate_study.py --output work/validation
python scripts/reproduce.py --quick --output work/smoke
```

The first command checks saved accounting and summary statistics; the second runs a smaller version of the experiments. [Saved study data](results/study/) support the report, and [validation records](docs/VALIDATION.md) describe the checks performed for the published results.
