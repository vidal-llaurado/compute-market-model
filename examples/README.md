# Worked cases

All examples are synthetic. `curve_case.json` describes a nine-factor market and financed fleet, including the existing book, offered contract, initial curves, P/Q laws and fifteen financial instruments. `initial_surface.json` contains the corresponding initial curves, discount factors and terminal forwards.

From the repository root, the installed package can validate the case and reconstruct its surface:

```bash
python -m compute_market curves validate examples/curve_case.json
python -m compute_market curves surface examples/curve_case.json \
  --output work/surface.json
```

`dynamic_portfolio.py` reads this case, samples a small finite control problem from the curve model, prices the offered contract and prints the incremental hedge. The sampled case is saved with the result.

The underlying curves can also be used directly:

```python
import numpy as np
from compute_market.curves.defaults import example_model

model = example_model()
x = np.zeros(model.dimension)
future = model.value("G1", 0.0, x, 1.0)
forward = model.forward("G1", 0.0, x, 1.0)
bond = model.discount(0.0, x, 1.0)
delta, gamma = model.greeks("G1", 0.0, x, 1.0)
paths = model.simulate([0.0, 0.25, 0.5, 1.0], 4096, seed=71, measure="P")
```

Compute reference prices are positive; electricity prices can be negative. P supplies operating forecasts and Q supplies clean financial prices. Stochastic rates produce a convexity adjustment between futures and terminal forwards. Commercial funding spreads enter the financing terms but are not tradable.

`settlement_case.json` tests the general optimizer's accounting for futures, funded claims, OTC collateral and execution tiers. The scripts in `primitives/` demonstrate physical allocation, payoffs and stress tests.

The [saved study](../results/study/) contains the inputs and outcomes used in the research report. [Implementation notes](../docs/IMPLEMENTATION.md#reproducibility) describe how to check or reproduce them.
