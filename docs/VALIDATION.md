# Verification

Run the tests from the repository root after installing the dependencies:

```bash
python -m pytest -q
python scripts/validate_study.py
```

The second command recomputes statistics and cash identities from all 43 saved
policy samples. It checks wealth, tail losses, service shortfalls, margin and
financing costs against the reported results. New check outputs go to `work/validation/`.

For a complete replay of the saved holdout and operating policies:

```bash
python scripts/validate_study.py --replay
```

Replay uses the saved market paths and conditional innovations to rerun the three
holdout policies, the futures-only comparison and the operating-asset policy.
The fitted forecasting law and continuation remain fixed.

## Reproduce the experiments

```bash
python scripts/reproduce.py --quick --output work/smoke
python scripts/reproduce.py --output work/reproduction
```

The quick run uses smaller samples. The full run uses 4,096 training paths,
1,024 holdout paths, 256 conditional draws per decision and 512 paths per stress.
Both write separate outputs; `results/study/` contains the published experiment.
The reproduction script sets one linear-algebra thread per process.

## Published validation records

The records in `results/validation/` describe the initial release, commit `6ff7a47`,
with Python 3.13.5 and the dependencies in `requirements-tested.txt`.
They are preserved alongside the original experiment:

- `pytest.xml` and `coverage.json`: 468 passing tests and 91.05% statement coverage.
- `study_checks.json`: statistics and cash identities for 43 policy samples;
  the largest cash-component reconstruction error was 2.8e-09 currency units.
- `holdout_replay.json`: exact replay of 14 numeric fields for each holdout policy
  and the futures-only comparison.
- `asset_replay.json`: exact replay of actions and cash flows on 8,192 asset paths.
- `distribution_checks.json`: installed-package tests and command-line checks.
- `environment.json` and `source_inventory.json`: the execution environment and
  source hashes for those records.

The commands above check the current checkout and save their outputs separately
from these original records. Numerical intervals in the report measure sampling
uncertainty under the specified synthetic model.
