# VIPR Fuel Cell Plugin

This plugin integrates the PEMFC conditional invertible neural network (cINN)
inference used by the HZwo-DigiTwin acceptance scenarios into VIPR.

For each time step, measurable or derived PEMFC performance signals condition
the inverse cINN. The result is a posterior distribution over nine operating
parameters:

- cell current density,
- anode and cathode relative humidity,
- anode and cathode temperature,
- anode and cathode pressure,
- hydrogen and oxygen stoichiometry.

Anomaly detection is deliberately outside the plugin's initial scope.

## Components

- `pemfc_mat`: loads the `out/Data` group and `tout` time axis from MATLAB 7.3
  acceptance files.
- `pemfc_cinn`: directly loads a Lightning-style cINN checkpoint and its two
  JSON min/max scalers. No VIPR registry loader is required.
- `PEMFCConditionPreprocessor`: selects model conditions, removes invalid time
  steps, validates ranges and scales conditions.
- `pemfc_posterior`: samples and summarizes the posterior for every time step.
- `PEMFCDataCollector`: writes parameter tables and time-series diagrams to the
  VIPR result.

## Installation

```bash
pip install -e ./vipr-core
pip install -e './vipr-fuel-cell-plugin[test]'
```

The model checkpoints, scalers and acceptance MAT files are not duplicated in
this repository. Their redistribution license must be clarified first. The
example configurations therefore use paths relative to the local configuration
file and expect the artifacts to be supplied by the user.

## Run

Copy an example config next to, or adjust it to point at, the HZwo-DigiTwin
artifacts and run:

```bash
vipr --config examples/configs/acceptance_1_run_434.yaml inference run
```

`acceptance_1_run_434.yaml` and `acceptance_2_run_434.yaml` use the
eleven-condition model. `acceptance_3_run_436.yaml` uses the reduced
four-condition model.
