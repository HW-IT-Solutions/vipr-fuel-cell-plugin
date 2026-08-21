# Model and data interface

The Test Case 1 cINN reconstructs posterior distributions for nine PEMFC
operating parameters from eleven measurable or derived conditioning
quantities.

## Conditioning quantities

| # | Conditioning quantity | Unit |
| ---: | --- | --- |
| 1 | Cell voltage | V |
| 2 | Concentration loss | V |
| 3 | Ohmic loss | V |
| 4 | Hydrogen crossover loss | V |
| 5 | Activation loss | V |
| 6 | Anode limiting current density | A/cm² |
| 7 | Cathode limiting current density | A/cm² |
| 8 | Anode water flux | mol/(m² s) |
| 9 | Cathode water flux | mol/(m² s) |
| 10 | Anode hydrogen flux | mol/(m² s) |
| 11 | Cathode oxygen flux | mol/(m² s) |

## Reconstructed operating parameters

| # | Reconstructed operating parameter | Unit |
| ---: | --- | --- |
| 1 | Cell current density | A/cm² |
| 2 | Anode inlet relative humidity | 1 (dimensionless) |
| 3 | Cathode inlet relative humidity | 1 (dimensionless) |
| 4 | Anode inlet temperature | K |
| 5 | Cathode inlet temperature | K |
| 6 | Anode inlet pressure | atm |
| 7 | Cathode inlet pressure | atm |
| 8 | Hydrogen stoichiometry | 1 (dimensionless) |
| 9 | Oxygen stoichiometry | 1 (dimensionless) |

## Mapping data to the model

The data profile and model manifest form the stable interface between a sensor
CSV and the checkpoint:

```text
sensor_data.csv            profile.yaml       model.yaml       checkpoint
CSV column              -> condition ID    -> tensor name   -> model input
measured_cell_voltage_V -> cell_voltage    -> U_cell_V
```

Each layer owns different metadata:

- `profile.yaml` describes the profile provenance and time axis and maps its
  CSV columns to stable condition IDs.
- `model.yaml` assigns stable IDs, labels, and units to model conditions and
  reconstructed targets.
- The checkpoint defines the tensor names, dimensions, and order expected by
  the trained network.
- The condition and parameter scalers contain the numerical ranges used before
  and after the cINN mapping.

The preprocessor joins `profile.yaml` and `model.yaml` through the condition
IDs, selects the conditions required by the model, and restores checkpoint
order. Profile conditions may appear in any order. Additional conditions are
allowed and ignored.

For `test_case_1`, all eleven condition IDs listed above must be mapped. A
missing ID is reported together with its checkpoint name. For reconstructed
targets, the stable ID is used in exported tables and diagram identifiers; it
does not join the profile to the model.

The packaged files are:

- [`profile.yaml`](../vipr_fuel_cell/resources/datasets/operating_profile/profile.yaml)
- [`model.yaml`](../vipr_fuel_cell/resources/models/test_case_1/model.yaml)

See [usage and configuration](usage.md) to connect custom profiles or model
bundles.
