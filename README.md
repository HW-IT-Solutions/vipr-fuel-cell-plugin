# VIPR Fuel Cell Plugin

This plugin demonstrates uncertainty-aware reconstruction of PEMFC operating
parameters from measurable sensor signals using a conditional invertible neural
network (cINN). It implements the inverse mapping described in:

R. Löser, S. Creutzburg, N. Mothes, and M. Dix, "Conditional invertible neural
network for online-capable monitoring of polymer electrolyte membrane fuel
cells," *International Journal of Hydrogen Energy* 250 (2026) 155929.
<https://doi.org/10.1016/j.ijhydene.2026.155929>

For every point in the sensor profile, the plugin reconstructs a posterior over
nine operating parameters: current density, anode and cathode humidity,
temperature and pressure, and hydrogen and oxygen stoichiometry. Anomaly
detection is deliberately outside the initial scope.

## Components

- `pemfc_dataset` loads a curated sensor profile and its English metadata.
- `pemfc_cinn` directly loads a locally provisioned cINN and its min/max
  scalers. No VIPR registry loader is used.
- `PEMFCConditionPreprocessor` selects, validates and scales model conditions.
- `pemfc_posterior` samples and summarizes the conditional posterior.
- `PEMFCDataCollector` exports parameter summaries and time-series diagrams.

Both loaders also accept custom files through `data_path`, `metadata_path`, and
`checkpoint_path`. Relative custom paths are resolved next to the VIPR config.

## Installation

```bash
pip install -e './vipr-core'
pip install -e './vipr-fuel-cell-plugin[test]'
```

The paper's Test Case 1 model bundle is stored in `models/test_case_1/`. The
checkpoint is versioned with Git LFS, while its metadata and small scaler files
are regular Git objects. Install Git LFS before cloning or run `git lfs pull`
afterwards; see [`models/README.md`](models/README.md).

## Run

The example config contains only workflow choices and logical resource names;
it has no machine-specific paths, sensor arrays, run IDs, or registry entries.

```bash
vipr --config \
  @vipr_fuel_cell/inversion/examples/configs/pemfc_operating_state_reconstruction.yaml \
  inference run
```

The built-in dataset is selected as `operating_profile`, and the direct model
loader selects the paper's eleven-condition cINN as `test_case_1`.

## License

The source code is licensed under the GNU Lesser General Public License v3.0 or
later (`LGPL-3.0-or-later`); see [`LICENSE.txt`](LICENSE.txt). The project
authors are listed in [`AUTHORS.md`](AUTHORS.md).

The trained model and bundled research data may be subject to additional rights
and approvals described in [`NOTICE.md`](NOTICE.md).
