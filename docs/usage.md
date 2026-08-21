# Usage and configuration

## Run the packaged example

After installing VIPR Core and this source checkout in editable mode, run:

```bash
vipr --config \
  @vipr_fuel_cell/examples/configs/pemfc_operating_state_reconstruction.yaml \
  inference run
```

The configuration references the packaged sensor CSV and profile through
package-resource paths. Its relative `model_dir` points from the configuration
file to the complete bundle in `models/test_case_1/`.

Relative paths in a copied configuration are resolved from the directory that
contains that configuration. Packaged files can use VIPR's
`@package/path/to/file` syntax.

## Model bundle

The manifest and all files it describes are kept together:

```text
models/test_case_1/
├── model.yaml
├── checkpoint.ckpt
├── scaler_x.json
└── scaler_y.json
```

The checkpoint is tracked with Git LFS. See [`models/README.md`](../models/README.md)
for checkout and verification details.

The example declares `model_dir: ../../../models/test_case_1`. A relative
`model_dir` is resolved only from the directory containing the active VIPR
configuration. An absolute path can be used for a separately provisioned
bundle.

Model files are not included in the wheel. For a wheel installation, copy the
example configuration and set `model_dir` to the directory containing the
downloaded model bundle.

No automatic Hugging Face download is currently configured. The manifest and
its artifacts can later be hosted and downloaded together without changing the
loader contract.

## Use a custom sensor profile

Copy the packaged example configuration and change only its `load_data` paths:

```yaml
load_data:
  handler: pemfc_dataset
  parameters:
    data_path: ../datasets/my_profile/sensor_data.csv
    profile_path: ../datasets/my_profile/profile.yaml
```

Keep the `INFERENCE_PREPROCESS_PRE_FILTER` block from the example enabled. It
selects the model conditions, restores checkpoint order, validates the values,
and applies the condition scaler.

The accompanying profile maps CSV columns to condition IDs:

```yaml
schema_version: 1
id: my_profile
title: My operating profile
description: Sensor data for a custom operating profile
time: {column: time_s, label: Time, unit: s}
conditions:
  - {id: cell_voltage, column: measured_cell_voltage_V}
  - {id: concentration_loss, column: calculated_concentration_loss_V}
```

For `test_case_1`, the complete profile must map all eleven conditions expected
by the model; the excerpt shows only the first two. Conditions may appear in
any order, and additional conditions are ignored. The [model and data
interface](model-interface.md) explains how these IDs connect the CSV to the
model manifest and checkpoint.

## Use a custom model bundle

Every model directory requires a `model.yaml`. Copy
[`models/test_case_1`](../models/test_case_1), then adapt its condition and
target descriptors, artifact filenames, and SHA-256 hashes:

```yaml
load_model:
  handler: pemfc_cinn
  parameters:
    model_dir: ../models/my_model
    device: cpu
```

All condition and target names in the manifest must match the checkpoint
exactly. Checkpoint and scaler filenames are resolved only within the declared
model directory.

## Generated outputs

The packaged example reconstructs all 300 processed simulation steps and
retains an empirical posterior snapshot at simulation step 41. It writes:

- nine posterior-mean trajectory diagrams and their CSV data;
- one SVG with a 3 x 3 grid of empirical posterior histograms;
- one CSV/TXT table containing histogram bins, densities, counts, and means;
- standalone plotting scripts and the data required to reproduce the figures.

The generated
`scripts/plot_pemfc_posterior_distributions_index_40.py` reproduces the snapshot
SVG without loading the model or requiring VIPR. Snapshot indices are
zero-based on the preprocessed coordinate axis: configured snapshot index 40
corresponds to simulation step 41 because preprocessing drops the invalid
initial row. The histograms visualize the sampled posterior directly; no
Gaussian fit is used.
