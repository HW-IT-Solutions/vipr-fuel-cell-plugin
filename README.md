# VIPR Fuel Cell Plugin

This plugin reconstructs operating parameters of a polymer electrolyte membrane
fuel cell (PEMFC) from measurable or derived quantities. A conditional
invertible neural network (cINN) returns a posterior distribution for each
operating parameter instead of a single point estimate.

The implementation demonstrates that VIPR supports an inverse problem beyond
reflectometry. It reproduces Test Case 1 from:

R. Löser, S. Creutzburg, N. Mothes, and M. Dix, "Conditional invertible neural
network for online-capable monitoring of polymer electrolyte membrane fuel
cells," *International Journal of Hydrogen Energy* 250 (2026) 155929.
<https://doi.org/10.1016/j.ijhydene.2026.155929>

## Quick start

Python 3.10 or newer and [Git LFS](https://git-lfs.com/) are required. Clone
VIPR Core and the plugin into the same parent directory:

```bash
git clone https://codebase.helmholtz.cloud/vipr/vipr-core.git
git clone https://github.com/HW-IT-Solutions/vipr-fuel-cell-plugin.git
```

Install both packages:

```bash
pip install -e './vipr-core'
pip install -e './vipr-fuel-cell-plugin[test]'
```

Run the packaged example:

```bash
vipr --config \
  @vipr_fuel_cell/examples/configs/pemfc_operating_state_reconstruction.yaml \
  inference run
```

The example uses packaged sensor data and metadata. Its verified checkpoint and
scalers are stored in `models/test_case_1/`; see
[`models/README.md`](models/README.md) for provisioning details.
To run the plugin with your own sensor profile or model bundle, follow the
[usage and configuration guide](docs/usage.md).

## Example results

For each of the 300 processed simulation steps, the cINN uses a sensor profile
of eleven measured or derived input quantities and generates 1,000 joint
samples of nine operating parameters. These samples represent the reconstructed
posterior distribution.

At simulation step 41, the figure shows the empirical marginal posterior
distributions of all nine operating parameters. The dashed lines mark their
posterior means.

![Empirical posterior distributions of the nine reconstructed PEMFC operating parameters](docs/images/pemfc_posterior_distributions_index_40.svg)

The posterior mean of the reconstructed anode inlet pressure follows the
simulated operating-profile change:

![Posterior mean of the reconstructed anode inlet pressure over the sensor profile](docs/images/pemfc_anode_inlet_pressure.svg)

The full list of conditioning quantities and reconstructed parameters is
documented in the [model and data interface](docs/model-interface.md).

## Workflow

The plugin follows VIPR's five-step inference workflow:

```mermaid
flowchart LR
    DATA["sensor CSV<br/>and profile"] --> LOAD_DATA["1. Load data"]
    MODEL["model manifest<br/>and artifacts"] --> LOAD_MODEL["2. Load model"]
    LOAD_DATA --> PREPROCESS["3. Preprocess<br/>Select, validate and scale conditions"]
    LOAD_MODEL --> PREPROCESS
    PREPROCESS --> PREDICT["4. Predict<br/>Sample posterior with inverse cINN"]
    LOAD_MODEL --> PREDICT
    PREDICT --> POSTPROCESS["5. Postprocess<br/>Export summaries, plots and data"]
```

The [architecture guide](docs/architecture.md) traces the data and model through
these five steps and explains the VIPR handlers, preprocessing filter, and
result hook used by the plugin.

## Acknowledgements

The PEMFC use case was developed with contributions from Fraunhofer Institute
for Machine Tools and Forming Technology IWU and Helm & Walter IT-Solutions
GmbH (SaxonyAI).

## License

The source code is licensed under the GNU Lesser General Public License v3.0 or
later (`LGPL-3.0-or-later`); see [`LICENSE.txt`](LICENSE.txt). The project
authors are listed in [`AUTHORS.md`](AUTHORS.md).

The trained model and bundled research data may be subject to additional rights
and approvals described in [`NOTICE.md`](NOTICE.md).
