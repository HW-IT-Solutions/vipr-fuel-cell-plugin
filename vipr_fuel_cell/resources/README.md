# Bundled resources

The bundled sensor profile supports the published PEMFC operating-state
reconstruction example. Its final redistribution terms must be confirmed before
this repository is made public.

- `datasets/operating_profile/profile.yaml` maps stable condition IDs to columns
  in the adjacent `sensor_data.csv` and describes the profile and time axis.
- `models/test_case_1/model.yaml` defines the cINN conditions and targets as
  well as the filenames and SHA-256 hashes of its external artifacts.

The checkpoint and matching scalers are provisioned separately as described in
the repository's top-level `models/README.md`.

Publication: R. Löser, S. Creutzburg, N. Mothes, and M. Dix, "Conditional
invertible neural network for online-capable monitoring of polymer electrolyte
membrane fuel cells," *International Journal of Hydrogen Energy* 250 (2026)
155929. <https://doi.org/10.1016/j.ijhydene.2026.155929>
