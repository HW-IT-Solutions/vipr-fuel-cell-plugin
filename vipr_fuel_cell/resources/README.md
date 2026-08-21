# Bundled resources

The bundled sensor profile supports the published PEMFC operating-state
reconstruction example. Its final redistribution terms must be confirmed before
this repository is made public.

- `datasets/operating_profile/profile.yaml` maps stable condition IDs to columns
  in the adjacent `sensor_data.csv` and describes the profile and time axis.

The complete cINN bundle, including its manifest, checkpoint, and matching
scalers, is provisioned separately as described in the repository's top-level
`models/README.md`.

Publication: R. Löser, S. Creutzburg, N. Mothes, and M. Dix, "Conditional
invertible neural network for online-capable monitoring of polymer electrolyte
membrane fuel cells," *International Journal of Hydrogen Energy* 250 (2026)
155929. <https://doi.org/10.1016/j.ijhydene.2026.155929>
