# Model files

The operating-state reconstruction model is stored in:

```text
models/operating_state_reconstruction/
├── checkpoint.ckpt
├── scaler_x.json
└── scaler_y.json
```

The checkpoint is managed by Git LFS; the small scaler files are regular Git
objects. Install Git LFS before cloning the repository. If the repository was
already cloned, fetch the checkpoint with:

```bash
git lfs install
git lfs pull
sha256sum -c models/checksums.sha256
```

The expected checksums in `checksums.sha256` identify the exact artifacts used
for the published reconstruction example. The redistribution terms of the model
bundle must be confirmed before the repository is made public.
