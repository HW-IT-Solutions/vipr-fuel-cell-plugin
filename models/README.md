# Model files

The model bundle corresponding to Test Case 1 in the published PEMFC study is
stored in:

```text
models/test_case_1/
├── checkpoint.ckpt
├── metadata.yaml
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

The expected checksums in `checksums.sha256` identify the exact checkpoint and
scalers used for the published reconstruction example. `metadata.yaml` maps the
bundle to the paper's eleven-condition Test Case 1. The redistribution terms of
the model bundle must be confirmed before the repository is made public.
