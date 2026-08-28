# Model files

The model bundle corresponding to Test Case 1 in the published PEMFC study is
stored in:

```text
models/test_case_1/
├── model.yaml
├── checkpoint.ckpt
├── scaler_x.json
└── scaler_y.json
```

The versioned release archive contains the complete checkpoint. No Git LFS
installation is required when using that archive.

In a Git checkout, the checkpoint is managed by Git LFS while the small scaler
files are regular Git objects. Install Git LFS before cloning the repository.
If the repository was already cloned, fetch the checkpoint with:

```bash
git lfs install
git lfs pull
```

The plugin reads `model.yaml` from this directory and verifies the other three
files against its SHA-256 values before loading them. The manifest also defines
stable IDs, labels, and units for all conditions and reconstructed targets;
tensor names, order, and dimensions come from the checkpoint itself. The
redistribution terms of the model bundle must be confirmed before the
repository is made public.
