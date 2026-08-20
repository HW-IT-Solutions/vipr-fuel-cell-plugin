# Model files

The model bundle corresponding to Test Case 1 in the published PEMFC study is
stored in:

```text
models/test_case_1/
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
```

The plugin verifies all three files against the SHA-256 values in the packaged
[`model.yaml`](../vipr_fuel_cell/resources/models/test_case_1/model.yaml) before
loading them. That manifest also defines stable IDs, labels, and units for all
conditions and reconstructed targets; tensor names, order, and dimensions come
from the checkpoint itself. The redistribution terms of the model bundle must
be confirmed before the repository is made public.
