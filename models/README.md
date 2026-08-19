# Local model files

Model weights and scalers are deliberately not stored in this repository while
their redistribution terms are being clarified. Place the approved files in:

```text
models/operating_state_reconstruction/
├── checkpoint.ckpt
├── scaler_x.json
└── scaler_y.json
```

If an approved download location becomes available, the files can be installed
manually without changing the VIPR configuration:

```bash
mkdir -p models/operating_state_reconstruction

MODEL_BASE_URL="https://replace-with-approved-download-location"
wget "$MODEL_BASE_URL/checkpoint.ckpt" \
  -O models/operating_state_reconstruction/checkpoint.ckpt
wget "$MODEL_BASE_URL/scaler_x.json" \
  -O models/operating_state_reconstruction/scaler_x.json
wget "$MODEL_BASE_URL/scaler_y.json" \
  -O models/operating_state_reconstruction/scaler_y.json

sha256sum -c models/checksums.sha256
```

Do not use the placeholder URL. The expected checksums in `checksums.sha256`
identify the exact artifacts used for the published reconstruction example.
