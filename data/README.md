# `data/` — local datasets (not versioned)

Everything under `data/` is gitignored except this README. Large datasets live
here so scenario configs can use portable, repo-relative paths instead of
machine-specific absolute paths.

## Layout

```
data/
└── nih_sample/                 # NIH ChestX-ray14 sample (~5.6k images)
    ├── images/                 # the .png chest X-rays
    ├── sample_labels.csv       # original Kaggle labels
    └── sample_labels.xrvclean.csv   # auto-generated (numeric Patient Age) — safe to delete
```

## Get the data

A Kaggle API token is required (`%USERPROFILE%\.kaggle\kaggle.json`, or the
`KAGGLE_API_TOKEN` env var). Then, from the repo root:

```bash
uv run python scripts/download_nih_sample.py
```

It downloads the sample and arranges it into `data/nih_sample/`. The scenarios in
`configs/examples/v1_nih_*.yaml` already point here. Run `hai-sim` from the repo
root so the relative paths resolve.

## Model weights are NOT here

Pre-trained TorchXRayVision weights are managed separately by the library and
auto-download to `~/.torchxrayvision/models_data/` on first use — no action
needed, and they are shared across projects.
