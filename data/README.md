# Data

The studies of the paper replay the outputs of the three AI models, stored once per model
and dataset in `cache/`. They never open an image, so no image is needed to reproduce them.

```
data/
├── cache/                          shipped: what the models said about every image
│   ├── <dataset>__<model>.npz/.json  one model on one dataset (a "cell")
│   ├── composed/<dataset>__ens4.*    the four DenseNet-121 members in one cell
│   └── calibration/<dataset>__*.json Platt scaling per finding, fitted on the even rows
├── provenance/                     shipped: image lists and checksums of the two datasets
├── vindr/labels.csv                built locally by scripts/build_vindr_labels.py
├── nih_full/                       optional, scripts/fetch_nih_images.py (about 42 GB)
└── vindr/images/                   optional, the VinDr-CXR 1024 px PNGs (about 7 GB)
```

## The cells

| model | cells | images |
|---|---|---|
| DenseNet-121 ensemble (TorchXRayVision weights `chex`, `mimic_ch`, `pc`, `rsna`) | `nih__<member>`, `vindr__<member>`, `composed/*__ens4` | 30,805 NIH ChestX-ray14, 15,000 VinDr-CXR |
| MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`, BF16) | `nih__medgemma-1.5-4b`, `vindr__medgemma-1.5-4b` | same |
| MedGemma 27B (`google/medgemma-27b-it`, FP8) | `nih__medgemma-27b`, `vindr__medgemma-27b` | same |

Each `.npz` holds the raw probability of every member for every image and finding
(`member_probs`, shape images × members × 18 findings), the image ids, and the inference
time per image. The `.json` next to it is a readable manifest: the model and dataset
fingerprints, the finding names and the timing. The configurations that built the cells
are in `configs/cache/`.

## Labels

**NIH ChestX-ray14.** The labels are inside the NIH cells. The NIH Clinical Center places
no restriction on the use of the dataset and asks to be acknowledged as its provider and
to cite Wang et al., CVPR 2017
([doi:10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369)).

**VinDr-CXR.** The VinDr-CXR labels are distributed under their own terms, so the
VinDr-CXR cells ship without them: their manifest names `data/vindr/labels.csv` instead,
and the cell reads its ground truth from that file. Build it once:

```bash
uv run python scripts/build_vindr_labels.py
```

The script downloads `train.csv` (about 2 MB, the annotations of the three radiologists
per image) from the Kaggle competition
[`vinbigdata-chest-xray-abnormalities-detection`](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection).
It needs a Kaggle API token (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) and the
competition rules accepted once on that page. A finding is present when at least two of
the three radiologists marked it. Cite Nguyen et al., Scientific Data 2022
([doi:10.1038/s41597-022-01498-w](https://doi.org/10.1038/s41597-022-01498-w)).

## Images (optional)

Only needed to build new cells or to run a scenario that reads the pixels.

| dataset | source | how | check |
|---|---|---|---|
| NIH ChestX-ray14 | Kaggle dataset `nih-chest-xrays/data`, version 3 | `uv run --with kagglehub python scripts/fetch_nih_images.py` | done by the script against `provenance/nih_sha256sums.txt` |
| VinDr-CXR | Kaggle dataset `awsaf49/vinbigdata-1024-image-dataset`, version 1 (the training images as 1024 px PNG) | `kagglehub.dataset_download("awsaf49/vinbigdata-1024-image-dataset")`, then point `data/vindr/images` to its `vinbigdata/train` folder | from the download's `vinbigdata/` folder: `sha256sum -c <repo>/data/provenance/vindr_sha256sums.txt` |
