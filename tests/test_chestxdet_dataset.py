"""ChestXDetDataset parsing, label mapping and mask co-registration.

Uses a tiny synthetic fixture (one 40x40 image + a hand-written annotation) so no
212 MB download is needed. Skipped when the [models] extra (torchxrayvision /
scikit-image) is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torchxrayvision")
pytest.importorskip("skimage")

from PIL import Image as PILImage

from hai_sim.analysis.localization.chestxdet import ChestXDetDataset
from hai_sim.core.types import LabelSchema

XRV = LabelSchema.xrv()
I_EFFUSION = XRV.labels.index("Effusion")  # 7  -> mapped, present in fixture
I_CARDIO = XRV.labels.index("Cardiomegaly")  # 10 -> mapped, absent in fixture
I_INFILTRATION = XRV.labels.index("Infiltration")  # 2 -> not covered by ChestX-Det


@pytest.fixture
def fixture_dir(tmp_path):
    # 40x40 grayscale gradient (avoids a low-contrast image), one PNG.
    grad = np.tile(np.arange(40, dtype=np.uint8).reshape(1, 40) * 6, (40, 1))
    PILImage.fromarray(grad, mode="L").save(tmp_path / "0001.png")

    # Effusion polygon over the TOP-LEFT quadrant (x,y in [0,20]); Calcification is
    # present but unmapped, so it must not leak into any xrv label.
    record = {
        "file_name": "0001.png",
        "syms": ["Effusion", "Calcification"],
        "boxes": [[0, 0, 20, 20], [30, 30, 39, 39]],
        "polygons": [[[0, 0], [20, 0], [20, 20], [0, 20]], [[30, 30], [39, 30], [39, 39]]],
    }
    (tmp_path / "ann.json").write_text(json.dumps([record]), encoding="utf-8")
    return tmp_path


def test_len_and_shapes(fixture_dir):
    ds = ChestXDetDataset(fixture_dir / "ann.json", fixture_dir, resize=8)
    assert len(ds) == 1
    ex = ds[0]
    assert ex.image_id == "0001.png"
    assert ex.image.shape == (1, 8, 8)
    assert len(ex.labels) == XRV.n == len(ex.masks)


def test_label_mapping(fixture_dir):
    ex = ChestXDetDataset(fixture_dir / "ann.json", fixture_dir, resize=8)[0]
    # present mapped category -> 1
    assert ex.labels[I_EFFUSION] == 1
    # mapped category the radiologist did NOT annotate -> expert negative (0)
    assert ex.labels[I_CARDIO] == 0
    # xrv pathology ChestX-Det never annotates -> not-annotated sentinel (-1)
    assert ex.labels[I_INFILTRATION] == -1


def test_masks_only_for_present_mapped_labels(fixture_dir):
    ex = ChestXDetDataset(fixture_dir / "ann.json", fixture_dir, resize=8)[0]
    assert ex.masks[I_EFFUSION] is not None  # present -> mask
    assert ex.masks[I_CARDIO] is None  # absent -> no mask
    assert ex.masks[I_INFILTRATION] is None  # not covered -> no mask


def test_mask_is_coregistered_top_left(fixture_dir):
    ex = ChestXDetDataset(fixture_dir / "ann.json", fixture_dir, resize=8)[0]
    mask = ex.masks[I_EFFUSION]
    assert mask is not None
    assert mask.shape == (8, 8)
    assert mask.dtype == bool
    # polygon covered the top-left quadrant of the 40x40 image -> after resize to 8
    # the True region is the top-left corner, not the bottom-right.
    assert bool(mask[0, 0]) is True
    assert bool(mask[7, 7]) is False
    assert mask.sum() > 0


def test_calcification_does_not_leak_into_nodule(fixture_dir):
    # Calcification is unmapped; the only present *mapped* label is Effusion.
    ex = ChestXDetDataset(fixture_dir / "ann.json", fixture_dir, resize=8)[0]
    positive = [i for i, v in enumerate(ex.labels) if v == 1]
    assert positive == [I_EFFUSION]
