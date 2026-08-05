"""The general label-source seam: cross-schema label maps, CSV label sources, and the
image-folder dataset that ties several label CSVs to one image set (selectable by
``gt_source``)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image as PILImage

from hai_sim.core.types import LabelSchema
from hai_sim.data.folder import ImageFolderDataset
from hai_sim.data.sources import CsvLabelSource
from hai_sim.labels import build_label_map, identity_label_map, resolve_label_map
from hai_sim.sim.dataset_generator import DatasetPatientGenerator

XRV = LabelSchema.xrv()
VINDR = LabelSchema.vindr()
XIDX = {name: i for i, name in enumerate(XRV.labels)}


def test_label_map_rename_split_drop_missing() -> None:
    m = resolve_label_map("vindr14->xrv", VINDR, XRV)
    src = [0] * VINDR.n
    src[VINDR.labels.index("Pleural effusion")] = 1
    src[VINDR.labels.index("Nodule/Mass")] = 1
    src[VINDR.labels.index("Aortic enlargement")] = 1  # VinDr-only -> dropped
    out = m.map_gt(src)
    assert out[XIDX["Effusion"]] == 1  # rename
    assert out[XIDX["Nodule"]] == 1 and out[XIDX["Mass"]] == 1  # split
    assert out[XIDX["Edema"]] == -1  # no VinDr source -> not annotated
    assert out[XIDX["Cardiomegaly"]] == 0  # mapped, absent


def test_label_map_bad_labels_fail() -> None:
    with pytest.raises(ValueError):
        build_label_map({"NotAThing": "Effusion"}, VINDR, XRV)
    with pytest.raises(ValueError):
        build_label_map({"Cardiomegaly": "NotATarget"}, VINDR, XRV)


def test_resolve_unknown_named_map_fails() -> None:
    with pytest.raises(ValueError):
        resolve_label_map("nope->xrv", VINDR, XRV)


def test_identity_map_requires_subset() -> None:
    with pytest.raises(ValueError):
        identity_label_map(VINDR, XRV)  # VinDr labels are not in the xrv schema
    same = identity_label_map(XRV, XRV)
    v = [1] + [0] * (XRV.n - 1)
    assert same.map_gt(v) == tuple(v)


def test_inline_map() -> None:
    m = resolve_label_map({"Cardiomegaly": "Cardiomegaly"}, VINDR, XRV)
    src = [0] * VINDR.n
    src[VINDR.labels.index("Cardiomegaly")] = 1
    assert m.map_gt(src)[XIDX["Cardiomegaly"]] == 1
    assert m.map_gt(src)[XIDX["Effusion"]] == -1  # not in the inline map


def _write_vindr_csv(path, rows) -> None:
    cols = ["image_id", "Cardiomegaly", "Pleural effusion", "Nodule/Mass"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_csv_label_source_maps_and_misses(tmp_path) -> None:
    p = tmp_path / "labels.csv"
    _write_vindr_csv(
        p, [{"image_id": "a", "Cardiomegaly": 1, "Pleural effusion": 0, "Nodule/Mass": 1}]
    )
    src = CsvLabelSource(
        p, resolve_label_map("vindr14->xrv", VINDR, XRV), id_column="image_id", name="vindr"
    )
    got = src.labels_for("a")
    assert got is not None
    assert got[XIDX["Cardiomegaly"]] == 1
    assert got[XIDX["Effusion"]] == 0
    assert got[XIDX["Nodule"]] == 1 and got[XIDX["Mass"]] == 1
    assert got[XIDX["Edema"]] == -1  # no column, no source -> not annotated
    assert src.labels_for("missing") is None


def test_csv_source_no_matching_columns_fails(tmp_path) -> None:
    p = tmp_path / "labels.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "Pleural effusion", "Nodule/Mass"])
        w.writeheader()
        w.writerow({"image_id": "a", "Pleural effusion": 1, "Nodule/Mass": 0})
    # Declared as already-xrv, but none of these columns is an xrv label -> fail fast.
    with pytest.raises(ValueError):
        CsvLabelSource(p, identity_label_map(XRV, XRV), id_column="image_id")


def _make_png(path) -> None:
    PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)


def _folder_with_two_sources(tmp_path):
    imgdir = tmp_path / "img"
    imgdir.mkdir()
    _make_png(imgdir / "a.png")
    _make_png(imgdir / "b.png")
    maj = tmp_path / "maj.csv"
    _write_vindr_csv(
        maj,
        [
            {"image_id": "a", "Cardiomegaly": 1, "Pleural effusion": 0, "Nodule/Mass": 0},
            {"image_id": "b", "Cardiomegaly": 0, "Pleural effusion": 1, "Nodule/Mass": 0},
        ],
    )
    unan = tmp_path / "unan.csv"
    _write_vindr_csv(
        unan,
        [
            {"image_id": "a", "Cardiomegaly": 0, "Pleural effusion": 0, "Nodule/Mass": 0},
            {"image_id": "b", "Cardiomegaly": 0, "Pleural effusion": 1, "Nodule/Mass": 0},
        ],
    )
    m = resolve_label_map("vindr14->xrv", VINDR, XRV)
    s_maj = CsvLabelSource(maj, m, id_column="image_id", name="majority")
    s_unan = CsvLabelSource(unan, m, id_column="image_id", name="unanimous")
    return ImageFolderDataset(imgdir, XRV, [s_maj, s_unan], preprocess=None)


def test_image_folder_dataset_multiple_sources(tmp_path) -> None:
    ds = _folder_with_two_sources(tmp_path)
    assert ds.capabilities.gt_sources == ("native", "majority", "unanimous")
    assert len(ds) == 2
    ex_a = ds[0]  # sorted -> a.png first
    assert ex_a.example_id == "a"
    assert ex_a.image.shape == (8, 8)
    # native == primary == the first source (majority)
    assert ex_a.ground_truth[XIDX["Cardiomegaly"]] == 1
    assert ex_a.label_sources["unanimous"][XIDX["Cardiomegaly"]] == 0


def test_generator_selects_named_gt_source(tmp_path) -> None:
    ds = _folder_with_two_sources(tmp_path)
    gen = DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=1, gt_source="unanimous")
    _, patient = next(iter(gen))
    assert patient.ground_truth[XIDX["Cardiomegaly"]] == 0  # unanimous says absent for 'a'

    gen2 = DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=1, gt_source="majority")
    _, patient2 = next(iter(gen2))
    assert patient2.ground_truth[XIDX["Cardiomegaly"]] == 1  # majority says present

    with pytest.raises(ValueError):
        DatasetPatientGenerator(ds, arrival_rate=1.0, n_patients=1, gt_source="nope")
