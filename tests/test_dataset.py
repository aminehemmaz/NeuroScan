"""The splits must never put a patient on both sides. These tests enforce it."""

import csv

import numpy as np
import pytest

from neuroscan.dataset import (check_no_patient_overlap, class_distribution,
                               cross_validation_splits, leaky_split, load_index,
                               patient_folds, to_frame)


def write_dataset(tmp_path, rows):
    """Build a small dataset on disk: one blank PNG per row plus the index."""
    import cv2

    images = tmp_path / "images"
    images.mkdir()
    for row in rows:
        name = f"{row['slice_id']:04d}.png"
        cv2.imwrite(str(images / name), np.full((32, 32), row["label"] * 40, np.uint8))
        row["file"] = f"images/{name}"
    with (tmp_path / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def sample_rows():
    rows = []
    slice_id = 1
    for label in (1, 2, 3):
        for patient in range(5):
            name = f"P{label}{patient}"
            for _ in range(3):
                rows.append({"slice_id": slice_id, "patient_id": name, "label": label,
                             "class": "x", "fold": patient % 5 + 1, "tumour_pixels": 10})
                slice_id += 1
    return rows


@pytest.fixture
def dataset(tmp_path):
    return load_index(write_dataset(tmp_path, sample_rows()))


def test_index_is_read(dataset):
    assert len(dataset) == 45
    assert {item.label for item in dataset} == {1, 2, 3}
    assert all(item.path.exists() for item in dataset)


def test_missing_image_is_reported(tmp_path):
    rows = sample_rows()
    root = write_dataset(tmp_path, rows)
    (root / rows[0]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        load_index(root)


def test_missing_column_is_reported(tmp_path):
    rows = sample_rows()
    for row in rows:
        del row["patient_id"]
    write_dataset(tmp_path, rows)
    with pytest.raises(ValueError, match="patient_id"):
        load_index(tmp_path)


def test_every_slice_is_tested_exactly_once(dataset):
    tested = []
    for _, _, test in cross_validation_splits(dataset):
        tested.extend(item.slice_id for item in test)
    assert sorted(tested) == sorted(item.slice_id for item in dataset)


def test_cross_validation_never_shares_a_patient(dataset):
    for fold, train, test in cross_validation_splits(dataset):
        train_patients = {item.patient_id for item in train}
        test_patients = {item.patient_id for item in test}
        assert not (train_patients & test_patients), f"fold {fold} shares a patient"


def test_train_and_test_cover_the_dataset(dataset):
    for _, train, test in cross_validation_splits(dataset):
        assert len(train) + len(test) == len(dataset)


def test_overlap_check_raises():
    class Fake:
        def __init__(self, pid):
            self.patient_id = pid

    with pytest.raises(AssertionError, match="both"):
        check_no_patient_overlap([Fake("A"), Fake("B")], [Fake("B")])


def test_overlap_check_accepts_disjoint_patients():
    class Fake:
        def __init__(self, pid):
            self.patient_id = pid

    check_no_patient_overlap([Fake("A")], [Fake("B")])


def test_leaky_split_does_share_patients(dataset):
    """The deliberately wrong split must really be wrong, or the demo is pointless."""
    train, test = leaky_split(dataset, test_size=0.3, seed=0)
    shared = {i.patient_id for i in train} & {i.patient_id for i in test}
    assert shared, "with 3 slices per patient a random split should share patients"


def test_folds_are_grouped(dataset):
    folds = patient_folds(dataset)
    assert set(folds) == {1, 2, 3, 4, 5}
    for members in folds.values():
        assert members


def test_class_distribution_counts_patients(dataset):
    table = class_distribution(dataset)
    assert list(table["slices"]) == [15, 15, 15]
    assert list(table["patients"]) == [5, 5, 5]


def test_frame_has_one_row_per_slice(dataset):
    frame = to_frame(dataset)
    assert len(frame) == len(dataset)
    assert set(frame.columns) >= {"slice_id", "patient_id", "label", "class", "fold"}
