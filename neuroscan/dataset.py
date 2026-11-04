"""Loading the dataset and splitting it without leaking a patient between folds.

Every image is one MRI slice. Several slices come from the same patient, and two
slices of the same patient look almost alike. Splitting slices at random
therefore puts near-copies of a training image into the test set, and the score
measured that way is far too optimistic.

The functions below always split on the **patient**, never on the slice.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASS_NAMES


@dataclass(frozen=True)
class Slice:
    """One MRI slice and the metadata needed to split it correctly."""

    slice_id: int
    patient_id: str
    label: int
    fold: int
    path: Path

    @property
    def class_name(self):
        return CLASS_NAMES[self.label]


def load_index(data_dir):
    """Read data/index.csv and return the list of slices it describes."""
    data_dir = Path(data_dir)
    table = pd.read_csv(data_dir / "index.csv")
    missing = [name for name in ("slice_id", "patient_id", "label", "fold", "file") if name not in table]
    if missing:
        raise ValueError(f"index.csv is missing the columns {missing}")

    slices = [
        Slice(int(row.slice_id), str(row.patient_id), int(row.label), int(row.fold),
              data_dir / str(row.file))
        for row in table.itertuples()
    ]
    absent = [s.path.name for s in slices if not s.path.exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} images listed in index.csv are missing, "
                                f"for example {absent[0]}")
    return slices


def to_frame(slices):
    """The list of slices as a pandas DataFrame, for tables and figures."""
    return pd.DataFrame([
        {"slice_id": s.slice_id, "patient_id": s.patient_id, "label": s.label,
         "class": s.class_name, "fold": s.fold, "file": s.path.name}
        for s in slices
    ])


def patient_folds(slices):
    """Group the slices by the fold given in the index (folds are patient-safe)."""
    folds = {}
    for item in slices:
        folds.setdefault(item.fold, []).append(item)
    return dict(sorted(folds.items()))


def cross_validation_splits(slices):
    """Yield ``(fold, train, test)`` for each fold, splitting on the patient.

    The fold of a slice comes from the official 5-fold assignment published with
    the dataset, which keeps every slice of a patient inside the same fold.
    """
    folds = patient_folds(slices)
    for fold, test in folds.items():
        train = [item for item in slices if item.fold != fold]
        check_no_patient_overlap(train, test)
        yield fold, train, test


def check_no_patient_overlap(train, test):
    """Raise if a patient appears in both sides of a split."""
    shared = {item.patient_id for item in train} & {item.patient_id for item in test}
    if shared:
        raise AssertionError(f"{len(shared)} patients are in both the training and the test set: "
                             f"{sorted(shared)[:5]}")


def leaky_split(slices, test_size=0.3, seed=42):
    """A deliberately wrong split, on the slice instead of the patient.

    Kept so the documentation can show how much the score is inflated when the
    same patient is on both sides. Never use it to report a result.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(slices))
    cut = int(len(slices) * (1 - test_size))
    train = [slices[i] for i in order[:cut]]
    test = [slices[i] for i in order[cut:]]
    return train, test


def class_distribution(slices):
    """Number of slices and of patients per class."""
    rows = []
    for label, name in CLASS_NAMES.items():
        group = [item for item in slices if item.label == label]
        rows.append({"label": label, "class": name, "slices": len(group),
                     "patients": len({item.patient_id for item in group})})
    return pd.DataFrame(rows)
