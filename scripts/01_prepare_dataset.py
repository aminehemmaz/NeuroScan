"""Step 1: build the image dataset from the archives published on Figshare.

Reads the four ``brainTumorDataPublic_*.zip`` archives, keeps a balanced subset
of patients, and writes one PNG per slice plus ``data/index.csv``
carrying the slice id, the patient id, the tumour class and the official
cross-validation fold.

The patient id is what makes an honest evaluation possible: several slices come
from the same patient, and they must never be split between training and test.

    python scripts/01_prepare_dataset.py --archives path/to/zips
"""

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroscan import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
ARCHIVE_NAMES = [
    "brainTumorDataPublic_1-766.zip",
    "brainTumorDataPublic_767-1532.zip",
    "brainTumorDataPublic_1533-2298.zip",
    "brainTumorDataPublic_2299-3064.zip",
]


def read_metadata(blob):
    """Label and patient id of one .mat slice, without loading the image."""
    with h5py.File(io.BytesIO(blob), "r") as handle:
        group = handle["cjdata"]
        label = int(np.array(group["label"]).ravel()[0])
        patient = "".join(chr(int(code)) for code in np.array(group["PID"]).ravel())
    return label, patient.strip()


def read_slice(blob):
    """Label, patient id, image and tumour mask of one .mat slice.

    Patient ids are not all numeric: some are written like ``MR048294``. They
    are therefore kept as text.
    """
    with h5py.File(io.BytesIO(blob), "r") as handle:
        group = handle["cjdata"]
        label = int(np.array(group["label"]).ravel()[0])
        patient = "".join(chr(int(code)) for code in np.array(group["PID"]).ravel())
        image = np.array(group["image"])
        mask = np.array(group["tumorMask"]).astype(np.uint8)
    return label, patient.strip(), image, mask


def to_png(image, size):
    """16-bit MRI intensities to an 8-bit square image, scaled per slice."""
    image = image.astype(np.float64)
    low, high = image.min(), image.max()
    scaled = np.zeros_like(image) if high <= low else (image - low) / (high - low)
    scaled = (255 * scaled).astype(np.uint8)
    if scaled.shape[:2] != (size, size):
        scaled = cv2.resize(scaled, (size, size), interpolation=cv2.INTER_AREA)
    return scaled


def read_folds(path):
    """Official 5-fold assignment, one fold per slice id."""
    with h5py.File(path, "r") as handle:
        key = "cvind" if "cvind" in handle else "cvInd"
        return np.array(handle[key]).ravel().astype(int)


def collect(archives):
    """Every slice of the four archives, grouped by class then by patient."""
    by_class = defaultdict(lambda: defaultdict(list))
    for name in ARCHIVE_NAMES:
        path = archives / name
        if not path.exists():
            raise SystemExit(f"missing archive: {path}\nDownload the four archives from "
                             "https://doi.org/10.6084/m9.figshare.1512427")
        with zipfile.ZipFile(path) as archive:
            for entry in archive.namelist():
                if not entry.endswith(".mat"):
                    continue
                slice_id = int(Path(entry).stem)
                label, patient = read_metadata(archive.read(entry))
                by_class[label][patient].append((slice_id, path.name, entry))
    return by_class


def choose(by_class, patients_per_class, slices_per_patient, seed):
    """Pick the same number of patients per class, and a few slices each."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(by_class):
        patients = sorted(by_class[label])
        if len(patients) < patients_per_class:
            raise SystemExit(f"class {CLASS_NAMES[label]} has only {len(patients)} patients")
        # Prefer the patients with the most slices, then draw at random among them.
        ranked = sorted(patients, key=lambda p: -len(by_class[label][p]))[:patients_per_class * 2]
        picked = rng.choice(ranked, size=patients_per_class, replace=False)
        for patient in sorted(str(p) for p in picked):
            slices = sorted(by_class[label][patient])
            step = max(1, len(slices) // slices_per_patient)
            for slice_id, archive_name, entry in slices[::step][:slices_per_patient]:
                chosen.append((label, patient, slice_id, archive_name, entry))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True,
                        help="folder holding the four Figshare .zip archives")
    parser.add_argument("--folds", type=Path, default=None, help="cvind.mat (optional)")
    parser.add_argument("--patients-per-class", type=int, default=25)
    parser.add_argument("--slices-per-patient", type=int, default=6)
    parser.add_argument("--size", type=int, default=256, help="output image size in pixels")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    print("Reading the archives (this takes a couple of minutes)...")
    by_class = collect(arguments.archives)
    for label in sorted(by_class):
        slices = sum(len(v) for v in by_class[label].values())
        print(f"  {CLASS_NAMES[label]:<12} {len(by_class[label]):>4} patients, {slices:>5} slices")

    chosen = choose(by_class, arguments.patients_per_class, arguments.slices_per_patient,
                    arguments.seed)
    print(f"\nKeeping {len(chosen)} slices from "
          f"{len({(c[0], c[1]) for c in chosen})} patients")

    folds = read_folds(arguments.folds) if arguments.folds else None

    IMAGES.mkdir(parents=True, exist_ok=True)
    for old in IMAGES.glob("*.png"):
        old.unlink()

    archives = {}
    rows = []
    for label, patient, slice_id, archive_name, entry in sorted(chosen):
        if archive_name not in archives:
            archives[archive_name] = zipfile.ZipFile(arguments.archives / archive_name)
        _, _, image, mask = read_slice(archives[archive_name].read(entry))
        name = f"{slice_id:04d}_{patient}_c{label}.png"
        cv2.imwrite(str(IMAGES / name), to_png(image, arguments.size))
        rows.append({
            "slice_id": slice_id,
            "patient_id": patient,
            "label": label,
            "class": CLASS_NAMES[label],
            "fold": int(folds[slice_id - 1]) if folds is not None else 0,
            "tumour_pixels": int(mask.sum()),
            "file": f"images/{name}",
        })
    for archive in archives.values():
        archive.close()

    if folds is None:
        rows = assign_folds(rows, arguments.seed)

    fields = ["slice_id", "patient_id", "label", "class", "fold", "tumour_pixels", "file"]
    with (DATA / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} images written to {IMAGES}")
    print(f"Index written to {DATA / 'index.csv'}")
    report(rows)


def assign_folds(rows, seed, n_folds=5):
    """Fallback when cvind.mat is absent: 5 folds built patient by patient."""
    rng = np.random.default_rng(seed)
    patients = sorted({(row["label"], row["patient_id"]) for row in rows})
    rng.shuffle(patients)
    by_class = defaultdict(list)
    for label, patient in patients:
        by_class[label].append(patient)
    fold_of = {}
    for label, members in by_class.items():
        for position, patient in enumerate(members):
            fold_of[(label, patient)] = position % n_folds + 1
    for row in rows:
        row["fold"] = fold_of[(row["label"], row["patient_id"])]
    return rows


def report(rows):
    print("\nSlices per class and per fold")
    folds = sorted({row["fold"] for row in rows})
    header = f"{'class':<12}" + "".join(f"{f'fold {f}':>9}" for f in folds) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    for label in sorted(CLASS_NAMES):
        line = f"{CLASS_NAMES[label]:<12}"
        total = 0
        for fold in folds:
            count = sum(1 for row in rows if row["label"] == label and row["fold"] == fold)
            total += count
            line += f"{count:>9}"
        print(line + f"{total:>9}")


if __name__ == "__main__":
    main()
