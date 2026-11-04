"""Step 3: measure what splitting on the slice instead of the patient does.

Runs every classifier twice on the same features: once with the official
patient-wise folds, once with a random split of the slices. The second is the
mistake the first version of this project made, and the gap between the two
numbers is the size of the illusion it produced.

    python scripts/03_leakage_demo.py
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from neuroscan.dataset import cross_validation_splits, leaky_split, load_index, to_frame  # noqa: E402
from neuroscan.features import build_matrix, histogram_features  # noqa: E402
from neuroscan.models import build_models, evaluate  # noqa: E402
from neuroscan.plotting import leakage_effect, style  # noqa: E402
from neuroscan.texture import texture_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGURES, RESULTS = ROOT / "data", ROOT / "docs" / "figures", ROOT / "results"


def leaky_score(model, slices, features_of, repeats=5):
    """Balanced accuracy when the split ignores the patient, averaged over repeats."""
    scores = []
    for seed in range(repeats):
        train, test = leaky_split(slices, test_size=0.3, seed=seed)
        X_train, y_train = features_of(train)
        X_test, y_test = features_of(test)
        estimator = clone(model).fit(X_train, y_train)
        scores.append(balanced_accuracy_score(y_test, estimator.predict(X_test)))
    return float(np.mean(scores))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    slices = load_index(DATA)
    frame = to_frame(slices)

    train, test = leaky_split(slices, test_size=0.3, seed=0)
    shared = {item.patient_id for item in train} & {item.patient_id for item in test}
    print("A random split of the slices\n")
    print(f"  {len(train)} training slices, {len(test)} test slices")
    print(f"  patients present on both sides: {len(shared)} out of {frame.patient_id.nunique()}")
    print("  every one of them puts near-identical slices in training and in test\n")

    store = {}

    def features_of(items):
        key = tuple(item.slice_id for item in items)
        if key not in store:
            store[key] = build_matrix(items, extractor=texture_features)
        return store[key]

    rows = []
    print(f"{'classifier':<26}{'patient-wise':>14}{'slice-wise':>13}{'inflation':>12}")
    print("-" * 65)
    for name, model in build_models().items():
        honest = evaluate(model, cross_validation_splits(slices), features_of, name)
        honest_score = honest.summary()["balanced_accuracy"]
        leaky = leaky_score(model, slices, features_of)
        rows.append({"model": name, "patient": honest_score, "leaky": leaky,
                     "inflation": leaky - honest_score})
        print(f"{name:<26}{100 * honest_score:>13.1f}%{100 * leaky:>12.1f}%"
              f"{100 * (leaky - honest_score):>+11.1f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "leakage.csv", index=False)
    leakage_effect(rows, FIGURES / "leakage_effect.png")

    print(f"\nmean inflation: {100 * table['inflation'].mean():+.1f} points")
    print("\nThe slice-wise numbers are not results. They are what the same code")
    print("reports when the test set contains other slices of the training patients.")
    print(f"\nTable written to {RESULTS / 'leakage.csv'}")


if __name__ == "__main__":
    style()
    main()
