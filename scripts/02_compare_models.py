"""Step 2: compare every feature set and every classifier, patient by patient.

Runs the same 5-fold cross-validation for each combination, with the folds of
the published dataset so that no patient is ever on both sides of a split.
Writes a results table, the per-class metrics, and the figures of the report.

    python scripts/02_compare_models.py
    python scripts/02_compare_models.py --with-deep     # adds VGG16 features
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neuroscan import CLASS_NAMES  # noqa: E402
from neuroscan.dataset import (class_distribution, cross_validation_splits,  # noqa: E402
                               load_index, to_frame)
from neuroscan.evaluation import majority_class_baseline  # noqa: E402
from neuroscan.features import (build_matrix, combined_features,  # noqa: E402
                                histogram_features, statistical_features)
from neuroscan.models import build_models, evaluate  # noqa: E402
from neuroscan.plotting import (comparison, confusion, dataset_overview,  # noqa: E402
                                histogram_signature, roc_curves, sample_images,
                                style)
from neuroscan.texture import texture_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGURES, RESULTS = ROOT / "data", ROOT / "docs" / "figures", ROOT / "results"


def feature_sets(with_deep=False):
    """The descriptions of an image that the classifiers are given."""
    sets = {
        "Histograms": partial(build_matrix, extractor=histogram_features),
        "Statistics": partial(build_matrix, extractor=statistical_features),
        "Histograms + statistics": partial(build_matrix, extractor=combined_features),
        "Texture (LBP + HOG)": partial(build_matrix, extractor=texture_features),
    }
    if with_deep:
        from neuroscan import deep

        sets["Deep features (VGG16)"] = deep.build_matrix
    return sets


def cached(builder, store):
    """Compute the features of a set of slices once, then reuse them."""
    def inner(slices):
        key = tuple(item.slice_id for item in slices)
        if key not in store:
            store[key] = builder(slices)
        return store[key]
    return inner


def describe_dataset(slices, frame):
    print("Dataset\n")
    print(class_distribution(slices).to_string(index=False))
    print(f"\n{len(slices)} slices, {frame.patient_id.nunique()} patients, "
          f"{frame.fold.nunique()} folds")
    spanning = frame.groupby("patient_id")["fold"].nunique()
    print(f"patients present in more than one fold: {(spanning > 1).sum()}")
    print(f"majority-class baseline: {100 * majority_class_baseline(frame.label):.1f} %\n")


def dataset_figures(slices, frame):
    dataset_overview(frame, FIGURES / "dataset_overview.png")

    examples = {}
    for label, name in CLASS_NAMES.items():
        chosen = [item for item in slices if item.label == label][:4]
        examples[name] = [item.path for item in chosen]
    sample_images(examples, FIGURES / "sample_slices.png")

    from neuroscan.features import DEFAULT_BINS, read_grayscale, split_into_regions
    from neuroscan.features import region_histogram

    signatures = {}
    for label, name in CLASS_NAMES.items():
        group = [item for item in slices if item.label == label]
        middles = []
        for item in group:
            regions = split_into_regions(read_grayscale(item.path))
            middles.append(region_histogram(regions[5], DEFAULT_BINS))
        signatures[name] = np.mean(middles, axis=0)
    histogram_signature(signatures, FIGURES / "histogram_signature.png", DEFAULT_BINS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-deep", action="store_true",
                        help="also run VGG16 features (needs TensorFlow)")
    arguments = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    slices = load_index(DATA)
    frame = to_frame(slices)
    describe_dataset(slices, frame)
    dataset_figures(slices, frame)

    models = build_models()
    rows, per_class_rows, best = [], [], None

    for feature_name, builder in feature_sets(arguments.with_deep).items():
        print(f"\n=== {feature_name}")
        features_of = cached(builder, {})
        start = time.perf_counter()
        for model_name, model in models.items():
            result = evaluate(model, cross_validation_splits(slices), features_of,
                              f"{model_name} + {feature_name}")
            summary = result.summary()
            summary["features"] = feature_name
            summary["classifier"] = model_name
            rows.append(summary)
            for entry in result.per_class():
                entry.update(features=feature_name, classifier=model_name)
                per_class_rows.append(entry)
            auc = f"{summary['auc_ovr']:.3f}" if summary["auc_ovr"] else "   -  "
            print(f"  {model_name:<26} balanced acc "
                  f"{100 * summary['balanced_accuracy']:5.1f} % "
                  f"± {100 * summary['balanced_accuracy_std']:4.1f}   "
                  f"macro F1 {summary['macro_f1']:.3f}   AUC {auc}")
            if best is None or summary["balanced_accuracy"] > best[0]["balanced_accuracy"]:
                best = (summary, result)
        print(f"  ({time.perf_counter() - start:.0f}s for this feature set)")

    table = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    table.to_csv(RESULTS / "comparison.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(RESULTS / "per_class.csv", index=False)

    print("\n\nBest five combinations\n")
    display = table.head(5)[["features", "classifier", "balanced_accuracy",
                             "balanced_accuracy_std", "macro_f1", "auc_ovr"]]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    summary, result = best
    print(f"\nBest: {summary['model']}")
    for entry in result.per_class():
        print(f"  {entry['class']:<12} sensitivity {100 * entry['sensitivity']:5.1f} %   "
              f"specificity {100 * entry['specificity']:5.1f} %   F1 {entry['f1']:.3f}")

    baseline = majority_class_baseline(frame.label)
    comparison(rows, FIGURES / "model_comparison.png", baseline=baseline)
    confusion(result.confusion(), FIGURES / "confusion_best.png",
              f"Confusion matrix — {summary['model']}")
    if result.y_score is not None:
        roc_curves(result.y_true, result.y_score, FIGURES / "roc_best.png",
                   f"ROC curves — {summary['model']}")

    (RESULTS / "best_model.json").write_text(
        json.dumps({"summary": summary, "per_class": result.per_class(),
                    "confusion": result.confusion().tolist(),
                    "classes": [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]},
                   indent=2),
        encoding="utf-8")

    print(f"\nTables written to {RESULTS}")
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    style()
    main()
