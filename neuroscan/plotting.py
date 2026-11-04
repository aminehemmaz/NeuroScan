"""Figures: shared style, dataset overview, metrics and model comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import CLASS_NAMES  # noqa: E402

CLASS_COLOURS = {"meningioma": "#2D7DD2", "glioma": "#D1495B", "pituitary": "#E8A33D"}
MODEL_COLOUR = "#2D7DD2"
HIGHLIGHT = "#D1495B"
GRID = {"color": "#D8DEE4", "linewidth": 0.8}
LABELS = sorted(CLASS_NAMES)


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#8A94A0",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def save(figure, path):
    figure.savefig(path)
    plt.close(figure)
    return path


def dataset_overview(frame, path, title="The dataset"):
    """Slices and patients per class, and how the folds are filled."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    counts = frame["class"].value_counts().reindex(CLASS_COLOURS).fillna(0)
    axes[0].bar(counts.index, counts.values,
                color=[CLASS_COLOURS[name] for name in counts.index], width=0.65)
    axes[0].set_title("Slices per class")
    axes[0].set_ylabel("slices")
    for index, value in enumerate(counts.values):
        axes[0].text(index, value, int(value), ha="center", va="bottom", fontsize=9)

    patients = frame.groupby("class")["patient_id"].nunique().reindex(CLASS_COLOURS).fillna(0)
    axes[1].bar(patients.index, patients.values,
                color=[CLASS_COLOURS[name] for name in patients.index], width=0.65)
    axes[1].set_title("Patients per class")
    axes[1].set_ylabel("patients")
    for index, value in enumerate(patients.values):
        axes[1].text(index, value, int(value), ha="center", va="bottom", fontsize=9)

    pivot = frame.pivot_table(index="fold", columns="class", values="slice_id", aggfunc="count").fillna(0)
    bottom = np.zeros(len(pivot))
    for name in CLASS_COLOURS:
        if name in pivot:
            axes[2].bar(pivot.index, pivot[name], bottom=bottom,
                        color=CLASS_COLOURS[name], label=name, width=0.65)
            bottom += pivot[name].values
    axes[2].set_title("Slices per cross-validation fold")
    axes[2].set_xlabel("fold")
    axes[2].set_ylabel("slices")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(axis="y", **GRID)
        axis.set_axisbelow(True)
    figure.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    return save(figure, path)


def sample_images(paths_by_class, path, title="One slice per tumour type"):
    """A row of example slices, one per class."""
    columns = max(len(items) for items in paths_by_class.values())
    figure, axes = plt.subplots(len(paths_by_class), columns,
                                figsize=(2.3 * columns, 2.5 * len(paths_by_class)))
    axes = np.atleast_2d(axes)
    for row, (name, items) in enumerate(paths_by_class.items()):
        for column in range(columns):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(items):
                import cv2
                image = cv2.imread(str(items[column]), cv2.IMREAD_GRAYSCALE)
                axis.imshow(image, cmap="gray")
                if column == 0:
                    axis.set_title(name, color=CLASS_COLOURS.get(name, "#22313F"),
                                   fontsize=11, fontweight="bold", loc="left")
    figure.suptitle(title, fontsize=13, fontweight="bold")
    return save(figure, path)


def histogram_signature(histograms_by_class, path, bins,
                        title="Average grey-level histogram of a region"):
    """Mean histogram of one region for each class: the signal the models use."""
    figure, axes = plt.subplots(figsize=(9, 4))
    centres = (np.arange(bins) + 0.5) * 256 / bins
    for name, values in histograms_by_class.items():
        axes.plot(centres, values, label=name, color=CLASS_COLOURS.get(name), linewidth=1.8)
    axes.set_xlabel("grey level")
    axes.set_ylabel("share of the pixels of the region")
    axes.set_title(title)
    axes.legend()
    axes.grid(axis="y", **GRID)
    axes.set_axisbelow(True)
    return save(figure, path)


def comparison(rows, path, baseline=None, title="Balanced accuracy, 5-fold patient-wise"):
    """Horizontal bars comparing the models, with the error over the folds."""
    rows = sorted(rows, key=lambda row: row["balanced_accuracy"])
    names = [row["model"] for row in rows]
    values = [100 * row["balanced_accuracy"] for row in rows]
    errors = [100 * row["balanced_accuracy_std"] for row in rows]
    colours = [HIGHLIGHT if row is rows[-1] else MODEL_COLOUR for row in rows]

    figure, axes = plt.subplots(figsize=(9, 0.62 * len(rows) + 2))
    axes.barh(names, values, xerr=errors, color=colours, height=0.6,
              error_kw={"ecolor": "#5A6673", "capsize": 3, "linewidth": 1})
    if baseline is not None:
        axes.axvline(100 * baseline, color="#5A6673", linestyle="--", linewidth=1.2)
        axes.text(100 * baseline, -0.8, f" majority class {100 * baseline:.0f} %",
                  color="#5A6673", fontsize=8, va="top")
    axes.set_xlim(0, 105)
    axes.set_xlabel("balanced accuracy (%)")
    axes.set_title(title)
    for index, (value, error) in enumerate(zip(values, errors)):
        axes.text(value + error + 1.5, index, f"{value:.1f}", va="center", fontsize=9)
    axes.grid(axis="x", **GRID)
    axes.set_axisbelow(True)
    return save(figure, path)


def confusion(matrix, path, title):
    """Confusion matrix, counts and row percentages."""
    names = [CLASS_NAMES[label] for label in LABELS]
    matrix = np.asarray(matrix)
    percentages = 100 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    figure, axes = plt.subplots(figsize=(5.6, 5))
    image = axes.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            shade = "white" if percentages[i, j] > 55 else "#22313F"
            axes.text(j, i, f"{matrix[i, j]}\n{percentages[i, j]:.0f} %",
                      ha="center", va="center", color=shade, fontsize=10)
    axes.set_xticks(range(len(names)), names, rotation=20, ha="right")
    axes.set_yticks(range(len(names)), names)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label="% of the true class")
    return save(figure, path)


def roc_curves(y_true, y_score, path, title="ROC curves, one versus rest"):
    """One ROC curve per class, from the probabilities of a model."""
    figure, axes = plt.subplots(figsize=(6, 5.4))
    for index, label in enumerate(LABELS):
        name = CLASS_NAMES[label]
        truth = (np.asarray(y_true) == label).astype(int)
        if truth.sum() == 0:
            continue
        false_positive, true_positive, _ = roc_curve(truth, y_score[:, index])
        area = np.trapz(true_positive, false_positive)
        axes.plot(false_positive, true_positive, color=CLASS_COLOURS.get(name),
                  linewidth=1.9, label=f"{name} (AUC {area:.3f})")
    axes.plot([0, 1], [0, 1], color="#8A94A0", linestyle="--", linewidth=1)
    axes.set_xlabel("1 − specificity")
    axes.set_ylabel("sensitivity")
    axes.set_title(title)
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(**GRID)
    axes.set_axisbelow(True)
    return save(figure, path)


def leakage_effect(rows, path, title="What splitting on the slice does to the score"):
    """Side-by-side bars: the wrong split against the patient-wise split."""
    names = [row["model"] for row in rows]
    leaky = [100 * row["leaky"] for row in rows]
    honest = [100 * row["patient"] for row in rows]
    positions = np.arange(len(names))

    figure, axes = plt.subplots(figsize=(9.5, 0.7 * len(names) + 2.4))
    axes.barh(positions + 0.2, leaky, height=0.36, color=HIGHLIGHT,
              label="split on the slice (leaking)")
    axes.barh(positions - 0.2, honest, height=0.36, color=MODEL_COLOUR,
              label="split on the patient (correct)")
    axes.set_yticks(positions, names)
    axes.set_xlim(0, 108)
    axes.set_xlabel("balanced accuracy (%)")
    axes.set_title(title)
    for position, (bad, good) in enumerate(zip(leaky, honest)):
        axes.text(bad + 1, position + 0.2, f"{bad:.1f}", va="center", fontsize=8.5)
        axes.text(good + 1, position - 0.2, f"{good:.1f}", va="center", fontsize=8.5)
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(axis="x", **GRID)
    axes.set_axisbelow(True)
    return save(figure, path)
