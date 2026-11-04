"""Metrics for a multi-class medical classifier.

Accuracy alone is misleading on this dataset: the three tumour types are not
equally frequent, and a model that never predicts the rarest class can still
look good. Every result is therefore reported with balanced accuracy, the macro
F1 score, and the sensitivity and specificity of each class.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from . import CLASS_NAMES

LABELS = sorted(CLASS_NAMES)


@dataclass
class FoldResult:
    """Predictions of one cross-validation fold."""

    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray = None  # probabilities, when the model gives them
    fit_seconds: float = 0.0


@dataclass
class ModelResult:
    """Everything measured for one model, over all the folds."""

    name: str
    folds: list = field(default_factory=list)

    @property
    def y_true(self):
        return np.concatenate([fold.y_true for fold in self.folds])

    @property
    def y_pred(self):
        return np.concatenate([fold.y_pred for fold in self.folds])

    @property
    def y_score(self):
        if any(fold.y_score is None for fold in self.folds):
            return None
        return np.vstack([fold.y_score for fold in self.folds])

    @property
    def fit_seconds(self):
        return sum(fold.fit_seconds for fold in self.folds)

    def per_fold_balanced_accuracy(self):
        return np.array([balanced_accuracy_score(f.y_true, f.y_pred) for f in self.folds])

    def summary(self):
        """One row of the comparison table."""
        scores = self.per_fold_balanced_accuracy()
        row = {
            "model": self.name,
            "accuracy": float((self.y_true == self.y_pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(self.y_true, self.y_pred)),
            "balanced_accuracy_std": float(scores.std()),
            "macro_f1": float(f1_score(self.y_true, self.y_pred, average="macro", labels=LABELS)),
            "fit_seconds": round(self.fit_seconds, 1),
        }
        row["auc_ovr"] = self.macro_auc()
        return row

    def macro_auc(self):
        """One-vs-rest AUC, averaged over the classes; None without probabilities."""
        scores = self.y_score
        if scores is None or len(np.unique(self.y_true)) < 2:
            return None
        try:
            return float(roc_auc_score(self.y_true, scores, multi_class="ovr",
                                       average="macro", labels=LABELS))
        except ValueError:
            return None

    def confusion(self):
        return confusion_matrix(self.y_true, self.y_pred, labels=LABELS)

    def per_class(self):
        """Sensitivity, specificity, precision and F1 for every class."""
        matrix = self.confusion()
        total = matrix.sum()
        rows = []
        for index, label in enumerate(LABELS):
            true_positive = matrix[index, index]
            false_negative = matrix[index].sum() - true_positive
            false_positive = matrix[:, index].sum() - true_positive
            true_negative = total - true_positive - false_negative - false_positive
            sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            denominator = precision + sensitivity
            rows.append({
                "class": CLASS_NAMES[label],
                "support": int(matrix[index].sum()),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(2 * precision * sensitivity / denominator) if denominator else 0.0,
            })
        return rows


def majority_class_baseline(y_true):
    """Accuracy of always answering the most frequent class."""
    counts = np.bincount(np.asarray(y_true))
    return float(counts.max() / counts.sum())
