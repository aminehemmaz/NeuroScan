"""Tests for the classifier written from scratch and for the metrics."""

import numpy as np
import pytest

from neuroscan.evaluation import (FoldResult, ModelResult, majority_class_baseline)
from neuroscan.models import NearestClassHistogram, build_models


def toy_problem(seed=0, per_class=30):
    """Three well separated clouds, one per class."""
    rng = np.random.default_rng(seed)
    centres = {1: [0, 0], 2: [6, 0], 3: [0, 6]}
    X, y = [], []
    for label, centre in centres.items():
        X.append(rng.normal(centre, 0.6, (per_class, 2)))
        y += [label] * per_class
    return np.vstack(X), np.array(y)


def test_nearest_class_learns_the_centres():
    X, y = toy_problem()
    model = NearestClassHistogram().fit(X, y)
    assert list(model.classes_) == [1, 2, 3]
    assert model.centroids_[0] == pytest.approx([0, 0], abs=0.3)
    assert model.centroids_[1] == pytest.approx([6, 0], abs=0.3)


def test_nearest_class_separates_easy_clouds():
    X, y = toy_problem()
    model = NearestClassHistogram().fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95


def test_nearest_class_uses_every_training_image():
    """The first version of this project averaged only the last file of a folder."""
    X = np.array([[0.0], [10.0], [100.0], [110.0]])
    y = np.array([1, 1, 2, 2])
    model = NearestClassHistogram().fit(X, y)
    assert model.centroids_[0, 0] == pytest.approx(5.0)
    assert model.centroids_[1, 0] == pytest.approx(105.0)


def test_nearest_class_probabilities_are_valid():
    X, y = toy_problem()
    model = NearestClassHistogram().fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


@pytest.mark.parametrize("name", list(build_models()))
def test_every_model_fits_and_predicts(name):
    X, y = toy_problem()
    model = build_models()[name].fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert set(predictions) <= {1, 2, 3}


def perfect_result():
    y = np.array([1, 1, 2, 2, 3, 3])
    scores = np.eye(3)[y - 1]
    return ModelResult("perfect", [FoldResult(1, y, y.copy(), scores, 0.1)])


def test_perfect_predictions_score_one():
    summary = perfect_result().summary()
    assert summary["accuracy"] == 1.0
    assert summary["balanced_accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["auc_ovr"] == 1.0


def test_balanced_accuracy_punishes_ignoring_a_rare_class():
    """90 common cases, 10 rare ones, all predicted common: accuracy lies."""
    y_true = np.array([1] * 90 + [2] * 10)
    y_pred = np.ones(100, dtype=int)
    result = ModelResult("lazy", [FoldResult(1, y_true, y_pred)])
    summary = result.summary()
    assert summary["accuracy"] == pytest.approx(0.90)
    assert summary["balanced_accuracy"] == pytest.approx(0.50)


def test_per_class_metrics_on_a_known_confusion():
    y_true = np.array([1, 1, 1, 2, 2, 3])
    y_pred = np.array([1, 1, 2, 2, 2, 3])
    rows = ModelResult("x", [FoldResult(1, y_true, y_pred)]).per_class()
    by_name = {row["class"]: row for row in rows}
    assert by_name["meningioma"]["sensitivity"] == pytest.approx(2 / 3)
    assert by_name["meningioma"]["specificity"] == pytest.approx(1.0)
    assert by_name["glioma"]["sensitivity"] == pytest.approx(1.0)
    assert by_name["glioma"]["precision"] == pytest.approx(2 / 3)
    assert by_name["pituitary"]["f1"] == pytest.approx(1.0)


def test_confusion_matrix_totals_match_the_data():
    y_true = np.array([1, 2, 3, 1, 2, 3])
    y_pred = np.array([1, 2, 3, 2, 2, 1])
    matrix = ModelResult("x", [FoldResult(1, y_true, y_pred)]).confusion()
    assert matrix.sum() == len(y_true)
    assert matrix.shape == (3, 3)


def test_folds_are_concatenated():
    folds = [FoldResult(1, np.array([1, 2]), np.array([1, 2])),
             FoldResult(2, np.array([3]), np.array([3]))]
    result = ModelResult("x", folds)
    assert len(result.y_true) == 3
    assert result.summary()["accuracy"] == 1.0


def test_auc_is_none_without_probabilities():
    result = ModelResult("x", [FoldResult(1, np.array([1, 2]), np.array([1, 2]))])
    assert result.summary()["auc_ovr"] is None


def test_majority_baseline():
    assert majority_class_baseline(np.array([1, 1, 1, 2])) == pytest.approx(0.75)
