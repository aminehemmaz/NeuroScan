"""The classifiers compared in this project, and the loop that evaluates them.

The first model is the rule the project started from: compare an image with the
average histogram of each class and keep the closest. The others are the usual
classifiers, each behind the same scikit-learn interface so that they can be run
through exactly the same evaluation.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import check_no_patient_overlap
from .evaluation import FoldResult, ModelResult


class NearestClassHistogram(BaseEstimator, ClassifierMixin):
    """Nearest class mean, the rule this project started from.

    Training averages the feature vectors of each class. Predicting returns the
    class whose average is closest, in Euclidean distance.

    The first version of the project rebuilt its list of training vectors inside
    the file loop, so only the last image of each folder was ever averaged. Here
    the average really is taken over every training image.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return -distances

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        """Distances turned into scores, so a ROC curve can be drawn."""
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        return weights / weights.sum(axis=1, keepdims=True)


def build_models(seed=42):
    """The classifiers to compare, each on the same features."""
    return {
        "Nearest class histogram": NearestClassHistogram(),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                class_weight="balanced", random_state=seed)),
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Random forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    }


def evaluate(model, splits, features_of, name):
    """Run one model through every fold and collect its predictions.

    `splits` yields ``(fold, train, test)``; `features_of` turns a list of
    slices into ``(X, y)``.
    """
    from sklearn.base import clone

    result = ModelResult(name=name)
    for fold, train, test in splits:
        check_no_patient_overlap(train, test)
        X_train, y_train = features_of(train)
        X_test, y_test = features_of(test)

        estimator = clone(model)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        y_pred = estimator.predict(X_test)
        y_score = None
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_test)
            classes = getattr(estimator, "classes_", None)
            if classes is None and hasattr(estimator, "steps"):
                classes = estimator.steps[-1][1].classes_
            y_score = _align_columns(probabilities, classes)

        result.folds.append(FoldResult(fold, np.asarray(y_test), np.asarray(y_pred),
                                       y_score, elapsed))
    return result


def _align_columns(probabilities, classes):
    """Reorder probability columns to the fixed class order 1, 2, 3."""
    from . import CLASS_NAMES

    labels = sorted(CLASS_NAMES)
    if classes is None:
        return probabilities
    aligned = np.zeros((probabilities.shape[0], len(labels)))
    for position, label in enumerate(classes):
        if label in labels:
            aligned[:, labels.index(label)] = probabilities[:, position]
    return aligned
