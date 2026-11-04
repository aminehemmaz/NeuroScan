# Code source

[🇬🇧 English](../source-code.md) · 🇫🇷 Français · [⬅ Retour au README](../../README.fr.md)

Le package `neuroscan` tient en six modules, affichés ici en entier avec ce que fait chacun
et les raisons de son écriture. Les scripts de [`scripts/`](../../scripts) et le
[notebook](../../notebooks/neuroscan_analysis.ipynb) ne font qu'appeler ces fonctions.

Deux modules ne sont pas montrés : `plotting.py`, qui ne fait que dessiner les figures, et
`__init__.py`, qui contient les noms des classes. Les commentaires du code sont en anglais,
la langue standard des projets open source.

| Module | Rôle |
|---|---|
| [`dataset.py`](#datasetpy) | Charger les données et les séparer sans fuite |
| [`features.py`](#featurespy) | Histogrammes par région et statistiques d'intensité |
| [`texture.py`](#texturepy) | Texture : motifs binaires locaux et gradients |
| [`deep.py`](#deeppy) | Caractéristiques profondes d'un réseau pré-entraîné |
| [`models.py`](#modelspy) | Les classifieurs et la boucle d'évaluation |
| [`evaluation.py`](#evaluationpy) | Métriques pour un classifieur médical |

---

## dataset.py

**Charger les données et les séparer sans fuite** — 114 lignes
C'est le module sur lequel repose la crédibilité de tout le projet.

`Slice` porte, à côté du chemin de l'image, l'**identifiant du patient**. Tout le reste en
découle : `cross_validation_splits` produit les cinq plis publiés avec le jeu de données et
appelle `check_no_patient_overlap` sur chacun, qui **lève une erreur** au lieu d'avertir.
Une séparation qui fuirait ne peut pas passer silencieusement.

`leaky_split` est la mauvaise séparation, conservée volontairement. Elle ne sert jamais à
annoncer un résultat, seulement à mesurer de combien elle en aurait gonflé un. Écrire
l'erreur, la nommer et tester qu'elle en est bien une est plus utile que de l'effacer.

```python
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
```

## features.py

**Histogrammes par région et statistiques d'intensité** — 98 lignes
La description d'où est parti le projet : découper la coupe en grille, décrire chaque case
par l'histogramme de ses niveaux de gris, concaténer.

Deux corrections de la première version sont visibles ici. `split_into_regions` calcule les
bords avec `np.linspace`, pour qu'un côté non divisible par la taille de la grille
conserve sa dernière ligne — la division entière la supprimait silencieusement. Et
`region_histogram` utilise 32 classes sur 0–255 au lieu de 512, qui sur une image 8 bits en
laissaient la moitié définitivement vides.

`build_matrix` est l'endroit où la seconde forme de fuite est écartée : il renvoie **une
ligne par coupe**, jamais une ligne par région. Les régions décrivent une image ; ce ne
sont pas des données supplémentaires.

```python
"""Handcrafted features: regional grey-level histograms, and texture statistics.

The original idea of the project is kept: an MRI slice is cut into a grid of
regions, and each region is described by the histogram of its grey levels. A
tumour changes the intensity distribution of the region it sits in, so the
concatenated histograms carry a usable signal while staying easy to read.
"""

import cv2
import numpy as np

DEFAULT_ROWS, DEFAULT_COLS, DEFAULT_BINS = 4, 4, 32


def read_grayscale(path, size=256):
    """Load an image as a square 8-bit grey-level array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    if size and image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def split_into_regions(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """Cut an image into a ``rows x cols`` grid, without losing the last pixels."""
    height, width = image.shape[:2]
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    col_edges = np.linspace(0, width, cols + 1).astype(int)
    return [
        image[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]]
        for i in range(rows) for j in range(cols)
    ]


def region_histogram(region, bins=DEFAULT_BINS):
    """Normalised histogram of the grey levels of a region.

    An 8-bit image holds values 0 to 255, so the histogram is computed over that
    range only: asking for more bins than there are grey levels, as a first
    version of this project did with 512 bins, leaves half of them always empty.
    """
    histogram = cv2.calcHist([region], [0], None, [bins], [0, 256]).ravel()
    total = histogram.sum()
    return histogram / total if total else histogram


def histogram_features(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, bins=DEFAULT_BINS):
    """Concatenated histograms of every region: one vector per image."""
    return np.concatenate([
        region_histogram(region, bins) for region in split_into_regions(image, rows, cols)
    ])


def intensity_statistics(region):
    """Five simple statistics describing the intensity of a region."""
    values = region.astype(np.float64).ravel()
    if values.size == 0:
        return np.zeros(5)
    mean = values.mean()
    std = values.std()
    centred = values - mean
    skew = (centred ** 3).mean() / (std ** 3) if std > 1e-9 else 0.0
    kurtosis = (centred ** 4).mean() / (std ** 4) if std > 1e-9 else 0.0
    probabilities = np.bincount(region.ravel(), minlength=256) / values.size
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * np.log2(nonzero)).sum()
    return np.array([mean / 255, std / 255, skew, kurtosis, entropy / 8])


def statistical_features(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """Intensity statistics of every region: a short, readable descriptor."""
    return np.concatenate([
        intensity_statistics(region) for region in split_into_regions(image, rows, cols)
    ])


def combined_features(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, bins=DEFAULT_BINS):
    """Histograms and statistics of every region, in a single vector."""
    return np.concatenate([
        histogram_features(image, rows, cols, bins),
        statistical_features(image, rows, cols),
    ])


def build_matrix(slices, extractor=histogram_features, size=256, **kwargs):
    """Feature matrix ``X`` and label vector ``y`` for a list of slices.

    One row per slice, never one row per region: a region is not an independent
    observation, and treating it as one is what inflated the scores of the first
    version of this project.
    """
    features, labels = [], []
    for item in slices:
        image = read_grayscale(item.path, size=size)
        features.append(extractor(image, **kwargs))
        labels.append(item.label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels)
```

## texture.py

**Texture : motifs binaires locaux et gradients** — 59 lignes
Un histogramme compte combien de pixels de chaque luminosité contient une région, et ne
dit rien de leur agencement. Ces deux descripteurs regardent justement l'agencement.

`lbp_histogram` compare chaque pixel à ses voisins sur un cercle et enregistre la suite des
réponses « plus clair / plus sombre ». Comme il ne compare que des voisins, il ne change
pas quand toute l'image s'éclaircit — utile en IRM, où l'intensité n'a pas de valeur
absolue. `hog_features` enregistre la direction des contours, ce qui sépare le bord net
d'un méningiome du bord flou d'un gliome.

Ensemble, ils forment le jeu de descripteurs qui remporte la comparaison.

```python
"""Texture descriptors: local binary patterns and gradient histograms.

A histogram says how many pixels of each brightness a region holds, but nothing
about how they are arranged. Two very different tissues can share a histogram.
The descriptors below look at the local arrangement instead, which is what tells
a smooth region from a textured one.
"""

import numpy as np
from skimage.feature import hog, local_binary_pattern

from .features import DEFAULT_COLS, DEFAULT_ROWS, split_into_regions

LBP_POINTS, LBP_RADIUS = 8, 1


def lbp_histogram(region, points=LBP_POINTS, radius=LBP_RADIUS):
    """Histogram of rotation-invariant uniform local binary patterns.

    Each pixel is compared with its neighbours on a circle; the pattern of
    "brighter / darker" answers is the code of the pixel. Uniform patterns
    correspond to edges, corners and flat areas, and there are ``points + 2``
    of them.
    """
    codes = local_binary_pattern(region, points, radius, method="uniform")
    bins = points + 2
    histogram, _ = np.histogram(codes.ravel(), bins=bins, range=(0, bins))
    total = histogram.sum()
    return histogram / total if total else histogram.astype(float)


def lbp_features(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS,
                 points=LBP_POINTS, radius=LBP_RADIUS):
    """Local binary pattern histogram of every region of the grid."""
    return np.concatenate([
        lbp_histogram(region, points, radius)
        for region in split_into_regions(image, rows, cols)
    ])


def hog_features(image, orientations=9, pixels_per_cell=32, cells_per_block=2):
    """Histogram of oriented gradients over the whole slice.

    Describes the direction of the edges, which separates the rounded, well
    defined border of a pituitary tumour from the diffuse border of a glioma.
    """
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def texture_features(image, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """Local binary patterns and gradient histograms in one vector."""
    return np.concatenate([lbp_features(image, rows, cols), hog_features(image)])
```

## deep.py

**Caractéristiques profondes d'un réseau pré-entraîné** — 81 lignes
Transfert d'apprentissage : VGG16 entraîné sur des photographies, privé de son
classifieur, utilisé seulement pour transformer une image en 512 nombres.

`pooling="avg"` est le choix qui compte. Conserver la sortie brute donnerait 25 088 nombres
par image, pour 449 images — une recette pour le surapprentissage. Moyenner chaque carte de
caractéristiques en donne 512, qui généralisent.

`extract` renvoie **un vecteur par image**. La première version de ce projet entraînait sur
une région à la fois mais prédisait sur huit régions concaténées : entraînement et
prédiction utilisaient des vecteurs de longueurs différentes, et la prédiction plantait.

TensorFlow est optionnel ; `available()` permet au reste du projet de fonctionner sans lui.

```python
"""Deep features: a VGG16 network pre-trained on ImageNet, used as an extractor.

Training a convolutional network from scratch needs far more than a few hundred
MRI slices. The usual answer is transfer learning: keep a network trained on a
large collection of natural images, drop its classifier, and use the activations
of its last convolutional block as a description of the image. A small
classifier is then trained on those vectors.

TensorFlow is an optional dependency. Import this module only when it is
installed; the rest of the project runs without it.
"""

import numpy as np

_MODEL = None
INPUT_SIZE = 224


def available():
    """Whether TensorFlow can be imported in this environment."""
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


def get_model(pooling="avg"):
    """VGG16 without its classifier, loaded once and kept in memory.

    ``pooling="avg"`` averages each of the 512 feature maps, which gives a
    512-number vector per image instead of 25 088. On a few hundred images, the
    shorter vector is the one that generalises.
    """
    global _MODEL
    if _MODEL is None:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow.keras.applications import VGG16

        _MODEL = VGG16(weights="imagenet", include_top=False, pooling=pooling,
                       input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    return _MODEL


def preprocess(images):
    """Grey-level slices to the batch VGG16 expects."""
    import cv2
    from tensorflow.keras.applications.vgg16 import preprocess_input

    batch = np.stack([
        cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        for image in images
    ]).astype("float32")
    return preprocess_input(batch)


def extract(images, batch_size=16, pooling="avg"):
    """Feature vector of each image, as a 2-D array.

    One vector per **image**, not per region: a region is not an independent
    observation, and the first version of this project crashed at prediction
    time because it trained on one region but predicted on eight concatenated.
    """
    model = get_model(pooling=pooling)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start:start + batch_size])
        outputs.append(model.predict(batch, verbose=0))
    return np.vstack(outputs).astype(np.float32)


def build_matrix(slices, size=256, batch_size=16, pooling="avg"):
    """Deep feature matrix ``X`` and labels ``y`` for a list of slices."""
    from .features import read_grayscale

    images = [read_grayscale(item.path, size=size) for item in slices]
    return extract(images, batch_size=batch_size, pooling=pooling), \
        np.asarray([item.label for item in slices])
```

## models.py

**Les classifieurs et la boucle d'évaluation** — 118 lignes
`NearestClassHistogram` est la règle d'où est parti ce projet, réécrite correctement : elle
moyenne **tous** les vecteurs d'entraînement d'une classe. L'original reconstruisait son
accumulateur à l'intérieur de la boucle sur les fichiers, si bien que seule la dernière
image de chaque dossier était moyennée — le modèle était entraîné sur une image par classe
sans que personne ne le remarque.

Elle est conservée, bien qu'elle perde, parce qu'une comparaison sans plancher ne veut rien
dire.

`evaluate` est volontairement banal : pour chaque pli, vérifier la séparation, construire
les descripteurs, cloner le modèle, ajuster, prédire, stocker. Tous les modèles suivent le
même chemin, seule façon pour que les 25 chiffres du tableau de comparaison puissent être
lus les uns contre les autres.

```python
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
```

## evaluation.py

**Métriques pour un classifieur médical** — 115 lignes
L'accuracy est calculée, mais jamais donnée seule. `summary` renvoie ensemble l'accuracy
équilibrée, sa dispersion entre plis, le F1 macro et l'AUC un-contre-tous.

`per_class` est la partie que lirait un clinicien : sensibilité (sur les vrais cas de ce
type, combien ont été trouvés) et spécificité (sur les cas qui ne sont pas ce type, combien
ont été correctement écartés), calculées depuis la matrice de confusion pour chaque classe.

Ce sont elles qui révèlent que le meilleur modèle trouve ici 92,7 % des tumeurs pituitaires
mais seulement 74,7 % des gliomes — une différence que le chiffre unique de 84,2 % masque
complètement.

```python
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
```
