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
