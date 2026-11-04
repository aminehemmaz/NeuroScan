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
