"""Tests for the feature extractors and the texture descriptors."""

import numpy as np
import pytest

from neuroscan.features import (DEFAULT_BINS, combined_features, histogram_features,
                                intensity_statistics, region_histogram,
                                split_into_regions, statistical_features)
from neuroscan.texture import hog_features, lbp_histogram, lbp_features


def gradient_image(size=64):
    return np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))


def test_regions_cover_the_whole_image():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    regions = split_into_regions(image, rows=4, cols=4)
    assert len(regions) == 16
    assert sum(region.size for region in regions) == image.size


def test_regions_keep_the_last_pixels_when_size_is_odd():
    """A 65-pixel side must not lose its last row: integer division would."""
    image = np.zeros((65, 65), np.uint8)
    regions = split_into_regions(image, rows=4, cols=4)
    assert sum(region.size for region in regions) == image.size


def test_histogram_sums_to_one():
    histogram = region_histogram(gradient_image(), bins=DEFAULT_BINS)
    assert histogram.shape == (DEFAULT_BINS,)
    assert histogram.sum() == pytest.approx(1.0)


def test_histogram_of_a_flat_region_has_one_full_bin():
    histogram = region_histogram(np.full((16, 16), 200, np.uint8), bins=8)
    assert histogram.max() == pytest.approx(1.0)
    assert (histogram > 0).sum() == 1


def test_histogram_of_an_empty_region_does_not_divide_by_zero():
    histogram = region_histogram(np.zeros((0, 0), np.uint8), bins=8)
    assert histogram.sum() == 0


def test_feature_vector_has_the_expected_length():
    features = histogram_features(gradient_image(), rows=4, cols=4, bins=16)
    assert features.shape == (4 * 4 * 16,)


def test_brightness_change_moves_the_histogram():
    image = np.full((32, 32), 50, np.uint8)
    dark = histogram_features(image, rows=2, cols=2, bins=16)
    bright = histogram_features(image + 100, rows=2, cols=2, bins=16)
    assert not np.allclose(dark, bright)


def test_statistics_describe_a_flat_region():
    stats = intensity_statistics(np.full((16, 16), 128, np.uint8))
    assert stats[0] == pytest.approx(128 / 255)
    assert stats[1] == pytest.approx(0.0)      # no spread
    assert stats[4] == pytest.approx(0.0)      # no entropy either


def test_statistics_of_an_empty_region():
    assert intensity_statistics(np.zeros((0, 0), np.uint8)).shape == (5,)


def test_entropy_is_higher_on_noise_than_on_a_flat_region():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    flat = np.full((32, 32), 128, np.uint8)
    assert intensity_statistics(noise)[4] > intensity_statistics(flat)[4]


def test_combined_features_concatenate_both_descriptions():
    image = gradient_image()
    expected = len(histogram_features(image)) + len(statistical_features(image))
    assert len(combined_features(image)) == expected


def test_features_are_finite():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    for extractor in (histogram_features, statistical_features, combined_features,
                      lbp_features):
        assert np.isfinite(extractor(image)).all()


def test_lbp_histogram_sums_to_one():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    histogram = lbp_histogram(image)
    assert histogram.shape == (10,)          # 8 points + 2 uniform patterns
    assert histogram.sum() == pytest.approx(1.0)


def test_lbp_ignores_a_uniform_brightness_shift():
    """Local binary patterns compare neighbours, so adding a constant changes nothing.

    Only the interior is compared: on the border, scikit-image pads the image
    with zeros, and a brightness shift does change how a border pixel compares
    with that padding.
    """
    from skimage.feature import local_binary_pattern

    rng = np.random.default_rng(3)
    image = rng.integers(0, 150, (48, 48), dtype=np.uint8)
    codes = local_binary_pattern(image, 8, 1, method="uniform")[1:-1, 1:-1]
    shifted = local_binary_pattern(image + 100, 8, 1, method="uniform")[1:-1, 1:-1]
    assert np.array_equal(codes, shifted)


def test_hog_separates_two_edge_directions():
    vertical = np.zeros((64, 64), np.uint8)
    vertical[:, 32:] = 255
    horizontal = vertical.T.copy()
    assert not np.allclose(hog_features(vertical), hog_features(horizontal))


def test_hog_length_is_stable():
    first = hog_features(gradient_image(128))
    second = hog_features(np.zeros((128, 128), np.uint8))
    assert first.shape == second.shape
