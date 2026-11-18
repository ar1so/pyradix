"""pyradiomics mode bins voxels exactly like PyRadiomics 3.1.0, i.e. like numpy.

PyRadiomics builds bin edges with numpy (`histogram` for a fixed bin count,
`arange` for a fixed bin width) and assigns bins with `numpy.digitize`. On
integer images after z-scoring, many voxels sit exactly on a bin edge, where a
closed-form floor() can disagree with numpy by one bin. First-order Entropy and
Uniformity depend only on the bin counts, so they expose any difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradix import Params, RadixExtractor, Volume


def _volume(array):
    return Volume(array=array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                  direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


def _mask(shape):
    mask = np.zeros(shape)
    mask[2:10, 4:20, 4:20] = 1
    return mask


def _zscored_integer_case(seed):
    """Integers spanning exactly 0..64 inside the ROI, then z-scored.

    With 32 bins every second integer lies on a bin edge, as in MRI after
    z-scoring, which is where closed-form binning and numpy can disagree.
    """
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 65, size=(12, 24, 24)).astype(np.float64)
    mask = _mask(raw.shape)
    raw[2, 4, 4], raw[2, 4, 5] = 0.0, 64.0  # pin the ROI min and max
    image = (raw - raw.mean()) / raw.std()
    return image, mask


def _tenths_case(seed):
    """Multiples of 0.1, so float rounding of bin edges at widths 0.1-0.3 matters."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(-40, 41, size=(12, 24, 24)) * 0.1
    return raw, _mask(raw.shape)


def _numpy_entropy_uniformity(roi, edges):
    bins = np.digitize(roi, edges)
    p = np.bincount(bins)[1:] / roi.size
    p = p[p > 0]
    return -(p * np.log2(p + np.spacing(1))).sum(), (p ** 2).sum()


def _pyradiomics_fixed_width_edges(roi, width):
    low = roi.min() - (roi.min() % width)
    edges = np.arange(low, roi.max() + 2 * width, width)
    return edges if len(edges) > 1 else np.array([edges[0] - 0.5, edges[0] + 0.5])


def _extract(image, mask, **binning):
    params = Params(mode="pyradiomics", label=1, voxel_array_shift=0.0,
                    feature_classes={"firstorder": ["Entropy", "Uniformity"]}, **binning)
    out = RadixExtractor(mode="pyradiomics", params=params).extract(_volume(image), _volume(mask))
    return out.values["original_firstorder_Entropy"], out.values["original_firstorder_Uniformity"]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("bin_count", [8, 32, 64])
def test_fixed_bin_count_matches_numpy_histogram(seed, bin_count):
    image, mask = _zscored_integer_case(seed)
    roi = image[mask.astype(bool)]
    edges = np.histogram(roi, bin_count)[1]
    edges[-1] += 1
    expected = _numpy_entropy_uniformity(roi, edges)
    got = _extract(image, mask, bin_width=None, bin_count=bin_count)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("width", [0.1, 0.25, 0.3])
def test_fixed_bin_width_matches_numpy_arange(seed, width):
    image, mask = _tenths_case(seed)
    roi = image[mask.astype(bool)]
    expected = _numpy_entropy_uniformity(roi, _pyradiomics_fixed_width_edges(roi, width))
    got = _extract(image, mask, bin_width=width, bin_count=None)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
