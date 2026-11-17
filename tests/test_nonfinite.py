"""NaN or infinite voxels inside the ROI give a clear error, not a panic."""

from __future__ import annotations

import numpy as np
import pytest

from pyradix import RadixError, RadixExtractor, Volume


def _vol(array):
    return Volume(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )

def _volumes(bad_value: float):
    rng = np.random.default_rng(0)
    image = rng.normal(100.0, 20.0, (10, 20, 20))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[3:7, 5:15, 5:15] = 1
    image[5, 10, 10] = bad_value
    return image, mask


@pytest.mark.parametrize("mode", ["ibsi", "pyradiomics"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inside_roi_is_a_clear_error(mode, bad_value):
    image, mask = _volumes(bad_value)
    with pytest.raises(RadixError, match=r"NaN or infinite \(first at z=5, y=10, x=10\)"):
        RadixExtractor(mode=mode).extract(_vol(image), _vol(mask))


def test_nonfinite_outside_roi_is_ignored():
    image, mask = _volumes(0.0)
    image[0, 0, 0] = np.nan
    out = RadixExtractor(mode="ibsi").extract(_vol(image), _vol(mask))
    assert all(v == v for v in out.values.values())
