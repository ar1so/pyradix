"""C1: PAT1 geometry vs IBSI interpolation notes (docs/PREPROCESS.md)."""

from pathlib import Path

import numpy as np
import pytest

from pyradix import RadixExtractor, load_volume, resample_geometry


def test_pat1_ibsi_2mm_shape(d2_image):
    if not Path(d2_image).exists():
        pytest.skip("D2 PAT1 not fetched")
    image = load_volume(d2_image)
    origin_zyx = (image.origin[2], image.origin[1], image.origin[0])
    dim, spacing, origin = resample_geometry(
        tuple(int(s) for s in image.array.shape),
        tuple(float(s) for s in image.spacing),
        origin_zyx,
        "ibsi",
        (2.0, 2.0, 2.0),
    )
    # IBSI: n' = ceil(n s / s'). PAT1 → (90, 99, 100). Quoted in PREPROCESS.md.
    assert dim == (90, 99, 100)
    assert spacing == (2.0, 2.0, 2.0)
    nz, ny, nx = (float(v) for v in image.array.shape)
    sz, sy, sx = image.spacing
    center_z = origin_zyx[0] + 0.5 * (nz - 1.0) * sz
    assert abs(origin[0] + 0.5 * (dim[0] - 1) * 2.0 - center_z) < 1e-6


def test_corner_grid_shifts_by_half_spacing(d2_image):
    if not Path(d2_image).exists():
        pytest.skip("D2 PAT1 not fetched")
    image = load_volume(d2_image)
    origin_zyx = (image.origin[2], image.origin[1], image.origin[0])
    _dim, _spacing, origin = resample_geometry(
        tuple(int(s) for s in image.array.shape),
        tuple(float(s) for s in image.spacing),
        origin_zyx,
        "pyradiomics",
        (2.0, 2.0, 2.0),
    )
    expected = tuple(
        origin_zyx[axis] + 0.5 * (2.0 - image.spacing[axis]) for axis in range(3)
    )
    assert origin == pytest.approx(expected)


def test_extract_resamples_pat1_roi(d2_image, d2_mask):
    if not Path(d2_image).exists():
        pytest.skip("D2 PAT1 not fetched")
    ex = RadixExtractor(
        mode="ibsi",
        params={
            "bin_width": 25.0,
            "resampled_pixel_spacing": (2.0, 2.0, 2.0),
            "interpolator": "linear",
        },
    )
    result = ex.extract(d2_image, d2_mask, label=1)
    assert result.provenance["resampled_shape"] == [90, 99, 100]
    assert result.values["original_firstorder_Minimum"] == pytest.approx(
        round(result.values["original_firstorder_Minimum"])
    )
    assert np.isfinite(result.values["original_shape_VoxelVolume"])
    assert result.provenance["roi_voxels"] > 0
