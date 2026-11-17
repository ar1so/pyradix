"""F1: IBSI-2 mean filter S60F on D3 digital filter phantoms.

Quote (IBSI-2 arXiv 2006.05470, §4.2 S60F / Table 6.1):
  g[k] = 1/M^D on odd support M (YNOF).
  1.a.1–4: checkerboard, 3D, M=15, paddings zero / nearest / periodic / mirror.
  1.b.1: impulse, 2D axial, M=15, zero padding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import apply_mean, load_volume

# Impulse is 255 at voxel (31, 31, 31) of a 64^3, 2 mm grid.
IMPULSE_I = 255.0
IMPULSE_IJK = (31, 31, 31)
M = 15


def test_1b1_impulse_2d_zero(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    out = np.asarray(apply_mean(img, M, padding="zero", ndims=2))
    r = M // 2
    z0, y0, x0 = IMPULSE_IJK
    expected = IMPULSE_I / (M * M)
    for y in range(y0 - r, y0 + r + 1):
        for x in range(x0 - r, x0 + r + 1):
            assert abs(out[z0, y, x] - expected) < 1e-10
    assert abs(out[z0, y0, x0] - expected) < 1e-10
    # Other slices stay zero (2D axial, zero pad).
    assert np.max(np.abs(out[z0 - 1])) < 1e-12
    assert np.max(np.abs(out[z0 + 1])) < 1e-12
    # Outside the 15×15 neighbourhood on the impulse slice.
    assert abs(out[z0, y0, x0 + r + 1]) < 1e-12


def test_1a_checkerboard_3d_paddings_agree_in_interior(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    maps = {
        name: np.asarray(apply_mean(img, M, padding=name, ndims=3))
        for name in ("zero", "nearest", "periodic", "mirror")
    }
    r = M // 2
    interior = (
        slice(r, img.shape[0] - r),
        slice(r, img.shape[1] - r),
        slice(r, img.shape[2] - r),
    )
    ref = maps["zero"][interior]
    for name, arr in maps.items():
        if name == "zero":
            continue
        assert np.allclose(arr[interior], ref, atol=1e-9, rtol=0), name
    # Periodic padding: kernel sums to 1 → total intensity conserved.
    assert abs(maps["periodic"].sum() - img.sum()) < 1e-6
    # Interior is a non-trivial average of the 8-voxel checkerboard.
    assert 0.0 < float(np.mean(ref)) < 255.0
