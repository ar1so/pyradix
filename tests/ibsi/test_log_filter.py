"""F2: IBSI-2 LoG filter L6PA on D3 digital filter phantoms.

Quote (IBSI-2 arXiv 2006.05470, §4.3 L6PA / Table 6.1):
  g_σ[k] = −1/σ² (1/(√(2π)σ))^D (D − ||k||²/σ²) exp(−||k||²/(2σ²))
  σ = σ*/s (voxel units). Support M = 1 + 2⌊dσ + 0.5⌋ with d = 4.
  2.a: impulse, 3D, σ* = 3.0 mm, zero padding.
  2.b: checkerboard, 3D, σ* = 5.0 mm, mirror.
  2.c: checkerboard, 2D, σ* = 5.0 mm, mirror.
  “check that the LoG kernel sums to approximately 0.”
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import apply_log, load_volume

IMPULSE_I = 255.0
IMPULSE_IJK = (31, 31, 31)
CUTOFF = 4.0


def _sigma_vox(sigma_mm: float, spacing: tuple[float, float, float]) -> float:
    return sigma_mm / spacing[2]


def _support_radius(sigma: float, cutoff: float = CUTOFF) -> int:
    return int(np.floor(cutoff * sigma + 0.5))


def _log_weight(k: np.ndarray, sigma: float, ndims: int) -> float:
    r2 = float(np.dot(k, k))
    s2 = sigma * sigma
    norm = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) ** ndims
    return (-1.0 / s2) * norm * (ndims - r2 / s2) * np.exp(-r2 / (2.0 * s2))


def _log_kernel(sigma: float, ndims: int, cutoff: float = CUTOFF) -> np.ndarray:
    r = _support_radius(sigma, cutoff)
    axes = [range(-r, r + 1)] * ndims
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    weights = np.empty(grid.shape[:-1], dtype=np.float64)
    it = np.nditer(weights, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        it[0] = _log_weight(grid[it.multi_index], sigma, ndims)
        it.iternext()
    return weights


def test_2a_impulse_3d_zero(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    vol = load_volume(d3_impulse)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm = 3.0
    sigma = _sigma_vox(sigma_mm, vol.spacing)
    r = _support_radius(sigma)
    assert r == 6  # σ = 1.5 vx → M = 13
    out = np.asarray(apply_log(img, sigma_mm, vol.spacing, padding="zero", ndims=3))
    z0, y0, x0 = IMPULSE_IJK
    # h[p + d] = 255 · g[d]  (even kernel, zero pad).
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                expected = IMPULSE_I * _log_weight(np.array([dz, dy, dx], dtype=np.float64), sigma, 3)
                got = out[z0 + dz, y0 + dy, x0 + dx]
                assert abs(got - expected) < 1e-10, (dz, dy, dx, got, expected)
    # Outside the support, zero-padded impulse stays 0.
    assert abs(out[z0, y0, x0 + r + 1]) < 1e-12
    assert abs(out[z0 + r + 1, y0, x0]) < 1e-12


def test_2b_checkerboard_3d_mirror(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    vol = load_volume(d3_checkerboard)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm = 5.0
    sigma = _sigma_vox(sigma_mm, vol.spacing)
    r = _support_radius(sigma)
    assert r == 10  # σ = 2.5 vx → M = 21
    kernel = _log_kernel(sigma, 3)
    assert abs(float(kernel.sum())) < 1e-3
    out = np.asarray(apply_log(img, sigma_mm, vol.spacing, padding="mirror", ndims=3))
    # Independent discrete conv at one interior voxel (x0 − k).
    z0 = y0 = x0 = img.shape[0] // 2
    expected = 0.0
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                expected += kernel[dz + r, dy + r, dx + r] * img[z0 - dz, y0 - dy, x0 - dx]
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    assert np.isfinite(out).all()


def test_2c_checkerboard_2d_mirror(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    vol = load_volume(d3_checkerboard)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm = 5.0
    sigma = _sigma_vox(sigma_mm, vol.spacing)
    r = _support_radius(sigma)
    kernel = _log_kernel(sigma, 2)
    assert abs(float(kernel.sum())) < 1e-3
    out = np.asarray(apply_log(img, sigma_mm, vol.spacing, padding="mirror", ndims=2))
    z0 = y0 = x0 = img.shape[0] // 2
    expected = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            expected += kernel[dy + r, dx + r] * img[z0, y0 - dy, x0 - dx]
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    # 2D axial: neighbouring slices never mix.
    z1 = z0 - 1
    expected_z1 = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            expected_z1 += kernel[dy + r, dx + r] * img[z1, y0 - dy, x0 - dx]
    assert abs(out[z1, y0, x0] - expected_z1) < 1e-8
    assert np.isfinite(out).all()
