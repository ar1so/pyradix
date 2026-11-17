"""F5: IBSI-2 undecimated separable wavelets on D3.

Quote (IBSI-2 arXiv 2006.05470, §4.6 / Table 6.1):
  Undecimated (CVCQ). Haar: g_H=[-1/√2, 1/√2], g_L=[1/√2, 1/√2].
  Even kernels: append 0 (odd). À trous “second alternative” (PyWavelets/MATLAB).
  h_LHL: L along k1=x, H along k2=y, L along k3=z.
  Level >1: filter previous LLL with dilated kernels. Rot-inv: Appendix A, average.
  5.a.1–2: db2, impulse, zero, LHL 1st level ± rot-inv average.
  6.a.1–2: coif1, sphere, periodic, HHL 1st level ± rot-inv average.
  7.a.1–2: Haar, checkerboard, mirror, LLL/HHH 2nd level, rot-inv average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import apply_wavelet, load_volume

IMPULSE_I = 255.0
IMPULSE_IJK = (31, 31, 31)

# PyWavelets analysis kernels (IBSI: “from the PyWavelets website”).
PAIRS = {
    "haar": (
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([-1.0, 1.0]) / np.sqrt(2.0),
    ),
    "db2": (
        np.array([-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]),
        np.array([-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]),
    ),
    "coif1": (
        np.array(
            [
                -0.01565572813546454,
                -0.0727326195128539,
                0.38486484686420286,
                0.8525720202123255,
                0.33789766245780922,
                -0.0727326195128539,
            ]
        ),
        np.array(
            [
                0.0727326195128539,
                0.33789766245780922,
                -0.8525720202123255,
                0.38486484686420286,
                0.0727326195128539,
                -0.01565572813546454,
            ]
        ),
    ),
}

ROTS_3D = (
    ((0, False), (1, False), (2, False)),
    ((2, True), (1, False), (0, False)),
    ((0, True), (1, False), (2, True)),
    ((2, False), (1, False), (0, True)),
    ((1, False), (2, False), (0, False)),
    ((1, False), (2, True), (0, True)),
    ((1, False), (0, True), (2, False)),
    ((0, True), (1, True), (2, False)),
    ((1, True), (0, False), (2, False)),
    ((2, True), (0, True), (1, False)),
    ((2, True), (1, True), (0, True)),
    ((2, True), (0, False), (1, True)),
    ((1, True), (0, True), (2, True)),
    ((0, False), (1, True), (2, True)),
    ((1, False), (0, False), (2, True)),
    ((2, False), (0, True), (1, True)),
    ((2, False), (1, True), (0, False)),
    ((2, False), (0, False), (1, False)),
    ((0, True), (2, False), (1, False)),
    ((1, True), (2, False), (0, True)),
    ((0, False), (2, False), (1, True)),
    ((0, True), (2, True), (1, True)),
    ((1, True), (2, True), (0, False)),
    ((0, False), (2, True), (1, False)),
)


def _atrous(base: np.ndarray, level: int) -> np.ndarray:
    if level <= 1:
        k = base.copy()
    else:
        holes = (1 << (level - 1)) - 1
        parts = []
        for c in base:
            parts.append(c)
            parts.extend([0.0] * holes)
        k = np.asarray(parts, dtype=np.float64)
    if len(k) % 2 == 0:
        k = np.append(k, 0.0)
    return k


def _pick(bank: list[np.ndarray], src: int, flipped: bool) -> np.ndarray:
    k = bank[src]
    return k[::-1].copy() if flipped else k


def _coeff(k: np.ndarray, t: int) -> float:
    r = len(k) // 2
    i = t + r
    if i < 0 or i >= len(k):
        return 0.0
    return float(k[i])


def _weight(kx, ky, kz, dz, dy, dx) -> float:
    return _coeff(kz, dz) * _coeff(ky, dy) * _coeff(kx, dx)


def test_5a1_db2_lhl_impulse_zero(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    out = np.asarray(apply_wavelet(img, "db2", "LHL", level=1, padding="zero"))
    lo, hi = PAIRS["db2"]
    lo, hi = _atrous(lo, 1), _atrous(hi, 1)
    z0, y0, x0 = IMPULSE_IJK
    r = max(len(lo), len(hi)) // 2
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                expected = IMPULSE_I * _weight(lo, hi, lo, dz, dy, dx)
                got = out[z0 + dz, y0 + dy, x0 + dx]
                assert abs(got - expected) < 1e-10, (dz, dy, dx, got, expected)


def test_5a2_db2_lhl_rotinv(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    out = np.asarray(
        apply_wavelet(img, "db2", "LHL", level=1, padding="zero", rotation_invariant=True)
    )
    lo, hi = (_atrous(k, 1) for k in PAIRS["db2"])
    bank = [lo, hi, lo]
    z0, y0, x0 = IMPULSE_IJK
    r = 2
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                expected = np.mean(
                    [
                        IMPULSE_I
                        * _weight(
                            _pick(bank, rx, fx),
                            _pick(bank, ry, fy),
                            _pick(bank, rz, fz),
                            dz,
                            dy,
                            dx,
                        )
                        for (rx, fx), (ry, fy), (rz, fz) in ROTS_3D
                    ]
                )
                got = out[z0 + dz, y0 + dy, x0 + dx]
                assert abs(got - expected) < 1e-10, (dz, dy, dx, got, expected)


def test_6a1_coif1_hhl_sphere_periodic(d3_sphere):
    if not Path(d3_sphere).exists():
        pytest.skip("D3 sphere not fetched")
    img = np.ascontiguousarray(load_volume(d3_sphere).array, dtype=np.float64)
    out = np.asarray(apply_wavelet(img, "coif1", "HHL", level=1, padding="periodic"))
    lo, hi = (_atrous(k, 1) for k in PAIRS["coif1"])
    z0 = y0 = x0 = img.shape[0] // 2
    expected = 0.0
    r = max(len(lo), len(hi)) // 2
    nz = img.shape[0]
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                expected += _weight(hi, hi, lo, dz, dy, dx) * img[
                    (z0 - dz) % nz, (y0 - dy) % nz, (x0 - dx) % nz
                ]
    assert abs(out[z0, y0, x0] - expected) < 1e-8


def test_6a2_coif1_hhl_rotinv(d3_sphere):
    if not Path(d3_sphere).exists():
        pytest.skip("D3 sphere not fetched")
    img = np.ascontiguousarray(load_volume(d3_sphere).array, dtype=np.float64)
    out = np.asarray(
        apply_wavelet(img, "coif1", "HHL", level=1, padding="periodic", rotation_invariant=True)
    )
    lo, hi = (_atrous(k, 1) for k in PAIRS["coif1"])
    bank = [hi, hi, lo]
    z0 = y0 = x0 = img.shape[0] // 2
    r = max(len(lo), len(hi)) // 2
    nz = img.shape[0]
    expected = 0.0
    for (rx, fx), (ry, fy), (rz, fz) in ROTS_3D:
        kx, ky, kz = _pick(bank, rx, fx), _pick(bank, ry, fy), _pick(bank, rz, fz)
        acc = 0.0
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    acc += _weight(kx, ky, kz, dz, dy, dx) * img[
                        (z0 - dz) % nz, (y0 - dy) % nz, (x0 - dx) % nz
                    ]
        expected += acc
    expected /= len(ROTS_3D)
    assert abs(out[z0, y0, x0] - expected) < 1e-8


def test_7a1_haar_lll_level2_rotinv(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    out = np.asarray(
        apply_wavelet(img, "haar", "LLL", level=2, padding="mirror", rotation_invariant=True)
    )
    single = np.asarray(apply_wavelet(img, "haar", "LLL", level=2, padding="mirror"))
    # Trailing 0 makes L_odd = [1/√2, 1/√2, 0] asymmetric under flip.
    assert np.max(np.abs(out - single)) > 1e-6
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 0.0


def test_7a2_haar_hhh_level2_rotinv(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    out = np.asarray(
        apply_wavelet(img, "haar", "HHH", level=2, padding="mirror", rotation_invariant=True)
    )
    single = np.asarray(apply_wavelet(img, "haar", "HHH", level=2, padding="mirror"))
    # H is antisymmetric: pooling changes the map.
    assert np.max(np.abs(out - single)) > 1e-6
    assert np.isfinite(out).all()
