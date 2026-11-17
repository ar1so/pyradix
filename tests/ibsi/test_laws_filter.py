"""F3: IBSI-2 Laws filter JTXT / PQSD on D3 digital filter phantoms.

Quote (IBSI-2 arXiv 2006.05470, §4.4 JTXT / Table 6.1):
  1-D kernels are L²-normalised (e.g. g_E5 = 1/√10 · [-1,-2,0,2,1]).
  h_E5L5S5: g_E5 along k1=x, g_L5 along k2=y, g_S5 along k3=z.
  Energy: h_energy[k] = (1/W) Σ |h[k+k0]| over Chebyshev δ; W=(2δ+1)^N.
  Energy after orientation pooling. Appendix A: 24 cube symmetries, max pool.
  3.a.1–3: impulse, E5L5S5, zero; ± rot-inv max; ± energy δ=7.
  3.b.1–3: checkerboard, E3W5R5, mirror; ± rot-inv max; ± energy δ=7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import apply_laws, load_volume

IMPULSE_I = 255.0
IMPULSE_IJK = (31, 31, 31)
DELTA = 7

KERNELS = {
    "L3": np.array([1.0, 2.0, 1.0]) / np.sqrt(6.0),
    "L5": np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / np.sqrt(70.0),
    "E3": np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0),
    "E5": np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / np.sqrt(10.0),
    "S3": np.array([-1.0, 2.0, -1.0]) / np.sqrt(6.0),
    "S5": np.array([-1.0, 0.0, 2.0, 0.0, -1.0]) / np.sqrt(6.0),
    "W5": np.array([-1.0, 2.0, 0.0, -2.0, 1.0]) / np.sqrt(10.0),
    "R5": np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / np.sqrt(70.0),
}

# Appendix A.2: (x, y, z) = (k1, k2, k3) as (bank_index, flip).
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


def _pick(bank: list[np.ndarray], src: int, flipped: bool) -> np.ndarray:
    k = bank[src]
    return k[::-1].copy() if flipped else k


def _outer(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    # g[dz, dy, dx] = kz[dz] * ky[dy] * kx[dx], each 1-D kernel centred.
    return kz[:, None, None] * ky[None, :, None] * kx[None, None, :]


def _bank(names: str) -> list[np.ndarray]:
    return [KERNELS[names[i : i + 2]] for i in range(0, len(names), 2)]


def _weight_at(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, dz: int, dy: int, dx: int) -> float:
    def coeff(k: np.ndarray, t: int) -> float:
        r = len(k) // 2
        i = t + r
        if i < 0 or i >= len(k):
            return 0.0
        return float(k[i])

    return coeff(kz, dz) * coeff(ky, dy) * coeff(kx, dx)


def _conv_at(
    img: np.ndarray,
    z: int,
    y: int,
    x: int,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
) -> float:
    acc = 0.0
    rz, ry, rx = len(kz) // 2, len(ky) // 2, len(kx) // 2
    for tz in range(-rz, rz + 1):
        for ty in range(-ry, ry + 1):
            for tx in range(-rx, rx + 1):
                acc += _weight_at(kx, ky, kz, tz, ty, tx) * img[z - tz, y - ty, x - tx]
    return acc


def test_3a1_impulse_e5l5s5_zero(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    out = np.asarray(apply_laws(img, "E5L5S5", padding="zero"))
    e5, l5, s5 = _bank("E5L5S5")
    z0, y0, x0 = IMPULSE_IJK
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                expected = IMPULSE_I * _weight_at(e5, l5, s5, dz, dy, dx)
                got = out[z0 + dz, y0 + dy, x0 + dx]
                assert abs(got - expected) < 1e-10, (dz, dy, dx, got, expected)
    assert abs(out[z0, y0, x0 + 3]) < 1e-12


def test_3a2_impulse_rotinv_max(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    out = np.asarray(apply_laws(img, "E5L5S5", padding="zero", rotation_invariant=True))
    bank = _bank("E5L5S5")
    z0, y0, x0 = IMPULSE_IJK
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                expected = max(
                    IMPULSE_I
                    * _weight_at(
                        _pick(bank, rx, fx),
                        _pick(bank, ry, fy),
                        _pick(bank, rz, fz),
                        dz,
                        dy,
                        dx,
                    )
                    for (rx, fx), (ry, fy), (rz, fz) in ROTS_3D
                )
                got = out[z0 + dz, y0 + dy, x0 + dx]
                assert abs(got - expected) < 1e-10, (dz, dy, dx, got, expected)
    # Max pooling is not the unsigned single orientation.
    single = np.asarray(apply_laws(img, "E5L5S5", padding="zero"))
    assert np.max(np.abs(out - single)) > 1e-6


def test_3a3_impulse_rotinv_energy(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    img = np.ascontiguousarray(load_volume(d3_impulse).array, dtype=np.float64)
    pooled = np.asarray(apply_laws(img, "E5L5S5", padding="zero", rotation_invariant=True))
    energy = np.asarray(
        apply_laws(img, "E5L5S5", padding="zero", rotation_invariant=True, energy_delta=DELTA)
    )
    z0, y0, x0 = IMPULSE_IJK
    r = DELTA
    # Zero pad: neighbourhood clips at the volume (impulse is far from the border).
    block = np.abs(pooled[z0 - r : z0 + r + 1, y0 - r : y0 + r + 1, x0 - r : x0 + r + 1])
    expected = float(block.mean())
    assert abs(energy[z0, y0, x0] - expected) < 1e-10
    assert energy[z0, y0, x0] > 0.0


def test_3b1_checkerboard_e3w5r5_mirror(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    out = np.asarray(apply_laws(img, "E3W5R5", padding="mirror"))
    e3, w5, r5 = _bank("E3W5R5")
    z0 = y0 = x0 = img.shape[0] // 2
    expected = _conv_at(img, z0, y0, x0, e3, w5, r5)
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    assert np.isfinite(out).all()


def test_3b2_checkerboard_rotinv_max(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    out = np.asarray(apply_laws(img, "E3W5R5", padding="mirror", rotation_invariant=True))
    bank = _bank("E3W5R5")
    z0 = y0 = x0 = img.shape[0] // 2
    expected = max(
        _conv_at(img, z0, y0, x0, _pick(bank, rx, fx), _pick(bank, ry, fy), _pick(bank, rz, fz))
        for (rx, fx), (ry, fy), (rz, fz) in ROTS_3D
    )
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    single = np.asarray(apply_laws(img, "E3W5R5", padding="mirror"))
    assert np.max(np.abs(out - single)) > 1e-6


def test_3b3_checkerboard_rotinv_energy(d3_checkerboard):
    if not Path(d3_checkerboard).exists():
        pytest.skip("D3 checkerboard not fetched")
    img = np.ascontiguousarray(load_volume(d3_checkerboard).array, dtype=np.float64)
    pooled = np.asarray(apply_laws(img, "E3W5R5", padding="mirror", rotation_invariant=True))
    energy = np.asarray(
        apply_laws(img, "E3W5R5", padding="mirror", rotation_invariant=True, energy_delta=DELTA)
    )
    z0 = y0 = x0 = img.shape[0] // 2
    r = DELTA
    block = np.abs(pooled[z0 - r : z0 + r + 1, y0 - r : y0 + r + 1, x0 - r : x0 + r + 1])
    expected = float(block.mean())
    assert abs(energy[z0, y0, x0] - expected) < 1e-8
    assert energy.min() >= 0.0
