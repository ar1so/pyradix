"""F4: IBSI-2 Gabor filter Q88H on D3 digital filter phantoms.

Quote (IBSI-2 arXiv 2006.05470, §4.5 / Table 6.1):
  g = exp(−(k̃1² + γ² k̃2²)/(2σ²) + j 2π k̃1/λ)
  (k̃1, k̃2) = R_θ k, R_θ = [[cosθ, sinθ], [sinθ, −cosθ]] (clockwise, k1=x, k2=y).
  σ=σ*/s, λ=λ*/s. Crop Mi = Ni if odd else Ni+1 (Eq. 4.5). Response is |h|.
  4.a.1: impulse, zero, 2D modulus, σ*=10 mm, λ*=4 mm, γ=1/2, θ=π/3.
  4.a.2: same + rot-inv Δθ=π/4 average + average over orthogonal planes.
  4.b.1: sphere, mirror, σ*=20 mm, λ*=8 mm, γ=5/2, θ=5π/4.
  4.b.2: same + rot-inv Δθ=π/8 average + orthogonal planes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import apply_gabor, load_volume

IMPULSE_I = 255.0
IMPULSE_IJK = (31, 31, 31)


def _gabor_complex(k1: int, k2: int, sigma: float, lam: float, gamma: float, theta: float) -> complex:
    c, s = np.cos(theta), np.sin(theta)
    t1 = k1 * c + k2 * s
    t2 = k1 * s - k2 * c
    env = np.exp(-(t1 * t1 + gamma * gamma * t2 * t2) / (2.0 * sigma * sigma))
    return env * np.exp(1j * 2.0 * np.pi * t1 / lam)


def _support_radius(n: int) -> int:
    m = n if n % 2 == 1 else n + 1
    return m // 2


def _mirror_index(i: int, n: int) -> int:
    # IBSI: k' = k mod N if floor(k/N) even else N-(k mod N + 1).
    tile = i // n
    rem = i % n
    return rem if tile % 2 == 0 else n - (rem + 1)


def _sample(img: np.ndarray, z: int, y: int, x: int, padding: str) -> float:
    nz, ny, nx = img.shape
    if padding == "zero":
        if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
            return float(img[z, y, x])
        return 0.0
    return float(img[_mirror_index(z, nz), _mirror_index(y, ny), _mirror_index(x, nx)])


def _conv_modulus(
    img: np.ndarray,
    z: int,
    y: int,
    x: int,
    axis1: int,
    axis2: int,
    sigma: float,
    lam: float,
    gamma: float,
    theta: float,
    padding: str,
) -> float:
    r1 = _support_radius(img.shape[axis1])
    r2 = _support_radius(img.shape[axis2])
    acc = 0.0 + 0.0j
    for k2 in range(-r2, r2 + 1):
        for k1 in range(-r1, r1 + 1):
            zz, yy, xx = z, y, x
            for axis, t in ((axis1, k1), (axis2, k2)):
                if axis == 0:
                    zz -= t
                elif axis == 1:
                    yy -= t
                else:
                    xx -= t
            acc += _gabor_complex(k1, k2, sigma, lam, gamma, theta) * _sample(img, zz, yy, xx, padding)
    return float(abs(acc))


def _thetas(delta: float) -> list[float]:
    t = 0.0
    out = []
    while t < np.pi - 1e-12:
        out.append(t)
        t += delta
    return out


def test_4a1_impulse_2d_zero(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    vol = load_volume(d3_impulse)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm, lam_mm, gamma, theta = 10.0, 4.0, 0.5, np.pi / 3.0
    out = np.asarray(
        apply_gabor(img, sigma_mm, lam_mm, gamma, vol.spacing, theta=theta, padding="zero")
    )
    sigma = sigma_mm / vol.spacing[2]
    lam = lam_mm / vol.spacing[2]
    z0, y0, x0 = IMPULSE_IJK
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            expected = IMPULSE_I * abs(_gabor_complex(dx, dy, sigma, lam, gamma, theta))
            got = out[z0, y0 + dy, x0 + dx]
            assert abs(got - expected) < 1e-9, (dy, dx, got, expected)
    assert abs(out[z0 - 1, y0, x0]) < 1e-12
    assert abs(out[z0 + 1, y0, x0]) < 1e-12


def test_4a2_impulse_rotinv_ortho(d3_impulse):
    if not Path(d3_impulse).exists():
        pytest.skip("D3 impulse not fetched")
    vol = load_volume(d3_impulse)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm, lam_mm, gamma = 10.0, 4.0, 0.5
    dtheta = np.pi / 4.0
    out = np.asarray(
        apply_gabor(
            img,
            sigma_mm,
            lam_mm,
            gamma,
            vol.spacing,
            padding="zero",
            rotation_invariant=True,
            delta_theta=dtheta,
            orthogonal_planes=True,
        )
    )
    sigma = sigma_mm / vol.spacing[2]
    lam = lam_mm / vol.spacing[2]
    thetas = _thetas(dtheta)
    z0, y0, x0 = IMPULSE_IJK
    planes = ((2, 1), (2, 0), (1, 0))  # axial, coronal, sagittal
    samples = ((0, 0, 0), (0, 2, 1), (3, 0, -2), (1, -1, 0))
    for dz, dy, dx in samples:
        acc = 0.0
        for th in thetas:
            for a1, a2 in planes:
                off = {2: dx, 1: dy, 0: dz}
                # Impulse contributes only if the unused axis offset is 0.
                unused = {0, 1, 2} - {a1, a2}
                if any(off[u] != 0 for u in unused):
                    continue
                acc += IMPULSE_I * abs(_gabor_complex(off[a1], off[a2], sigma, lam, gamma, th))
        expected = acc / (len(thetas) * len(planes))
        got = out[z0 + dz, y0 + dy, x0 + dx]
        assert abs(got - expected) < 1e-9, ((dz, dy, dx), got, expected)


def test_4b1_sphere_mirror(d3_sphere):
    if not Path(d3_sphere).exists():
        pytest.skip("D3 sphere not fetched")
    vol = load_volume(d3_sphere)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm, lam_mm, gamma, theta = 20.0, 8.0, 2.5, 5.0 * np.pi / 4.0
    out = np.asarray(
        apply_gabor(img, sigma_mm, lam_mm, gamma, vol.spacing, theta=theta, padding="mirror")
    )
    sigma = sigma_mm / vol.spacing[2]
    lam = lam_mm / vol.spacing[2]
    z0 = y0 = x0 = img.shape[0] // 2
    expected = _conv_modulus(img, z0, y0, x0, 2, 1, sigma, lam, gamma, theta, "mirror")
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    assert np.isfinite(out).all()


def test_4b2_sphere_rotinv_ortho(d3_sphere):
    if not Path(d3_sphere).exists():
        pytest.skip("D3 sphere not fetched")
    vol = load_volume(d3_sphere)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm, lam_mm, gamma = 20.0, 8.0, 2.5
    dtheta = np.pi / 8.0
    out = np.asarray(
        apply_gabor(
            img,
            sigma_mm,
            lam_mm,
            gamma,
            vol.spacing,
            padding="mirror",
            rotation_invariant=True,
            delta_theta=dtheta,
            orthogonal_planes=True,
        )
    )
    sigma = sigma_mm / vol.spacing[2]
    lam = lam_mm / vol.spacing[2]
    thetas = _thetas(dtheta)
    z0 = y0 = x0 = img.shape[0] // 2
    planes = ((2, 1), (2, 0), (1, 0))
    expected = 0.0
    for th in thetas:
        for a1, a2 in planes:
            expected += _conv_modulus(img, z0, y0, x0, a1, a2, sigma, lam, gamma, th, "mirror")
    expected /= len(thetas) * len(planes)
    assert abs(out[z0, y0, x0] - expected) < 1e-8
    assert np.isfinite(out).all()
