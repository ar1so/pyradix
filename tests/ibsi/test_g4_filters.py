"""G4: IBSI-2 convolutional filter suite (F1–F5).

VALIDATION.md: filter images match IBSI-2 references where published.
Phase-1 D3 gates are the per-filter modules. This file is the G4 roll-up:
every apply_* entrypoint runs on a synthetic impulse without error.
"""

from __future__ import annotations

import numpy as np

from pyradix import apply_gabor, apply_laws, apply_log, apply_mean, apply_wavelet


def test_g4_synthetic_impulse_all_filters():
    dim = (9, 9, 9)
    img = np.zeros(dim, dtype=np.float64)
    img[4, 4, 4] = 255.0
    spacing = (2.0, 2.0, 2.0)
    maps = {
        "mean": np.asarray(apply_mean(img, 3, padding="zero", ndims=3)),
        "log": np.asarray(apply_log(img, 3.0, spacing, padding="zero", ndims=3)),
        "laws": np.asarray(apply_laws(img, "E5L5S5", padding="zero")),
        "gabor": np.asarray(
            apply_gabor(img, 4.0, 4.0, 0.5, spacing, theta=0.0, padding="zero")
        ),
        "wavelet": np.asarray(apply_wavelet(img, "haar", "LHL", level=1, padding="zero")),
    }
    for name, arr in maps.items():
        assert arr.shape == dim, name
        assert np.isfinite(arr).all(), name
        assert np.max(np.abs(arr)) > 0.0, name
