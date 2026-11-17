"""Translation + Gaussian-noise perturbation table (REQ-M10 / G7)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyradix.extractor import ExtractResult, RadixExtractor
from pyradix.volume import Volume


def _shift(arr: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    return np.roll(arr, (dz, dy, dx), axis=(0, 1, 2))


def perturbation_table(
    extractor: RadixExtractor,
    image: Volume,
    mask: Volume,
    *,
    n: int = 5,
    translate_vox: int = 1,
    noise_sigma: float | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Extract ``n`` perturbed copies (integer roll + additive noise).

    Writes one row per feature: mean, sd, cv across perturbations. A one-case
    ICC is not identified; ``icc`` is left empty and ``cv`` is the stability
    column (G7: table is written, no crash).
    """
    if n < 2:
        raise ValueError("need at least 2 perturbations")
    rng = np.random.default_rng(seed)
    img0 = np.asarray(image.array, dtype=np.float64)
    mask0 = mask.array
    sigma = noise_sigma if noise_sigma is not None else 0.01 * float(np.ptp(img0))
    shifts = [(0, 0, 0)]
    for i in range(1, n):
        shifts.append((0, 0, (i * translate_vox) % max(img0.shape[2], 1)))
    series: dict[str, list[float]] = {}
    for i, (dz, dy, dx) in enumerate(shifts):
        noisy = img0 + rng.normal(0.0, sigma, img0.shape)
        vol_i = Volume(
            array=_shift(noisy, dz, dy, dx),
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction,
        )
        mask_i = Volume(
            array=_shift(np.asarray(mask0), dz, dy, dx),
            spacing=mask.spacing,
            origin=mask.origin,
            direction=mask.direction,
        )
        result: ExtractResult = extractor.extract(vol_i, mask_i)
        for key, val in result.values.items():
            series.setdefault(key, []).append(float(val))
    rows: list[dict[str, Any]] = []
    for key, vals in sorted(series.items()):
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        cv = sd / (abs(mean) + 1e-12)
        rows.append(
            {
                "feature": key,
                "n": n,
                "mean": mean,
                "sd": sd,
                "cv": cv,
                "icc": None,
            }
        )
    return rows
