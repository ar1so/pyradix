"""`diagnostics_Mask-*_VolumeNum` must match a plain 26-connected flood fill."""

from __future__ import annotations

from collections import deque

import numpy as np

from pyradix.clear import _volume_count


def _reference(roi: np.ndarray) -> int:
    visited = np.zeros(roi.shape, dtype=bool)
    nz, ny, nx = roi.shape
    count = 0
    for z0, y0, x0 in np.argwhere(roi):
        if visited[z0, y0, x0]:
            continue
        count += 1
        visited[z0, y0, x0] = True
        stack = deque([(z0, y0, x0)])
        while stack:
            z, y, x = stack.pop()
            for zz in range(max(z - 1, 0), min(z + 2, nz)):
                for yy in range(max(y - 1, 0), min(y + 2, ny)):
                    for xx in range(max(x - 1, 0), min(x + 2, nx)):
                        if roi[zz, yy, xx] and not visited[zz, yy, xx]:
                            visited[zz, yy, xx] = True
                            stack.append((zz, yy, xx))
    return count


def test_random_masks_match_flood_fill():
    rng = np.random.default_rng(0)
    for density in (0.02, 0.1, 0.3, 0.6):
        for shape in ((1, 9, 11), (5, 7, 6), (12, 10, 8)):
            roi = rng.random(shape) < density
            assert _volume_count(roi) == _reference(roi), (density, shape)


def test_corner_touch_is_one_volume_and_gap_is_two():
    roi = np.zeros((2, 2, 3), dtype=bool)
    roi[0, 0, 0] = roi[1, 1, 1] = True
    assert _volume_count(roi) == 1
    roi[:] = False
    roi[0, 0, 0] = roi[0, 0, 2] = True
    assert _volume_count(roi) == 2


def test_non_contiguous_view():
    roi = np.zeros((4, 6, 8), dtype=bool)
    roi[:, :, ::4] = True
    assert _volume_count(roi[:, ::2, :]) == _reference(np.ascontiguousarray(roi[:, ::2, :]))
