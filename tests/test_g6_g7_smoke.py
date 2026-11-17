"""V3: G6 + G7 smoke on D1."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradix import RadixExtractor, extract_voxel_maps, load_volume, perturbation_table


def test_v3_voxel_and_icc_smoke(d1_image, d1_mask):
    if not Path(d1_image).exists():
        pytest.skip("D1 digital phantom not fetched")
    image = load_volume(d1_image)
    mask = load_volume(d1_mask)
    maps = extract_voxel_maps(image.array, mask.array != 0, voxel_batch=16)
    assert "original_firstorder_Mean" in maps
    rows = perturbation_table(RadixExtractor(mode="ibsi"), image, mask, n=2, seed=2)
    assert rows
