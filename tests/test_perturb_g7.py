"""G7: perturbation run writes a stability table (no crash)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradix import RadixExtractor, load_volume, perturbation_table


def test_g7_d1_perturbation_table(d1_image, d1_mask):
    if not Path(d1_image).exists():
        pytest.skip("D1 digital phantom not fetched")
    image = load_volume(d1_image)
    mask = load_volume(d1_mask)
    rows = perturbation_table(
        RadixExtractor(mode="ibsi"),
        image,
        mask,
        n=3,
        translate_vox=1,
        seed=1,
    )
    assert len(rows) >= 20
    assert all("cv" in row and "feature" in row for row in rows)
    assert all(row["n"] == 3 for row in rows)
