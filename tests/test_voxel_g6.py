"""Voxel maps stay bounded and omit shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradix import extract_voxel_maps, load_volume


def test_g6_d1_voxel_maps_finish(d1_image, d1_mask):
    if not Path(d1_image).exists():
        pytest.skip("D1 digital phantom not fetched")
    image = load_volume(d1_image)
    mask = load_volume(d1_mask)
    roi = mask.array != 0
    maps = extract_voxel_maps(
        image.array,
        roi,
        kernel_radius=1,
        voxel_batch=8,
        features=("Mean", "Minimum", "Maximum"),
    )
    assert set(maps) == {
        "original_firstorder_Mean",
        "original_firstorder_Minimum",
        "original_firstorder_Maximum",
    }
    for name, arr in maps.items():
        assert arr.shape == image.array.shape, name
        assert np.isfinite(arr[roi]).all(), name
        assert np.all(arr[~roi] == 0.0), name


def test_voxel_texture_is_batched_and_shape_is_omitted():
    image = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    mask = np.zeros(image.shape, dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    narrow = extract_voxel_maps(
        image,
        mask,
        kernel_radius=1,
        voxel_batch=1,
        features=("Mean",),
    )
    wide = extract_voxel_maps(image, mask, kernel_radius=1, features=("Mean",))
    assert np.allclose(narrow["original_firstorder_Mean"], wide["original_firstorder_Mean"])
    from pyradix import Params

    params = Params(
        mode="pyradiomics",
        feature_classes={"firstorder": ["Mean"], "shape": ["VoxelVolume"], "glcm": ["JointEntropy"]},
        voxel_setting={"kernelRadius": 1, "maskedKernel": True, "initValue": -1.0},
    )
    maps = extract_voxel_maps(image, mask, params=params, voxel_batch=2)
    assert "original_shape_VoxelVolume" not in maps
    assert not any(key.startswith("diagnostics_") for key in maps)
    assert maps["original_firstorder_Mean"][~mask][0] == -1.0
    entropy = maps["original_glcm_JointEntropy"]
    assert np.isfinite(entropy[mask]).all()
    assert not np.all(entropy[mask] == -1.0)
