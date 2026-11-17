"""Diagnostics follow additionalInfo, and only name steps that ran."""

from __future__ import annotations

import numpy as np

from pyradix import Params, RadixExtractor
from pyradix.volume import Volume


def _case():
    image = np.zeros((3, 3, 3), dtype=np.float64)
    image[0, 0, 0] = 1.0
    image[0, 0, 1] = 2.0
    image[2, 2, 2] = 100.0
    mask = np.zeros((3, 3, 3), dtype=np.float64)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = 1.0
    mask[2, 2, 2] = 1.0
    spacing = (1.0, 1.5, 2.0)
    origin = (4.0, 5.0, 6.0)
    return (
        Volume(image, spacing, origin, ()),
        Volume(mask, spacing, origin, ()),
    )


def _extract(**kwargs):
    image, mask = _case()
    params = Params(mode="ibsi", bin_width=25, feature_classes={"firstorder": ["Mean"]}, **kwargs)
    return RadixExtractor(mode="ibsi", params=params).extract(image, mask)


def test_diagnostics_cover_the_original_mask_and_settings():
    result = _extract()
    diag = result.provenance["diagnostics"]
    assert diag["diagnostics_Versions_Radix"]
    assert diag["diagnostics_Versions_Numpy"]
    assert diag["diagnostics_Versions_Python"]
    assert diag["diagnostics_Image-original_Spacing"] == (2.0, 1.5, 1.0)
    assert diag["diagnostics_Image-original_Size"] == (3, 3, 3)
    assert diag["diagnostics_Image-original_Dimensionality"] == "3D"
    assert len(diag["diagnostics_Mask-original_Hash"]) == 40
    assert diag["diagnostics_Mask-original_BoundingBox"] == (0, 0, 0, 3, 3, 3)
    assert diag["diagnostics_Mask-original_VoxelNum"] == 3
    assert diag["diagnostics_Mask-original_VolumeNum"] == 2
    assert diag["diagnostics_Mask-original_Mean"] == (1.0 + 2.0 + 100.0) / 3.0
    assert diag["diagnostics_Mask-original_Minimum"] == 1.0
    assert diag["diagnostics_Mask-original_Maximum"] == 100.0
    index = diag["diagnostics_Mask-original_CenterOfMassIndex"]
    assert index == (1.0, 2.0 / 3.0, 2.0 / 3.0)
    com = diag["diagnostics_Mask-original_CenterOfMass"]
    assert com == (4.0 + index[0] * 2.0, 5.0 + index[1] * 1.5, 6.0 + index[2] * 1.0)
    assert diag["diagnostics_Configuration_EnabledImageTypes"]["Original"] == {}
    assert diag["diagnostics_Configuration_Settings"]["binWidth"] == 25
    assert "diagnostics_Image-interpolated_Spacing" not in diag
    assert "diagnostics_Mask-resegmented_VoxelNum" not in diag
    assert not any(key.startswith("diagnostics_") for key in result.values)


def test_additional_info_false_omits_diagnostics():
    result = _extract(additional_info=False)
    assert "diagnostics" not in result.provenance
    assert result.provenance["config_hash"]
    assert not any(key.startswith("diagnostics_") for key in result.values)


def test_resegmented_prefix_appears_only_after_resegmentation():
    result = _extract(resegment_range=(0.0, 10.0))
    diag = result.provenance["diagnostics"]
    assert diag["diagnostics_Mask-resegmented_VoxelNum"] == 2
    assert diag["diagnostics_Mask-resegmented_VolumeNum"] == 1
    assert diag["diagnostics_Mask-resegmented_Maximum"] == 2.0
    assert "diagnostics_Image-interpolated_Spacing" not in diag


def test_interpolated_prefix_appears_only_after_resampling():
    result = _extract(resampled_pixel_spacing=(2.0, 2.0, 2.0), interpolator="linear")
    diag = result.provenance["diagnostics"]
    assert diag["diagnostics_Image-interpolated_Spacing"] == (2.0, 2.0, 2.0)
    assert diag["diagnostics_Mask-interpolated_Spacing"] == (2.0, 2.0, 2.0)
    assert all(size >= 1 for size in diag["diagnostics_Image-interpolated_Size"])
    assert "diagnostics_Mask-resegmented_VoxelNum" not in diag
