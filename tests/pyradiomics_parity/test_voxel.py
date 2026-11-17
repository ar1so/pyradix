"""Voxel maps vs PyRadiomics 3.1.0 on a tiny ROI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyradix import Params
from pyradix.voxel import extract_voxel_maps

_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None
FEATURES = ("Mean", "Minimum", "Maximum")


def _volume():
    image = np.arange(5 * 5 * 5, dtype=np.float64).reshape(5, 5, 5)
    image[1, 1, 1] = -4.0
    image[0, 0, 0] = 40.0
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    return image, mask


def _reference() -> dict[str, list]:
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 is not available")
    image, mask = _volume()
    worker = r"""
import json
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

image = np.array(%s, dtype=np.float64)
mask = np.array(%s, dtype=np.uint8)
sitk_image = sitk.GetImageFromArray(image)
sitk_mask = sitk.GetImageFromArray(mask)
sitk_image.SetSpacing((1.0, 1.0, 1.0))
sitk_mask.SetSpacing((1.0, 1.0, 1.0))
extractor = featureextractor.RadiomicsFeatureExtractor(
    binWidth=25, label=1, kernelRadius=1, maskedKernel=True, initValue=0, voxelBatch=4, additionalInfo=False
)
extractor.disableAllFeatures()
extractor.enableFeaturesByName(firstorder=%s)
extractor.disableAllImageTypes()
extractor.enableImageTypeByName("Original")
extractor.enableImageTypeByName("Square")
result = extractor.execute(sitk_image, sitk_mask, voxelBased=True)
out = {}
for key, value in result.items():
    if key.startswith("diagnostics_"):
        continue
    out[key] = sitk.GetArrayFromImage(value).astype(float).tolist()
print(json.dumps(out))
""" % (
        image.tolist(),
        mask.tolist(),
        list(FEATURES),
    )
    completed = subprocess.run(
        [str(PYRAD), "-c", worker],
        check=False,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr[-2000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_tiny_roi_voxel_maps_match_pyradiomics():
    image, mask = _volume()
    params = Params(
        mode="pyradiomics",
        image_types=(("Original", {}), ("Square", {})),
        feature_classes={"firstorder": list(FEATURES), "shape": ["VoxelVolume"]},
        bin_width=25,
        voxel_setting={"kernelRadius": 1, "maskedKernel": True, "initValue": 0.0, "voxelBatch": 4},
    )
    maps = extract_voxel_maps(image, mask.astype(bool), params=params)
    assert "original_shape_VoxelVolume" not in maps
    assert not any(key.startswith("diagnostics_") for key in maps)
    reference = _reference()
    roi = mask.astype(bool)
    for key, expected in reference.items():
        got = maps[key]
        expected = np.asarray(expected, dtype=np.float64)
        for g, e in zip(got[roi].ravel(), expected[roi].ravel()):
            assert abs(g - e) <= max(1e-4, 1e-4 * abs(e)), (key, g, e)
