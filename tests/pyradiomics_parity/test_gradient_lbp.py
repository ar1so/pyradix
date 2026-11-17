"""Gradient, LBP2D, and LBP3D vs PyRadiomics 3.1.0."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyradix import Params, RadixExtractor
from pyradix.volume import Volume

_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None
FEATURES = ("Mean", "Minimum", "Maximum", "Energy")


def _volume():
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    image[1, 2, 3] = -4.0
    mask = np.zeros(image.shape, dtype=np.float64)
    mask[0, 0, 0] = 1.0
    mask[1:3, 1:4, 2:5] = 1.0
    spacing = (1.5, 1.0, 2.0)
    origin = (0.0, 0.0, 0.0)
    return (
        Volume(image, spacing, origin, ()),
        Volume(mask, spacing, origin, ()),
        image,
        mask.astype(np.uint8),
    )


def _reference(kind: str) -> dict[str, float]:
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 is not available")
    _, _, image, mask = _volume()
    worker = r"""
import json
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

image = np.array(%s, dtype=np.float64)
mask = np.array(%s, dtype=np.uint8)
sitk_image = sitk.GetImageFromArray(image)
sitk_mask = sitk.GetImageFromArray(mask)
sitk_image.SetSpacing((2.0, 1.0, 1.5))
sitk_mask.SetSpacing((2.0, 1.0, 1.5))
kind = %r
settings = dict(binWidth=25, label=1)
types = ["Gradient", "LBP2D", "LBP3D"]
if kind == "custom":
    settings.update(
        gradientUseSpacing=False,
        lbp2DRadius=1.5,
        lbp2DSamples=4,
        lbp2DMethod="default",
        force2Ddimension=1,
        lbp3DLevels=3,
        lbp3DIcosphereRadius=1.5,
        lbp3DIcosphereSubdivision=2,
    )
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.disableAllFeatures()
extractor.enableFeaturesByName(firstorder=%s)
extractor.disableAllImageTypes()
for name in types:
    extractor.enableImageTypeByName(name)
result = extractor.execute(sitk_image, sitk_mask)
print(json.dumps({key: float(value) for key, value in result.items() if not key.startswith("diagnostics_")}))
""" % (
        image.tolist(),
        mask.tolist(),
        kind,
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


def _assert_close(values, reference, prefixes):
    for prefix in prefixes:
        for feature in FEATURES:
            key = f"{prefix}_firstorder_{feature}"
            assert key in reference, key
            got = values[key]
            expected = reference[key]
            assert abs(got - expected) <= max(1e-4, 1e-4 * abs(expected)), (key, got, expected)


def test_default_gradient_and_lbp_match_pyradiomics():
    image, mask, _, _ = _volume()
    params = Params(
        mode="pyradiomics",
        image_types=(("Gradient", {}), ("LBP2D", {}), ("LBP3D", {})),
        feature_classes={"firstorder": list(FEATURES)},
        bin_width=25,
    )
    values = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask).values
    reference = _reference("default")
    _assert_close(
        values,
        reference,
        ("gradient", "lbp-2D", "lbp-3D-m1", "lbp-3D-m2", "lbp-3D-k"),
    )


def test_custom_gradient_and_lbp_settings_match_pyradiomics():
    image, mask, _, _ = _volume()
    params = Params(
        mode="pyradiomics",
        image_types=(("Gradient", {}), ("LBP2D", {}), ("LBP3D", {})),
        feature_classes={"firstorder": list(FEATURES)},
        bin_width=25,
        gradient_use_spacing=False,
        lbp2d_radius=1.5,
        lbp2d_samples=4,
        lbp2d_method="default",
        force_2d_dimension=1,
        lbp3d_levels=3,
        lbp3d_icosphere_radius=1.5,
        lbp3d_icosphere_subdivision=2,
    )
    values = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask).values
    assert "lbp-3D-m4_firstorder_Mean" not in values
    reference = _reference("custom")
    _assert_close(
        values,
        reference,
        ("gradient", "lbp-2D", "lbp-3D-m1", "lbp-3D-m2", "lbp-3D-m3", "lbp-3D-k"),
    )
