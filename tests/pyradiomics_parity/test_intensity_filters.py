"""Square, SquareRoot, Logarithm, and Exponential vs PyRadiomics 3.1.0."""

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
TYPES = ("Square", "SquareRoot", "Logarithm", "Exponential")
PREFIX = {
    "Square": "square",
    "SquareRoot": "squareroot",
    "Logarithm": "logarithm",
    "Exponential": "exponential",
}
FEATURES = ("Mean", "Minimum", "Maximum", "Energy")


def _signed_volume():
    """Peak magnitude sits outside the ROI, so the filter scale uses the whole image."""
    image = np.full((4, 4, 4), 0.5, dtype=np.float64)
    image[0, 0, 0] = 25.0
    image[1, 1, 1] = -6.0
    image[1, 1, 2] = 3.0
    image[1, 2, 1] = -1.5
    image[1, 2, 2] = 0.0
    image[2, 1, 1] = 8.0
    image[2, 1, 2] = -0.25
    image[2, 2, 1] = 4.0
    image[2, 2, 2] = 1.0
    mask = np.zeros((4, 4, 4), dtype=np.float64)
    mask[1:3, 1:3, 1:3] = 1.0
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    return (
        Volume(image, spacing, origin, ()),
        Volume(mask, spacing, origin, ()),
        image,
        mask.astype(np.uint8),
    )


@pytest.fixture(scope="module")
def reference() -> dict[str, float]:
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 is not available")
    _, _, image, mask = _signed_volume()
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
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25, label=1)
extractor.disableAllFeatures()
extractor.enableFeaturesByName(firstorder=%s)
extractor.disableAllImageTypes()
for name in %s:
    extractor.enableImageTypeByName(name)
result = extractor.execute(sitk_image, sitk_mask)
print(json.dumps({key: float(value) for key, value in result.items() if not key.startswith("diagnostics_")}))
""" % (
        image.tolist(),
        mask.tolist(),
        list(FEATURES),
        list(TYPES),
    )
    completed = subprocess.run(
        [str(PYRAD), "-c", worker],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_signed_intensity_filters_match_pyradiomics(reference):
    image, mask, _, _ = _signed_volume()
    params = Params(
        mode="pyradiomics",
        image_types=tuple((name, {}) for name in TYPES),
        feature_classes={"firstorder": list(FEATURES)},
        bin_width=25,
    )
    values = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask).values
    assert values["squareroot_firstorder_Minimum"] < 0
    assert values["logarithm_firstorder_Minimum"] < 0
    assert values["square_firstorder_Minimum"] >= 0
    assert values["exponential_firstorder_Minimum"] > 0
    for name in TYPES:
        for feature in FEATURES:
            key = f"{PREFIX[name]}_firstorder_{feature}"
            got = values[key]
            expected = reference[key]
            assert abs(got - expected) <= max(1e-4, 1e-4 * abs(expected)), (key, got, expected)
