"""LoG and wavelet vs PyRadiomics 3.1.0, and the IBSI filter path."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pyradix import Params, RadixError, RadixExtractor
from pyradix.volume import Volume

_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None
FEATURES = ("Mean", "Minimum", "Maximum", "Energy")


def _ramp():
    image = np.arange(6 * 5 * 4, dtype=np.float64).reshape(6, 5, 4)
    image[1, 1, 1] = -4.0
    image[2, 2, 2] = 0.0
    mask = np.ones(image.shape, dtype=np.float64)
    spacing = (1.5, 1.0, 2.0)
    origin = (0.0, 0.0, 0.0)
    return (
        Volume(image, spacing, origin, ()),
        Volume(mask, spacing, origin, ()),
        image,
        mask.astype(np.uint8),
        spacing,
    )


def test_pyradiomics_log_without_sigma_fails_validation():
    with pytest.raises(RadixError, match="sigma"):
        Params(mode="pyradiomics", image_types=(("LoG", {}),))


@pytest.fixture(scope="module")
def reference() -> dict[str, float]:
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 is not available")
    _, _, image, mask, spacing = _ramp()
    # SimpleITK spacing is (x, y, z); the volume spacing is (z, y, x).
    sitk_spacing = (spacing[2], spacing[1], spacing[0])
    worker = r"""
import json
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

image = np.array(%s, dtype=np.float64)
mask = np.array(%s, dtype=np.uint8)
sitk_image = sitk.GetImageFromArray(image)
sitk_mask = sitk.GetImageFromArray(mask)
sitk_image.SetSpacing(%s)
sitk_mask.SetSpacing(%s)
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25, label=1)
extractor.disableAllFeatures()
extractor.enableFeaturesByName(firstorder=%s)
extractor.disableAllImageTypes()
extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0]})
extractor.enableImageTypeByName("Wavelet")
result = extractor.execute(sitk_image, sitk_mask)
kept = {
    key: float(value)
    for key, value in result.items()
    if key.startswith("log-sigma-") or key.startswith("wavelet-")
}
print(json.dumps(kept))
""" % (
        image.tolist(),
        mask.tolist(),
        sitk_spacing,
        sitk_spacing,
        list(FEATURES),
    )
    completed = subprocess.run(
        [str(PYRAD), "-c", worker],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_log_and_wavelet_match_pyradiomics(reference):
    image, mask, _, _, _ = _ramp()
    params = Params(
        mode="pyradiomics",
        image_types=(("LoG", {"sigma": [1.0, 2.0]}), ("Wavelet", {})),
        feature_classes={"firstorder": list(FEATURES)},
        bin_width=25,
    )
    values = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask).values
    log_keys = [key for key in values if key.startswith("log-sigma-")]
    assert {key.rsplit("_", 2)[0] for key in log_keys} == {
        "log-sigma-1-0-mm-3D",
        "log-sigma-2-0-mm-3D",
    }
    wavelet_keys = [key for key in values if key.startswith("wavelet-")]
    bands = {key.split("_", 1)[0] for key in wavelet_keys}
    assert bands == {
        "wavelet-LLL",
        "wavelet-LLH",
        "wavelet-LHL",
        "wavelet-LHH",
        "wavelet-HLL",
        "wavelet-HLH",
        "wavelet-HHL",
        "wavelet-HHH",
    }
    for key, expected in reference.items():
        got = values[key]
        assert abs(got - expected) <= max(1e-4, 1e-4 * abs(expected)), (key, got, expected)


def test_ibsi_log_and_wavelet_are_finite():
    image, mask, _, _, _ = _ramp()
    params = Params(
        mode="ibsi",
        image_types=(("LoG", {"sigma": [2.0]}), ("Wavelet", {"wavelet": "haar"})),
        feature_classes={"firstorder": ["Mean"]},
        bin_width=1,
    )
    values = RadixExtractor(mode="ibsi", params=params).extract(image, mask).values
    assert np.isfinite(values["log-sigma-2.0_firstorder_Mean"])
    assert np.isfinite(values["wavelet-LLL_firstorder_Mean"])
    assert np.isfinite(values["wavelet-HHH_firstorder_Mean"])
    assert "log-sigma-2-0-mm-3D_firstorder_Mean" not in values
