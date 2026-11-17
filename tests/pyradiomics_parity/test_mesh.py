"""Mesh volume, surface area, surface-to-volume ratio, and sphericity vs PyRadiomics 3.1.0."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from pyradix import Params, RadixExtractor

ROOT = Path(__file__).resolve().parents[2]
_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None
KEYS = ("MeshVolume", "SurfaceArea", "SurfaceVolumeRatio", "Sphericity")
CASES = {
    "digital_phantom": (
        ROOT / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz",
        ROOT / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz",
    ),
    "brain1": (
        ROOT / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd",
        ROOT / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd",
    ),
    "pat1": (
        ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz",
        ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz",
    ),
}


def _ready() -> bool:
    return PYRAD is not None and PYRAD.is_file() and all(
        image.is_file() and mask.is_file() for image, mask in CASES.values()
    )


@pytest.fixture(scope="module")
def references() -> dict[str, dict[str, float]]:
    if not _ready():
        pytest.skip("PyRadiomics 3.1.0 or the phantom, brain1, and PAT1 volumes are not available")
    worker = r"""
import json, SimpleITK  # noqa: F401
from radiomics import featureextractor
cases = %s
keys = %s
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25, label=1)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("shape")
out = {}
for name, (image, mask) in cases.items():
    result = extractor.execute(image, mask)
    out[name] = {key: float(result["original_shape_" + key]) for key in keys}
print(json.dumps(out))
""" % (
        {name: (str(image), str(mask)) for name, (image, mask) in CASES.items()},
        list(KEYS),
    )
    completed = subprocess.run(
        [str(PYRAD), "-c", worker],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", list(CASES))
def test_pyradiomics_mesh_features(name, references):
    image, mask = CASES[name]
    params = Params(mode="pyradiomics", feature_classes={"shape": list(KEYS)}, bin_width=25)
    values = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask).values
    for key in KEYS:
        reference = references[name][key]
        got = values[f"original_shape_{key}"]
        assert abs(got - reference) <= max(1e-4, 1e-4 * abs(reference)), (name, key, got, reference)
