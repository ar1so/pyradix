"""Mask label, ROI size, and geometry checks run before features."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pyradix import RadixError, RadixExtractor
from pyradix.paths import image_io_path
from pyradix.volume import Volume

ROOT = Path(__file__).resolve().parents[1]
_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None


def _volume(array: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> Volume:
    return Volume(array=array, spacing=spacing, origin=origin, direction=())


def _save_nifti(path: Path, array_zyx: np.ndarray, spacing_zyx=(1.0, 1.0, 1.0)) -> None:
    data = np.transpose(array_zyx, (2, 1, 0))
    sz, sy, sx = spacing_zyx
    affine = np.diag([sx, sy, sz, 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def test_all_zero_mask_names_no_labels():
    image = _volume(np.ones((4, 4, 4)))
    mask = _volume(np.zeros((4, 4, 4)))
    with pytest.raises(RadixError, match=r"present labels: \[\]") as exc:
        RadixExtractor(mode="pyradiomics").extract(image, mask, label=1)
    assert exc.value.code == "case"


def test_label_argument_overrides_the_parameter_file():
    image = _volume(np.arange(64, dtype=np.float64).reshape(4, 4, 4))
    mask_array = np.zeros((4, 4, 4))
    mask_array[:2] = 1
    mask_array[2:] = 2
    mask = _volume(mask_array)
    ex = RadixExtractor(mode="pyradiomics", params={"label": 1, "bin_width": 25})
    result = ex.extract(image, mask, label=2)
    assert result.provenance["label"] == 2
    assert result.provenance["roi_voxels"] == 32


def test_roi_smaller_than_minimum_names_the_count():
    image = _volume(np.ones((4, 4, 4)))
    mask = _volume(np.zeros((4, 4, 4)))
    mask.array[:2, :2, :2] = 1
    ex = RadixExtractor(
        mode="pyradiomics",
        params={"minimum_roi_size": 10, "bin_width": 25},
    )
    with pytest.raises(RadixError, match="ROI size 8") as exc:
        ex.extract(image, mask)
    assert "minimumROISize 10" in exc.value.message


def test_roi_with_too_few_dimensions_names_the_size():
    image = _volume(np.ones((4, 4, 4)))
    mask = _volume(np.zeros((4, 4, 4)))
    mask.array[0, 0, :] = 1
    with pytest.raises(RadixError, match="minimumROIDimensions is 2") as exc:
        RadixExtractor(mode="pyradiomics").extract(image, mask)
    assert "ROI size 4" in exc.value.message


def test_geometry_mismatch_names_both_grids():
    image = _volume(np.ones((4, 4, 4)), spacing=(1.0, 1.0, 1.0))
    mask = _volume(np.ones((2, 2, 2)), spacing=(2.0, 2.0, 2.0))
    with pytest.raises(RadixError, match="correctMask is false") as exc:
        RadixExtractor(mode="pyradiomics").extract(image, mask)
    assert "[4, 4, 4]" in exc.value.message
    assert "[2, 2, 2]" in exc.value.message


def test_correct_mask_resamples_before_features():
    image = _volume(np.ones((4, 4, 4)), spacing=(1.0, 1.0, 1.0))
    mask = _volume(np.ones((2, 2, 2)), spacing=(2.0, 2.0, 2.0))
    ex = RadixExtractor(mode="pyradiomics", params={"correct_mask": True, "bin_width": 25})
    result = ex.extract(image, mask, label=1)
    assert result.provenance["label"] == 1
    assert result.provenance["roi_voxels"] > 0
    assert result.values["original_firstorder_Mean"] == 1.0


def test_each_label_matches_pyradiomics(tmp_path: Path):
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics interpreter is not installed")
    image = np.zeros((8, 8, 8), dtype=np.float64)
    mask = np.zeros((8, 8, 8), dtype=np.float64)
    image[:4] = 10
    image[4:] = 40
    mask[:4] = 1
    mask[4:] = 2
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _save_nifti(image_path, image)
    _save_nifti(mask_path, mask)
    ex = RadixExtractor(mode="pyradiomics", params={"bin_width": 25, "label": 1})
    for label in (1, 2):
        radix = ex.extract(image_path, mask_path, label=label)
        reference = _pyradiomics_original(image_path, mask_path, label)
        shared = set(radix.values) & set(reference)
        assert shared
        for key in shared:
            if key.startswith("original_shape"):
                continue
            got = radix.values[key]
            exp = reference[key]
            limit = max(1e-4, 1e-4 * abs(exp))
            assert abs(got - exp) <= limit, (label, key, got, exp)
        assert radix.provenance["label"] == label


def _pyradiomics_original(image: Path, mask: Path, label: int) -> dict[str, float]:
    code = """
import json, sys
from radiomics import featureextractor
image, mask, label = sys.argv[1:]
label = int(label)
ex = featureextractor.RadiomicsFeatureExtractor(binWidth=25, label=label)
ex.disableAllImageTypes()
ex.enableImageTypeByName("Original")
result = ex.execute(image, mask, label=label)
out = {}
for key, value in result.items():
    if str(key).startswith("original_"):
        out[str(key)] = float(value)
print(json.dumps(out))
"""
    proc = subprocess.run(
        [str(PYRAD), "-c", code, image_io_path(image), image_io_path(mask), str(label)],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    return {key: float(value) for key, value in json.loads(proc.stdout).items()}


def test_cohort_runner_records_a_small_roi_as_a_skip():
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_lung1

    assert compare_lung1._roi_policy_skip("RadixError: mask: ROI size 8 is below minimumROISize 10")
    assert not compare_lung1._roi_policy_skip("Timeout: extract exceeded 90s")
