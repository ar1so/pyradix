"""The Lung1 cases that used to hang in GLCM MCC must return.

LUNG1-055 is one of the ten. It is small (about 16k tumor voxels) and
previously did not return within 180 s. data/lung1 is local, not in git.

LUNG1-230 is the 148-gray-level GLCM whose usual eigenvalue solver panics.
`pyradiomics` mode must match installed PyRadiomics 3.1.0 on that MCC.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from pyradix import Params, RadixExtractor, load_volume

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "data" / "lung1" / "LUNG1-055"
CASE_230 = ROOT / "data" / "lung1" / "LUNG1-230"
_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None


@contextmanager
def _deadline(seconds: int, message: str):
    """Fail if the block runs past `seconds`.

    POSIX interrupts a hang with SIGALRM. Windows has no SIGALRM, so there the
    block runs to completion and the elapsed time is checked afterwards.
    """
    if hasattr(signal, "SIGALRM"):

        def _alarm(_signum, _frame):
            raise TimeoutError(message)

        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    else:
        start = time.monotonic()
        yield
        if time.monotonic() - start > seconds:
            raise TimeoutError(message)


def test_lung1_055_mcc_returns():
    image = CASE / "image.nii.gz"
    mask = CASE / "mask.nii.gz"
    if not image.is_file() or not mask.is_file():
        pytest.skip("LUNG1-055 not on disk — run scripts/fetch_lung1.py")

    params = Params(
        mode="pyradiomics",
        bin_width=25.0,
        bin_count=None,
        resampled_pixel_spacing=None,
        interpolator="sitkBSpline",
        image_types=("Original",),
        label=1,
        voxel_array_shift=0.0,
    )
    ex = RadixExtractor(mode="pyradiomics", params=params)
    img = load_volume(image)
    msk = load_volume(mask)

    with _deadline(30, "LUNG1-055 extract did not return in 30 s"):
        out = ex.extract(img, msk)

    mcc = out.values["original_glcm_MCC"]
    assert mcc == mcc  # finite: NaN fails
    assert 0.0 <= mcc <= 1.0


def test_lung1_230_mcc_matches_pyradiomics():
    image = CASE_230 / "image.nii.gz"
    mask = CASE_230 / "mask.nii.gz"
    if not image.is_file() or not mask.is_file():
        pytest.skip("LUNG1-230 not on disk — run scripts/fetch_lung1.py")
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 env not available")

    params = Params(
        mode="pyradiomics",
        bin_width=25.0,
        feature_classes={"glcm": ["MCC"]},
        label=1,
    )
    extractor = RadixExtractor(mode="pyradiomics", params=params)

    with _deadline(120, "LUNG1-230 extract did not return in 120 s"):
        got = extractor.extract(load_volume(image), load_volume(mask)).values["original_glcm_MCC"]

    worker = r"""
import json
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25, label=1)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("glcm")
result = extractor.execute(%r, %r)
print(json.dumps(float(result["original_glcm_MCC"])))
""" % (str(image), str(mask))
    completed = subprocess.run(
        [str(PYRAD), "-c", worker],
        check=True,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    reference = json.loads(completed.stdout.strip().splitlines()[-1])
    assert abs(got - reference) <= max(1e-4, 1e-4 * abs(reference))
