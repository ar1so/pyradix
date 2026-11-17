"""Full PyRadiomics 3.1.0 signature on the phantom, brain1, and PAT1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests" / "pyradiomics_parity" / "full_signature_job.py"
_raw_pyrad = os.environ.get("PYRAD_PYTHON", "").strip()
PYRAD = Path(_raw_pyrad) if _raw_pyrad else None
TOL = 1e-4

VOLUMES = {
    "digital_phantom": (
        ROOT / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz",
        ROOT / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz",
    ),
    "brain1": (
        ROOT / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd",
        ROOT / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd",
    ),
    "PAT1": (
        ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz",
        ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz",
    ),
}


def _ready(volume: str) -> None:
    if PYRAD is None or not PYRAD.is_file():
        pytest.skip("PyRadiomics 3.1.0 is not available")
    image, mask = VOLUMES[volume]
    if not image.is_file() or not mask.is_file():
        pytest.skip(f"{volume} is not fetched")


def _run(engine: str, volume: str, grid: str) -> dict[str, float]:
    image, mask = VOLUMES[volume]
    env = {
        **os.environ,
        "COMPARE_ENGINE": engine,
        "COMPARE_IMAGE": str(image),
        "COMPARE_MASK": str(mask),
        "COMPARE_GRID": grid,
    }
    if engine == "radix":
        env.pop("CONDA_PREFIX", None)
        python = sys.executable
    else:
        python = str(PYRAD)
    completed = subprocess.run(
        [python, str(JOB)],
        check=False,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        timeout=1800,
    )
    if completed.returncode != 0:
        pytest.fail(f"{volume} {grid} {engine} extract failed\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


def _misses(volume: str, grid: str, got: dict[str, float], reference: dict[str, float]) -> list[str]:
    misses = []
    for key, expected in reference.items():
        if key not in got:
            misses.append(f"{volume} {grid} {key}: missing from Radix, reference {expected}")
            continue
        actual = got[key]
        if abs(actual - expected) > max(TOL, TOL * abs(expected)):
            misses.append(f"{volume} {grid} {key}: radix {actual} reference {expected}")
    return misses


def _compare(volume: str, grid: str) -> None:
    _ready(volume)
    reference = _run("pyradiomics", volume, grid)
    got = _run("radix", volume, grid)
    assert reference, f"{volume} {grid} reference returned no features"
    misses = _misses(volume, grid, got, reference)
    assert not misses, "\n".join(misses)


@pytest.mark.parametrize("volume", ["digital_phantom", "brain1", "PAT1"])
def test_full_signature_native(volume: str):
    _compare(volume, "native")


def test_brain1_resampled_2mm_original():
    _compare("brain1", "resample_2mm")
