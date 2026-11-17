"""Requirement 8.3 and 8.5–8.7: Lung1 full signature on two grids."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.lung1.signature_cohort import (
    REPORT_PATH,
    SIGNATURE,
    agrees,
    build_report,
    tumor_present,
)

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "data" / "lung1" / "pairs.csv"
MODULE = ROOT / "tests" / "lung1" / "signature_cohort.py"


def test_tolerance_matches_requirement():
    assert agrees(0.0, 0.0)
    assert agrees(0.0, 1e-4)
    assert not agrees(0.0, 1.1e-4)
    assert agrees(10000.0, 10001.0)
    assert not agrees(10000.0, 10001.2)
    assert agrees(float("nan"), float("nan"))
    assert not agrees(float("nan"), 1.0)


def test_empty_mask_has_no_tumor():
    assert not tumor_present([[[0, 0], [0, 0]]])
    assert tumor_present([[[0, 1], [0, 0]]])


def test_report_counts_skip_fail_and_extract_error():
    pairs = [
        {"patient_id": "A", "image": "a", "mask": "a"},
        {"patient_id": "B", "image": "b", "mask": "b"},
        {"patient_id": "C", "image": "c", "mask": "c"},
        {"patient_id": "D", "image": "d", "mask": "d"},
    ]
    skips = [
        {"patient_id": "LUNG1-128", "reason": "no tumor mask"},
        {"patient_id": "A", "reason": "no tumor mask"},
    ]
    native = {
        "B": {
            "radix": {"values": {"k": 1.0}, "error": None},
            "pyradiomics": {"values": {"k": 1.0}, "error": None},
        },
        "C": {
            "radix": {"values": {"k": 5.0}, "error": None},
            "pyradiomics": {"values": {"k": 1.0}, "error": None},
        },
        "D": {
            "radix": {"values": {}, "error": "boom"},
            "pyradiomics": {"values": {"k": 1.0}, "error": None},
        },
    }
    report = build_report(pairs, skips, {"native": native}, {})
    grid = report["grids"]["native"]
    assert grid["n_pairs"] == 4
    assert grid["n_skip"] == 1
    assert grid["n_pass"] == 1
    assert grid["n_fail"] == 1
    assert grid["n_extract_error"] == 1
    assert grid["failures"][0]["patient_id"] == "C"
    assert grid["failures"][0]["feature"] == "k"
    assert grid["failures"][0]["radix"] == 5.0
    assert grid["failures"][0]["reference"] == 1.0
    assert report["skips"][0] == {"patient_id": "LUNG1-128", "reason": "no tumor mask"}


def _pyrad_python() -> Path | None:
    raw = os.environ.get("PYRAD_PYTHON", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def _pyrad_available() -> bool:
    python = _pyrad_python()
    if python is None:
        return False
    try:
        out = subprocess.check_output(
            [str(python), "-c", "import radiomics; print(radiomics.__version__)"],
            text=True, encoding="utf-8", errors="replace",
            timeout=90,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return "3.1.0" in out


def _problem_lines(report: dict) -> list[str]:
    lines = []
    for grid, block in report["grids"].items():
        for failure in block["failures"]:
            lines.append(
                "%s %s %s: radix %s reference %s"
                % (
                    failure["patient_id"],
                    grid,
                    failure["feature"],
                    failure["radix"],
                    failure["reference"],
                )
            )
        for error in block["extract_errors"]:
            lines.append("%s %s extract error: %s" % (error["patient_id"], grid, error["error"]))
    return lines


def test_lung1_signature_cohort():
    if not PAIRS.is_file():
        pytest.skip("Lung1 pairs.csv missing — run scripts/fetch_lung1.py")
    if not _pyrad_available():
        pytest.skip("installed PyRadiomics 3.1.0 env not available")
    env = {**os.environ, "PYRAD_PYTHON": str(_pyrad_python())}
    env.pop("CONDA_PREFIX", None)
    completed = subprocess.run([sys.executable, str(MODULE)], env=env, check=False)
    assert completed.returncode == 0, "signature cohort runner failed"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["settings"]["mode"] == "pyradiomics"
    assert report["settings"]["bin_width"] == 25.0
    assert report["settings"]["image_types"] == list(SIGNATURE)
    assert report["engines"]["pyradiomics"]["package"] == "radiomics"
    assert report["engines"]["pyradiomics"]["version"] == "v3.1.0"
    assert report["engines"]["radix"]["package"] == "pyradix"
    pyrad_python = Path(report["engines"]["pyradiomics"]["python"])
    radix_python = Path(report["engines"]["radix"]["python"])
    assert pyrad_python != radix_python
    recorded = {row["patient_id"] for row in report["skips"]}
    assert "LUNG1-128" in recorded
    assert all(row["reason"] == "no tumor mask" for row in report["skips"])
    assert set(report["grids"]) == {"native", "resample_2mm"}
    for grid, block in report["grids"].items():
        assert block["n_pairs"] == 421, grid
        accounted = block["n_pass"] + block["n_fail"] + block["n_extract_error"] + block["n_skip"]
        assert accounted == 421, grid
    assert not _problem_lines(report), "\n".join(_problem_lines(report))
