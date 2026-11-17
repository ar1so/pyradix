"""The ``pyradix`` command line."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd"
MASK = ROOT / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd"

pytestmark = pytest.mark.skipif(
    not IMAGE.is_file(), reason="brain1 not on disk; run scripts/fetch_validation_data.py"
)


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "pyradix", *args], capture_output=True, text=True, check=False
    )


def test_extract_json_to_stdout():
    done = _run("extract", str(IMAGE), str(MASK), "--mode", "pyradiomics")
    assert done.returncode == 0, done.stderr
    row = json.loads(done.stdout)
    from pyradix import RadixExtractor

    expected = RadixExtractor(mode="pyradiomics").extract(str(IMAGE), str(MASK)).values
    assert {k: row[k] for k in expected} == expected


def test_extract_csv_with_pyradiomics_params(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("setting:\n  binWidth: 25\n  label: 1\nimageType:\n  Original: {}\n")
    out = tmp_path / "features.csv"
    done = _run("extract", str(IMAGE), str(MASK), "--params", str(params),
                "--out", str(out), "--provenance")
    assert done.returncode == 0, done.stderr
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["provenance_mode"] == "pyradiomics"
    assert float(rows[0]["original_glcm_Contrast"]) > 0


def test_errors_are_one_clean_line():
    done = _run("extract", str(IMAGE), str(MASK), "--label", "9")
    assert done.returncode == 1
    assert done.stderr.startswith("error: ")
    assert "Traceback" not in done.stderr


def test_version():
    done = _run("--version")
    assert done.returncode == 0 and done.stdout.startswith("pyradix ")
