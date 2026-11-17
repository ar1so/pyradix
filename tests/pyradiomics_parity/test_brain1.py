"""M3/G5: pyradiomics mode vs frozen PyRadiomics 3.1.0 on D4 brain1."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pyradix import RadixExtractor

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "tests" / "references" / "pyradiomics_3.1.0_brain1.csv"
YAML_PATH = ROOT / "examples" / "pyradiomics.yaml"
# VALIDATION.md G5: within 1e-4 of frozen 3.1.0 (abs or relative, whichever is looser).
TOL = 1e-4


def _expected() -> dict[str, float]:
    with CSV_PATH.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return {row["radix_key"]: float(row["value"]) for row in csv.DictReader(lines)}


def test_m3_frozen_csv_exists():
    assert CSV_PATH.is_file(), "M3: freeze CSV missing"
    expected = _expected()
    assert len(expected) == 107
    assert all(k.startswith("original_") for k in expected)


def test_g5_brain1_pyradiomics_parity(d4_image, d4_mask):
    if not Path(d4_image).exists():
        pytest.skip("D4 brain1 not fetched")
    ex = RadixExtractor.from_pyradiomics_yaml(YAML_PATH)
    got = ex.extract(d4_image, d4_mask).values
    expected = _expected()
    failures: list[str] = []
    for key, exp in expected.items():
        if key not in got:
            failures.append(f"missing {key}")
            continue
        val = got[key]
        tol = max(TOL, abs(exp) * TOL)
        if abs(val - exp) > tol:
            rel = abs(val - exp) / abs(exp) if exp != 0 else float("inf")
            failures.append(
                f"{key}: got {val:.8g} exp {exp:.8g} abs={abs(val - exp):.3g} rel={rel:.3g}"
            )
    assert not failures, f"{len(failures)} fail(s)\n" + "\n".join(failures)
