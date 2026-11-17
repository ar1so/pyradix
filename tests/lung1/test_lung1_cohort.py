"""G9: Lung1 research compare — Radix vs installed PyRadiomics 3.1.0.

Not G8. The PyRadiomics arm is the `radiomics` package, not Radix compat mode.
Protocol: docs/VALIDATION.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_lung1.py"
PAIRS = ROOT / "data" / "lung1" / "pairs.csv"
METRICS = ROOT / "tests" / "references" / "lung1" / "metrics.json"
REPORT = ROOT / "data" / "lung1" / "LUNG1_STUDY.md"


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


def _n_pairs_expected() -> int:
    raw = os.environ.get("LUNG1_LIMIT", "").strip()
    if raw:
        return int(raw)
    if not PAIRS.is_file():
        return 0
    with PAIRS.open(encoding="utf-8") as fh:
        return max(0, sum(1 for line in fh) - 1)


def _metrics_complete() -> bool:
    if not METRICS.is_file():
        return False
    try:
        report = json.loads(METRICS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = _n_pairs_expected()
    if expected <= 0:
        return False
    if report.get("cohort", {}).get("n_pairs") != expected:
        return False
    if report.get("cohort", {}).get("n_incomplete", 1) != 0:
        return False
    pyrad = report.get("engines", {}).get("pyradiomics", {})
    if pyrad.get("package") != "radiomics":
        return False
    if pyrad.get("version") != "3.1.0":
        return False
    return True


@pytest.fixture(scope="module")
def lung1_report():
    if not PAIRS.is_file():
        pytest.skip("Lung1 pairs.csv missing — run scripts/fetch_lung1.py")
    if not _pyrad_available():
        pytest.skip("installed PyRadiomics 3.1.0 env not available")
    if os.environ.get("LUNG1_SKIP_LIVE") and not _metrics_complete():
        pytest.skip("LUNG1_SKIP_LIVE set and no complete metrics")
    if not _metrics_complete():
        env = {**os.environ}
        env.pop("CONDA_PREFIX", None)
        env.setdefault("PYRAD_PYTHON", str(_pyrad_python()))
        subprocess.check_call([sys.executable, str(SCRIPT)], env=env)
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    assert report["patients"], "Lung1 report has no patients"
    return report


def test_g9_uses_installed_pyradiomics_not_radix(lung1_report):
    pyrad = lung1_report["engines"]["pyradiomics"]
    radix = lung1_report["engines"]["radix"]
    assert pyrad["package"] == "radiomics"
    assert pyrad["version"] == "3.1.0"
    assert radix["package"] == "pyradix"
    assert Path(pyrad["python"]) != Path(radix["python"])
    assert "pyradix" not in Path(pyrad["python"]).parts
    assert lung1_report["settings"]["pyradiomics_arm"] == (
        "installed radiomics 3.1.0 (not pyradix)"
    )


def test_g9_cohort_coverage(lung1_report):
    expected = _n_pairs_expected()
    cohort = lung1_report["cohort"]
    assert cohort["n_pairs"] == expected
    assert cohort["n_incomplete"] == 0
    assert cohort["n_compared"] + cohort["n_error"] == expected
    assert "LUNG1-128" not in {p["patient_id"] for p in lung1_report["patients"]}
    flagged = {p["patient_id"] for p in lung1_report["patients"] if p["missing_slices"]}
    assert {"LUNG1-014", "LUNG1-021", "LUNG1-085"} <= flagged | set()


def test_g9_feature_agreement(lung1_report):
    agr = lung1_report["agreement"]
    assert agr["n_unexpected_fail"] == 0, (
        "unexpected feature fails: %s" % agr["fail_features"]
    )
    scored = agr["n_full_pass"] + agr["n_allowed_only"]
    assert scored == lung1_report["cohort"]["n_compared"]
    assert lung1_report["cohort"]["n_compared"] >= 1
    for patient in lung1_report["patients"]:
        if patient["status"] == "error":
            continue
        assert patient["status"] in {"pass", "allowed"}
        assert patient.get("n_missing", 0) == 0
        assert patient["n_shared"] >= 100


def test_g9_timeouts_are_documented(lung1_report):
    errors = [p for p in lung1_report["patients"] if p["status"] == "error"]
    for patient in errors:
        err = patient.get("error") or ""
        assert "Timeout" in err or "timeout" in err.lower(), patient
    text = REPORT.read_text(encoding="utf-8")
    assert "timeout" in text.lower()


def test_g9_extract_times_recorded(lung1_report):
    timing = lung1_report["timing"]
    assert timing["n"] == lung1_report["cohort"]["n_compared"]
    assert timing["n"] >= 1
    assert timing["radix_median_s"] > 0
    assert timing["pyradiomics_median_s"] > 0
    assert timing["radix_sum_s"] > 0
    assert timing["pyradiomics_sum_s"] > 0
    for patient in lung1_report["patients"]:
        if patient["status"] == "error":
            continue
        assert patient["radix_s"] is not None
        assert patient["pyradiomics_s"] is not None


def test_g9_methods_report(lung1_report):
    assert REPORT.is_file()
    text = REPORT.read_text(encoding="utf-8")
    assert "installed PyRadiomics 3.1.0" in text
    assert "not" in text.lower()
    assert "RadiomicsFeatureExtractor" in text
    assert "LUNG1-128" in text
    assert lung1_report["engines"]["pyradiomics"]["version"] in text
