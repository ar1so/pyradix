"""G8: full live batch vs installed PyRadiomics 3.1.0 (docs/VALIDATION.md)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_pyradiomics.py"

sys.path.insert(0, str(ROOT / "scripts"))
from compare_pyradiomics import (  # noqa: E402
    DATASETS,
    KNOWN_GAP_JOBS,
    build_report,
    job_ids,
    parse_job,
)


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


def _data_ready() -> bool:
    return all(spec["image"].exists() and spec["mask"].exists() for spec in DATASETS.values())


def _spawn(python: str, engine: str, jobs: list[str]) -> dict:
    env = {**os.environ, "COMPARE_ENGINE": engine, "COMPARE_JOBS": ",".join(jobs)}
    if engine == "radix":
        env.pop("CONDA_PREFIX", None)
    payload = subprocess.check_output([python, str(SCRIPT)], env=env, timeout=900)
    return json.loads(payload)


@pytest.fixture(scope="module")
def batch_report():
    if not _pyrad_available():
        pytest.skip("PyRadiomics 3.1.0 env not available")
    if not _data_ready():
        pytest.skip("validation volumes not fetched")
    jobs = job_ids()
    radix_blob = _spawn(sys.executable, "radix", jobs)
    pyrad_blob = _spawn(str(_pyrad_python()), "pyradiomics", jobs)
    report, _rows = build_report(radix_blob, pyrad_blob, jobs)
    assert report["jobs"], "batch produced no jobs"
    return report


def test_g8_batch_job_count(batch_report):
    assert set(batch_report["jobs"]) == set(job_ids())


@pytest.mark.parametrize("job_id", job_ids())
def test_g8_job_features_and_speed(batch_report, job_id):
    block = batch_report["jobs"][job_id]
    feats = block["features"]
    assert feats["n_missing"] == 0, f"{job_id}: radix missing {feats['n_missing']} default keys"
    assert feats["n_shared"] >= 100
    assert block["radix_median_s"] > 0
    assert block["pyradiomics_median_s"] > 0
    if job_id in KNOWN_GAP_JOBS:
        assert feats["n_fail"] > 0, f"{job_id} is a documented gap and should still miss"
        return
    assert feats["n_fail"] == 0, (
        f"{job_id}: {feats['n_fail']} unexpected fail(s), "
        f"worst {feats['worst_feature']} rel={feats['worst_rel_err']}"
    )
    dataset, _config, _cfg = parse_job(job_id)
    if dataset == "d2_pat1":
        assert feats["n_allowed"] >= 1, "D2 native jobs must record MeshVolume as allowed"


def test_g8_d4_bw25_full_match(batch_report):
    feats = batch_report["jobs"]["d4_brain1__bw25"]["features"]
    assert feats["n_pass"] == feats["n_shared"]
    assert feats["n_allowed"] == 0
    assert feats["n_pyradiomics"] == 107
