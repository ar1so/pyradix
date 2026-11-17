"""Shared helpers for the benchmarks.

Extraction is deterministic. The only mask subset used for scaling keeps the
voxels nearest the mask centroid, ordered by a stable argsort. No random
generator is seeded because none is used.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
# Figures are drawn locally and not committed. Set FIGURES_DIR to write them elsewhere.
FIGURES = Path(os.environ.get("FIGURES_DIR", ROOT / "benchmarks" / "figures"))
TMP = RESULTS / "tmp"

DATA = {
    "d1_digital_phantom": {
        "image": ROOT / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz",
        "mask": ROOT / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz",
        "label": "IBSI-1 digital phantom",
        "licence": "CC BY 4.0",
    },
    "d2_pat1": {
        "image": ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz",
        "mask": ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz",
        "label": "IBSI-1 lung CT PAT1",
        "licence": "CC BY-NC 3.0",
    },
    "d4_brain1": {
        "image": ROOT / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd",
        "mask": ROOT / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd",
        "label": "PyRadiomics brain1",
        "licence": "BSD-3-Clause package data",
    },
}

TOL = 1e-4
PYRAD_PYTHON = Path(os.environ.get("PYRAD_PYTHON", ROOT / ".venv-pyrad" / "bin" / "python"))
RADIX_PYTHON = Path(os.environ.get("RADIX_PYTHON", ROOT / ".venv" / "bin" / "python"))


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def cpu_model() -> str:
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        model = platform.processor()
    return model


def mem_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return kib * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def environment() -> dict:
    """Machine and library versions. The hostname is omitted on purpose."""
    radix: dict = {"import_error": None}
    try:
        import numpy
        import pyradix

        radix = {
            "version": pyradix.__version__,
            "engine_info": pyradix.engine_info(),
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
        }
    except Exception as exc:  # noqa: BLE001 — recorded, not hidden
        radix["import_error"] = f"{type(exc).__name__}: {exc}"

    pyrad = probe_pyradiomics()
    rustc = ""
    try:
        rustc = subprocess.check_output(["rustc", "--version"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        rustc = ""
    openblas = ""
    try:
        openblas = subprocess.check_output(
            ["pkg-config", "--modversion", "openblas"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        openblas = ""
    return {
        "recorded_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "git_head": git_head(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "mem_total_bytes": mem_total_bytes(),
        "rustc": rustc,
        "openblas": openblas,
        "pyradix": radix,
        "pyradiomics": pyrad,
        "random_seed": None,
        "randomness": (
            "No random generator is used. Feature extraction is deterministic. "
            "Scaling masks keep the voxels nearest the centroid."
        ),
    }


def probe_pyradiomics() -> dict:
    if not PYRAD_PYTHON.exists():
        return {"import_error": f"interpreter missing: {PYRAD_PYTHON}"}
    code = (
        "import json,numpy,SimpleITK,radiomics\n"
        "print(json.dumps({"
        "'version': str(radiomics.__version__),"
        "'python': __import__('sys').version.split()[0],"
        "'numpy': numpy.__version__,"
        "'simpleitk': str(SimpleITK.Version()).splitlines()[0]"
        "}))"
    )
    try:
        out = subprocess.check_output(
            [str(PYRAD_PYTHON), "-c", code], text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        return {"import_error": (exc.output or str(exc))[-2000:]}
    return json.loads(out.strip().splitlines()[-1])


def published_tol(value_text: str, tol_text: str) -> float:
    """Same half-unit rule as tests/ibsi/test_digital_phantom.py."""
    raw = (tol_text or "").strip()
    if raw:
        return float(raw)
    text = value_text.strip().lstrip("-")
    if "." in text and "e" not in text.lower():
        decimals = len(text.split(".", 1)[1])
        return 0.5 * (10 ** (-decimals))
    return 0.5


def read_reference_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    comments = [
        ln.strip().lstrip("#").strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.lstrip().startswith("#")
    ]
    return comments, list(csv.DictReader(lines))


def family_of(key: str) -> str:
    parts = key.split("_")
    return parts[1] if len(parts) > 2 else "other"


def compare_pair(radix: dict[str, float], other: dict[str, float], tol: float = TOL) -> dict:
    shared = sorted(set(radix) & set(other))
    rows = []
    n_pass = n_fail = 0
    worst_rel = 0.0
    worst_key = ""
    for key in shared:
        a = float(radix[key])
        b = float(other[key])
        abs_err = abs(a - b)
        rel = abs_err / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))
        limit = max(tol, abs(b) * tol)
        ok = abs_err <= limit
        n_pass += int(ok)
        n_fail += int(not ok)
        finite_rel = 0.0 if rel == float("inf") else rel
        if finite_rel >= worst_rel:
            worst_rel = finite_rel
            worst_key = key
        rows.append(
            {
                "feature": key,
                "family": family_of(key),
                "radix": a,
                "reference": b,
                "abs_err": abs_err,
                "rel_err": None if rel == float("inf") else rel,
                "pass": ok,
            }
        )
    return {
        "n_shared": len(shared),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_radix_only": len(set(radix) - set(other)),
        "n_reference_only": len(set(other) - set(radix)),
        "worst_rel_err": worst_rel,
        "worst_feature": worst_key,
        "rows": rows,
    }


def thread_env(nthreads: int) -> dict[str, str]:
    text = str(int(nthreads))
    return {
        "OPENBLAS_NUM_THREADS": text,
        "OMP_NUM_THREADS": text,
        "MKL_NUM_THREADS": text,
        "NUMEXPR_NUM_THREADS": text,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": text,
    }


def run_worker(spec: dict, nthreads: int, timeout_s: float) -> dict:
    """Run one engine in a child process pinned to the first ``nthreads`` CPUs."""
    ensure_dirs()
    spec_path = TMP / f"spec-{os.getpid()}-{spec.get('job_id', 'job')}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    python = RADIX_PYTHON if spec["engine"] == "radix" else PYRAD_PYTHON
    if not python.exists():
        return {"error": f"interpreter missing: {python}", "seconds": [], "values": {}}
    cmd = [str(python), str(ROOT / "benchmarks" / "worker.py"), str(spec_path)]
    cpus = ",".join(str(i) for i in range(int(nthreads)))
    if subprocess.call(["bash", "-lc", "command -v taskset >/dev/null"]) == 0:
        cmd = ["taskset", "-c", cpus, *cmd]
    env = os.environ.copy()
    env.update(thread_env(nthreads))
    # Keep the two interpreters from importing each other's site-packages.
    env.pop("PYTHONPATH", None)
    env.pop("CONDA_PREFIX", None)
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "error": f"timeout after {timeout_s}s",
            "seconds": [],
            "values": {},
            "stdout_tail": (exc.stdout or "")[-1500:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1500:] if isinstance(exc.stderr, str) else "",
        }
    if proc.returncode != 0:
        return {
            "error": f"exit {proc.returncode}",
            "seconds": [],
            "values": {},
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-2000:],
        }
    line = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    payload["stderr_tail"] = proc.stderr[-500:]
    return payload
