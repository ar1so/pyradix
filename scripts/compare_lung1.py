#!/usr/bin/env python3
"""Lung1 research compare: Radix vs installed PyRadiomics 3.1.0.

Not G8 (phantoms). Protocol: docs/VALIDATION.md

The PyRadiomics arm is the real `radiomics` package (v3.1.0), started from
the interpreter in PYRAD_PYTHON. It is not Radix `mode=pyradiomics`.

  .venv/bin/maturin develop --release
  .venv/bin/python scripts/compare_lung1.py
  .venv/bin/python scripts/compare_lung1.py --limit 5
  .venv/bin/python scripts/compare_lung1.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from interpreters import require_pyrad_python

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_pyradiomics import (  # noqa: E402
    TOL,
    _median,
    compare_features,
    run_pyradiomics_job,
    run_radix_job,
)

PAIRS_CSV = ROOT / "data" / "lung1" / "pairs.csv"
CLINICAL_CSV = ROOT / "data" / "lung1" / "Lung1.clinical.csv"
OUT_DIR = ROOT / "tests" / "references" / "lung1"
CHECKPOINT_RADIX = OUT_DIR / "radix.jsonl"
CHECKPOINT_PYRAD = OUT_DIR / "pyradiomics.jsonl"
OUT_JSON = OUT_DIR / "metrics.json"
OUT_CSV = OUT_DIR / "features.csv"
OUT_REPORT = ROOT / "data" / "lung1" / "LUNG1_STUDY.md"

DEFAULT_TIMEOUT_S = 90.0
WORKER_READY_S = 180.0

CFG = {
    "bin_width": 25.0,
    "bin_count": None,
    "resample": None,
}

ALLOWED_FAIL = {
    "original_shape_MeshVolume": (
        "Lorensen 256-cube vs PyRadiomics cshape 128-cube "
        "(same D2 / DEVIATIONS.md allowance; lung CT cohort)"
    ),
}

MISSING_SLICES = frozenset({"LUNG1-014", "LUNG1-021", "LUNG1-085"})
SKIP_NO_GTV = frozenset({"LUNG1-128"})


def _load_pairs(limit: int | None) -> list[dict]:
    rows = []
    with PAIRS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row["patient_id"].strip()
            if pid in SKIP_NO_GTV:
                continue
            image = ROOT / row["image"]
            mask = ROOT / row["mask"]
            rows.append(
                {
                    "patient_id": pid,
                    "image": str(image),
                    "mask": str(mask),
                    "missing_slices": row.get("missing_slices", "").lower()
                    in {"1", "true", "yes"},
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_clinical() -> dict[str, dict]:
    if not CLINICAL_CSV.is_file():
        return {}
    with CLINICAL_CSV.open(encoding="utf-8", newline="") as fh:
        return {row["PatientID"]: row for row in csv.DictReader(fh)}


def _read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["patient_id"]] = rec
    return out


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _engine_versions() -> dict:
    if os.environ.get("LUNG1_ENGINE") == "pyradiomics":
        import radiomics

        return {
            "package": "radiomics",
            "version": str(radiomics.__version__),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
        }
    import pyradix

    info = {}
    if hasattr(pyradix, "engine_info"):
        try:
            info = dict(pyradix.engine_info())
        except Exception:
            info = {}
    return {
        "package": "pyradix",
        "version": str(getattr(pyradix, "__version__", "0.1.0")),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "engine_info": info,
    }


def _patient_timeout() -> float:
    raw = os.environ.get("LUNG1_TIMEOUT", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_TIMEOUT_S


def _readline_timeout(proc: subprocess.Popen, timeout: float) -> str | None:
    q: queue.Queue[str | None] = queue.Queue()

    def _read() -> None:
        try:
            line = proc.stdout.readline() if proc.stdout is not None else ""
        except Exception:
            line = ""
        q.put(line)

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    line = q.get_nowait()
    if not line:
        return None
    return line


def _roi_policy_skip(error: str | None) -> bool:
    if not error:
        return False
    text = error.lower()
    return (
        "roi size" in text
        or "minimumroidimensions" in text
        or "minimumroisize" in text
    )


def _empty_rec(pid: str, engine: str, versions: dict, error: str | None) -> dict:
    return {
        "patient_id": pid,
        "engine": engine,
        "package": versions.get("package"),
        "version": versions.get("version"),
        "error": error,
        "values": {},
        "seconds": None,
        "roi_voxels": None,
        "shape": None,
    }


def _worker_main() -> int:
    engine = os.environ["LUNG1_ENGINE"]
    runner = run_radix_job if engine == "radix" else run_pyradiomics_job
    versions = _engine_versions()
    print(json.dumps({"ready": True, "versions": versions}), flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        pair = json.loads(line)
        pid = pair["patient_id"]
        rec = _empty_rec(pid, engine, versions, None)
        try:
            image = Path(pair["image"])
            mask = Path(pair["mask"])
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError("missing NIfTI for %s" % pid)
            out = runner(image, mask, CFG, repeats=1)
            rec["values"] = out["values"]
            rec["seconds"] = float(out["seconds"][0])
            rec["roi_voxels"] = out.get("roi_voxels")
            rec["shape"] = out.get("shape")
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        print(json.dumps(rec), flush=True)
    return 0


def _spawn_worker(python: str, engine: str) -> tuple[subprocess.Popen, dict]:
    env = {**os.environ, "LUNG1_ENGINE": engine, "LUNG1_WORKER": "1"}
    if engine == "radix":
        env.pop("CONDA_PREFIX", None)
    proc = subprocess.Popen(
        [python, str(Path(__file__))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    ready_line = _readline_timeout(proc, WORKER_READY_S)
    if ready_line is None:
        proc.kill()
        raise RuntimeError("%s worker did not become ready in %.0fs" % (engine, WORKER_READY_S))
    ready = json.loads(ready_line)
    if not ready.get("ready"):
        proc.kill()
        raise RuntimeError("%s worker ready payload invalid: %s" % (engine, ready_line[:200]))
    return proc, ready.get("versions") or _engine_versions()


def _stop_worker(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


def _run_engine_supervised(
    python: str, engine: str, pairs: list[dict], checkpoint: Path
) -> dict:
    timeout = _patient_timeout()
    done = _read_jsonl(checkpoint)
    remaining = [
        p for p in pairs if not (done.get(p["patient_id"]) or {}).get("values")
    ]
    print(
        "%s via %s  already %d/%d  timeout %.0fs"
        % (engine, python, len(pairs) - len(remaining), len(pairs), timeout),
        file=sys.stderr,
    )
    if not remaining:
        sample = next(iter(done.values())) if done else {}
        return {
            "package": sample.get("package"),
            "version": sample.get("version"),
        }
    proc, versions = _spawn_worker(python, engine)
    index = {p["patient_id"]: i for i, p in enumerate(pairs, start=1)}
    try:
        for pair in remaining:
            pid = pair["patient_id"]
            print(
                "== %s %d/%d %s ==" % (engine, index[pid], len(pairs), pid),
                file=sys.stderr,
            )
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(pair) + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                _stop_worker(proc)
                proc, versions = _spawn_worker(python, engine)
                proc.stdin.write(json.dumps(pair) + "\n")
                proc.stdin.flush()
            line = _readline_timeout(proc, timeout)
            if line is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                rec = _empty_rec(
                    pid,
                    engine,
                    versions,
                    "Timeout: extract exceeded %.0fs" % timeout,
                )
                print("   TIMEOUT %.0fs" % timeout, file=sys.stderr)
                _append_jsonl(checkpoint, rec)
                proc, versions = _spawn_worker(python, engine)
                continue
            rec = json.loads(line)
            if rec.get("error"):
                print("   ERROR %s" % rec["error"], file=sys.stderr)
            else:
                print(
                    "   %.3fs  n=%d"
                    % (rec.get("seconds") or 0.0, len(rec.get("values") or {})),
                    file=sys.stderr,
                )
            _append_jsonl(checkpoint, rec)
    finally:
        _stop_worker(proc)
    return versions


def _truncate_stale(checkpoint: Path, keep: set[str]) -> None:
    recs = _read_jsonl(checkpoint)
    if not recs:
        return
    if set(recs).issubset(keep):
        return
    tmp = checkpoint.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for pid in keep:
            if pid in recs:
                fh.write(json.dumps(recs[pid]) + "\n")
    tmp.replace(checkpoint)


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return "%.1f%%" % (100.0 * n / d)


def _count_table(counter: Counter, n: int) -> list[tuple[str, int, str]]:
    rows = []
    for key, k in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        label = key if key else "(missing)"
        rows.append((label, k, _pct(k, n)))
    return rows


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    pairs: list[dict],
    radix_blob: dict[str, dict],
    pyrad_blob: dict[str, dict],
    radix_meta: dict,
    pyrad_meta: dict,
) -> tuple[dict, list[dict]]:
    clinical = _load_clinical()
    patients = []
    feature_rows = []
    n_ok = n_mesh_only = n_unexpected = n_error = n_incomplete = n_roi_skip = 0
    fail_features: Counter = Counter()
    family_fail: Counter = Counter()
    radix_times = []
    pyrad_times = []
    roi = []
    compared = 0

    for pair in pairs:
        pid = pair["patient_id"]
        clin = clinical.get(pid, {})
        r = radix_blob.get(pid)
        p = pyrad_blob.get(pid)
        block = {
            "patient_id": pid,
            "missing_slices": pair["missing_slices"] or pid in MISSING_SLICES,
            "age": clin.get("age"),
            "overall_stage": clin.get("Overall.Stage"),
            "histology": clin.get("Histology"),
            "gender": clin.get("gender"),
            "survival_days": clin.get("Survival.time"),
            "deadstatus_event": clin.get("deadstatus.event"),
            "radix_s": None,
            "pyradiomics_s": None,
            "roi_voxels": None,
            "n_pass": 0,
            "n_fail": 0,
            "n_allowed": 0,
            "n_shared": 0,
            "worst_feature": "",
            "worst_rel_err": 0.0,
            "status": "incomplete",
            "error": None,
        }
        r_err = (r or {}).get("error")
        p_err = (p or {}).get("error")
        if r is None or p is None:
            n_incomplete += 1
            block["error"] = "missing engine output"
            patients.append(block)
            continue
        if r_err or p_err:
            if _roi_policy_skip(r_err) or _roi_policy_skip(p_err):
                n_roi_skip += 1
                block["status"] = "skip"
                block["error"] = "radix=%s; pyradiomics=%s" % (r_err, p_err)
                patients.append(block)
                continue
            n_error += 1
            block["status"] = "error"
            block["error"] = "radix=%s; pyradiomics=%s" % (r_err, p_err)
            patients.append(block)
            continue
        cmp_ = compare_features(r.get("values") or {}, p.get("values") or {}, "lung1")
        # MeshVolume is allowed on this cohort (same marching-cubes gap as D2).
        for row in cmp_["rows"]:
            if row["feature"] == "original_shape_MeshVolume" and row["reason"] == "tol":
                row["reason"] = "allowed"
                row["pass"] = True
                cmp_["n_fail"] = max(0, cmp_["n_fail"] - 1)
                cmp_["n_allowed"] += 1
                fam = cmp_["families"].get("shape")
                if fam:
                    fam["fail"] = max(0, fam["fail"] - 1)
                    fam["allowed"] = fam.get("allowed", 0) + 1
        r_s = r.get("seconds")
        p_s = p.get("seconds")
        block.update(
            {
                "radix_s": r_s,
                "pyradiomics_s": p_s,
                "roi_voxels": r.get("roi_voxels"),
                "n_pass": cmp_["n_pass"],
                "n_fail": cmp_["n_fail"],
                "n_allowed": cmp_["n_allowed"],
                "n_shared": cmp_["n_shared"],
                "worst_feature": cmp_["worst_feature"],
                "worst_rel_err": cmp_["worst_rel_err"],
                "n_missing": cmp_["n_missing"],
                "n_radix_extra": cmp_["n_radix_extra"],
            }
        )
        if r_s is not None:
            radix_times.append(float(r_s))
        if p_s is not None:
            pyrad_times.append(float(p_s))
        if r.get("roi_voxels"):
            roi.append(int(r["roi_voxels"]))
        compared += 1
        if cmp_["n_fail"] == 0 and cmp_["n_missing"] == 0:
            if cmp_["n_allowed"]:
                n_mesh_only += 1
                block["status"] = "allowed"
            else:
                n_ok += 1
                block["status"] = "pass"
        else:
            n_unexpected += 1
            block["status"] = "fail"
            for row in cmp_["rows"]:
                if row["reason"] in {"tol", "missing"}:
                    fail_features[row["feature"]] += 1
                    family_fail[row["family"]] += 1
        for row in cmp_["rows"]:
            feature_rows.append({"patient_id": pid, **row})
        patients.append(block)

    r_sum = sum(radix_times)
    p_sum = sum(pyrad_times)
    r_med = _median(radix_times)
    p_med = _median(pyrad_times)
    stages = Counter(p.get("overall_stage") or "" for p in patients)
    hist = Counter(p.get("histology") or "" for p in patients)
    gender = Counter(p.get("gender") or "" for p in patients)
    n_flagged = sum(1 for p in patients if p["missing_slices"])

    report = {
        "protocol": "docs/VALIDATION.md",
        "study": {
            "name": "Lung1 / NSCLC-Radiomics",
            "citation": "Aerts et al. Nat Commun 5:4006 (2014)",
            "doi_paper": "10.1038/ncomms5006",
            "doi_data": "10.7937/K9/TCIA.2015.PF0M9REI",
            "license": "CC BY-NC 3.0",
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {
            "bin_width": 25.0,
            "bin_count": None,
            "resample": None,
            "interpolator": "sitkBSpline",
            "label": 1,
            "imageType": "Original",
            "voxelArrayShift": 0,
            "tol": TOL,
            "timing": "extract-only after load",
            "patient_timeout_s": _patient_timeout(),
            "allowed_fail": ALLOWED_FAIL,
            "radix_mode": "pyradiomics",
            "pyradiomics_arm": "installed radiomics 3.1.0 (not pyradix)",
        },
        "engines": {
            "radix": {
                "python": "project virtualenv",
                "package": radix_meta.get("package", "pyradix"),
                "version": radix_meta.get("version"),
                "python_version": radix_meta.get("python_version"),
                "engine_info": radix_meta.get("engine_info"),
            },
            "pyradiomics": {
                "python": "PYRAD_PYTHON",
                "package": pyrad_meta.get("package", "radiomics"),
                "version": pyrad_meta.get("version"),
                "python_version": pyrad_meta.get("python_version"),
            },
        },
        "cohort": {
            "n_clinical": len(clinical),
            "n_pairs": len(pairs),
            "n_compared": compared,
            "n_error": n_error,
            "n_roi_skip": n_roi_skip,
            "n_incomplete": n_incomplete,
            "n_missing_slices_flag": n_flagged,
            "skip_no_gtv": sorted(SKIP_NO_GTV),
            "stages": dict(stages),
            "histology": dict(hist),
            "gender": dict(gender),
        },
        "agreement": {
            "n_full_pass": n_ok,
            "n_allowed_only": n_mesh_only,
            "n_unexpected_fail": n_unexpected,
            "fail_features": dict(fail_features.most_common(30)),
            "family_fail": dict(family_fail),
        },
        "timing": {
            "n": len(radix_times),
            "radix_sum_s": r_sum,
            "pyradiomics_sum_s": p_sum,
            "radix_median_s": r_med,
            "pyradiomics_median_s": p_med,
            "speedup_median": (p_med / r_med) if r_med else None,
            "speedup_sum": (p_sum / r_sum) if r_sum else None,
            "roi_voxels_median": _median([float(x) for x in roi]) if roi else None,
            "roi_voxels_min": min(roi) if roi else None,
            "roi_voxels_max": max(roi) if roi else None,
        },
        "patients": patients,
    }
    return report, feature_rows


def write_outputs(report: dict, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "patient_id",
                "family",
                "feature",
                "radix",
                "pyradiomics",
                "abs_err",
                "rel_err",
                "pass",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    OUT_REPORT.write_text(render_methods_md(report), encoding="utf-8")
    print("wrote %s" % OUT_JSON, file=sys.stderr)
    print("wrote %s" % OUT_CSV, file=sys.stderr)
    print("wrote %s" % OUT_REPORT, file=sys.stderr)


def render_methods_md(report: dict) -> str:
    study = report["study"]
    settings = report["settings"]
    engines = report["engines"]
    cohort = report["cohort"]
    agr = report["agreement"]
    timing = report["timing"]
    n_pairs = cohort["n_pairs"]
    n_cmp = cohort["n_compared"]
    n_clin = cohort["n_clinical"]
    patients = report["patients"]
    scored = agr["n_full_pass"] + agr["n_allowed_only"] + agr["n_unexpected_fail"]

    stage_rows = [
        [k, str(v), _pct(v, n_pairs)]
        for k, v, _p in [
            (a, b, c) for a, b, c in _count_table(Counter(cohort["stages"]), n_pairs)
        ]
    ]
    # _count_table already returns pct strings
    stage_rows = [[a, str(b), c] for a, b, c in _count_table(Counter(cohort["stages"]), n_pairs)]
    hist_rows = [[a, str(b), c] for a, b, c in _count_table(Counter(cohort["histology"]), n_pairs)]
    gender_rows = [[a, str(b), c] for a, b, c in _count_table(Counter(cohort["gender"]), n_pairs)]

    fail_rows = []
    for feat, n in list(agr["fail_features"].items())[:15]:
        fail_rows.append([feat, str(n), _pct(n, max(n_cmp, 1))])
    if not fail_rows:
        fail_rows = [["(none)", "0", "0%"]]

    family_rows = []
    for fam, n in sorted(agr["family_fail"].items(), key=lambda kv: (-kv[1], kv[0])):
        family_rows.append([fam, str(n), "patient-feature fails"])
    if not family_rows:
        family_rows = [["(none)", "0", "—"]]

    errors = [p for p in patients if p["status"] == "error"]
    error_rows = (
        [[p["patient_id"], (p.get("error") or "")[:120]] for p in errors[:20]]
        if errors
        else [["(none)", "—"]]
    )

    slow = sorted(
        [p for p in patients if p.get("pyradiomics_s") is not None],
        key=lambda p: float(p["pyradiomics_s"]),
        reverse=True,
    )[:8]
    slow_rows = [
        [
            p["patient_id"],
            str(p.get("roi_voxels") or "—"),
            "%.3f" % p["radix_s"] if p.get("radix_s") is not None else "—",
            "%.3f" % p["pyradiomics_s"],
        ]
        for p in slow
    ]
    if not slow_rows:
        slow_rows = [["(none)", "—", "—", "—"]]

    speedup = timing.get("speedup_median")
    speedup_sum = timing.get("speedup_sum")
    speedup_s = "%.2f×" % speedup if speedup else "n/a"
    speedup_sum_s = "%.2f×" % speedup_sum if speedup_sum else "n/a"

    radix_med = timing.get("radix_median_s")
    pyrad_med = timing.get("pyradiomics_median_s")

    status_rows = [
        ["full match at 1e-4", str(agr["n_full_pass"]), _pct(agr["n_full_pass"], scored or 1)],
        [
            "MeshVolume allowed only",
            str(agr["n_allowed_only"]),
            _pct(agr["n_allowed_only"], scored or 1),
        ],
        [
            "unexpected feature fail",
            str(agr["n_unexpected_fail"]),
            _pct(agr["n_unexpected_fail"], scored or 1),
        ],
        ["extract error", str(cohort["n_error"]), _pct(cohort["n_error"], n_pairs)],
        ["ROI skipped", str(cohort.get("n_roi_skip", 0)), _pct(cohort.get("n_roi_skip", 0), n_pairs)],
        ["incomplete", str(cohort["n_incomplete"]), _pct(cohort["n_incomplete"], n_pairs)],
    ]

    return r"""# Lung1 extractor study — methods and results

Research-method compare of **Radix** and **installed PyRadiomics 3.1.0** on the Aerts 2014 Lung1 cohort (D5). This is **not** gate G8 (phantom / brain1 batch) and **not** Radix `mode=pyradiomics` standing in for PyRadiomics.

Generated: `{generated_at}`

## 1. Objective

Measure, on a published NSCLC radiomics cohort:

1. **Feature agreement** — default Original `original_*` values from Radix vs the real PyRadiomics 3.1.0 library, tolerance \(10^{{-4}}\) (absolute or relative, whichever is looser).
2. **Extract speed** — wall time of `RadixExtractor.extract` vs `RadiomicsFeatureExtractor.execute` after both volumes are loaded.

No survival model, signature replay, or clinical claim is made here. Clinical columns are used only to describe the cohort.

## 2. Cohort

| Item | Value |
|---|---|
| Study | {citation} |
| Paper DOI | [{doi_paper}](https://doi.org/{doi_paper}) |
| Data DOI | [{doi_data}](https://doi.org/{doi_data}) |
| License | {license} (non-commercial) |
| Collection | TCIA NSCLC-Radiomics v4 (Lung1, MAASTRO) |
| Clinical rows | {n_clinical} |
| Image+GTV pairs | {n_pairs} |
| Compared (both engines) | {n_compared} |
| No GTV (excluded) | LUNG1-128 (post-operative) |
| Missing CT slices (flagged, kept) | LUNG1-014, LUNG1-021, LUNG1-085 |

Pretreatment CT and GTV-1 from DICOM SEG, converted to NIfTI (`data/lung1/<ID>/{{image,mask}}.nii.gz`). Version-2 RTSTRUCT is not used.

### Stage (extracted pairs)

{stage_table}

### Histology

{hist_table}

### Sex

{gender_table}

## 3. Extractors

Two **separate Python processes**. The PyRadiomics arm never imports `pyradix`.

| Arm | Process | Package | Version | Role |
|---|---|---|---|---|
| Radix | `{radix_python}` | `{radix_package}` | {radix_version} | New extractor. `RadixExtractor(mode="pyradiomics")` so binning/shape policies match the literature package. This is still Radix, not a PyRadiomics wrapper. |
| PyRadiomics | `{pyrad_python}` | `{pyrad_package}` | {pyrad_version} | Installed [PyRadiomics](https://pyradiomics.readthedocs.io/) 3.1.0 (`import radiomics`). |

Python versions: Radix {radix_py_ver}; PyRadiomics {pyrad_py_ver}.

## 4. Image biomarker settings (matched)

Copied from the G8 live protocol (the live compare in VALIDATION.md (G8)) so the cohort test is the same experiment on different volumes:

| Setting | Value |
|---|---|
| Discretization | FBS `binWidth=25` (PyRadiomics default) |
| Resample | none (native CT grid) |
| Interpolator | `sitkBSpline` (unused unless resample is set) |
| Label | 1 (GTV-1) |
| Image type | **Original only** (no LoG / wavelet feature vectors) |
| `voxelArrayShift` | 0 |
| Clock | extract-only after load (`time.perf_counter`) |
| Per-patient timeout | {timeout_s:.0f} s (worker is killed and the case is recorded as `Timeout`; the engine process is restarted so one GLCM-MCC hang cannot stall the cohort) |
| Tolerance | \(\lvert a-b\rvert \le \max(10^{{-4}},\,10^{{-4}}\lvert b\rvert)\) |

Radix extras (deprecated names) are not scored. The compared set is whatever 3.1.0 emits as `original_*`.

### Allowed miss

`original_shape_MeshVolume` — Lorensen 256-cube (Radix) vs PyRadiomics `cshape.c` 128-cube. Same documented D2 gap ([DEVIATIONS.md](DEVIATIONS.md)). Rel. error is typically just over \(10^{{-4}}\).

## 5. Endpoints

| Endpoint | Pass if |
|---|---|
| Coverage | every GTV pair attempted on both engines |
| Feature agreement | `n_fail == 0` after the MeshVolume allowance; no missing 3.1.0 keys |
| Speed | both extract times recorded (no claim that one is faster until measured) |
| Provenance | PyRadiomics arm `package=radiomics` and version `3.1.0` |

Gate **G9**: `tests/lung1/test_lung1_cohort.py`.

## 6. Results

Release Radix vs installed PyRadiomics {pyrad_version}. Extract-only after load. n = {n_compared} compared cases of {n_pairs} pairs.

### Agreement

{status_table}

### Timing (median / sum over compared cases)

| Engine | Median extract | Sum extract | ROI voxels (median, min–max) |
|---|---|---|---|
| Radix | {radix_med} | {radix_sum} | {roi_med} ({roi_min}–{roi_max}) |
| PyRadiomics 3.1.0 | {pyrad_med} | {pyrad_sum} | same masks |
| Ratio (3.1.0 / Radix) | {speedup} | {speedup_sum} | — |

### Unexpected failing features (top)

{fail_table}

### Family fail counts (patient × feature)

{family_table}

### Slowest cases (by PyRadiomics extract time)

{slow_table}

### Extract errors

{error_table}

## 7. How to reproduce

```bash
.venv/bin/maturin develop --release
.venv/bin/python scripts/compare_lung1.py
pytest tests/lung1/test_lung1_cohort.py
```

Subset: `.venv/bin/python scripts/compare_lung1.py --limit 10`

Writes:

- `tests/references/lung1/metrics.json` — per-patient pass/fail + times
- `tests/references/lung1/features.csv` — long table, one row per patient × feature
- this file

Checkpoints (`radix.jsonl`, `pyradiomics.jsonl`) resume a killed run.

## 8. Limitations

- Original image type only. Aerts 2014 also used LoG and wavelet features; those vectors are out of scope until Radix emits them from `extract()`.
- Native CT grid, no 3 mm isotropic resample from the 2014 pipeline. This is a software-agreement study, not a signature replay.
- License is CC BY-NC 3.0 — non-commercial.
- Speed is extract-only after load; DICOM→NIfTI conversion is excluded.
- Three series are missing slices (not through the GTV). They stay in the table and are flagged.
- A case that exceeds the per-patient timeout is an extract error, not a feature-value fail. Timeouts are usually GLCM MCC on a large gray-level set (same reason D2 `binWidth=5` is out of G8).

## 9. Citations

1. Aerts, H. J. W. L. et al. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. *Nat Commun* **5**, 4006 (2014). https://doi.org/10.1038/ncomms5006
2. Aerts et al. Data From NSCLC-Radiomics (version 4). TCIA. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
3. van Griethuysen et al. Computational radiomics system to decode the radiographic phenotype. *Cancer Research* (2017). PyRadiomics 3.1.0.
""".format(
        generated_at=report["generated_at"],
        citation=study["citation"],
        doi_paper=study["doi_paper"],
        doi_data=study["doi_data"],
        license=study["license"],
        n_clinical=n_clin,
        n_pairs=n_pairs,
        n_compared=n_cmp,
        stage_table=_md_table(["Overall.Stage", "n", "%"], stage_rows),
        hist_table=_md_table(["Histology", "n", "%"], hist_rows),
        gender_table=_md_table(["Sex", "n", "%"], gender_rows),
        radix_python=engines["radix"]["python"],
        radix_package=engines["radix"]["package"],
        radix_version=engines["radix"]["version"],
        timeout_s=settings.get("patient_timeout_s") or DEFAULT_TIMEOUT_S,
        radix_py_ver=engines["radix"].get("python_version") or "?",
        pyrad_python=engines["pyradiomics"]["python"],
        pyrad_package=engines["pyradiomics"]["package"],
        pyrad_version=engines["pyradiomics"]["version"],
        pyrad_py_ver=engines["pyradiomics"].get("python_version") or "?",
        status_table=_md_table(["Outcome", "n", "% of scored"], status_rows),
        radix_med="%.3f s" % radix_med if radix_med is not None else "n/a",
        pyrad_med="%.3f s" % pyrad_med if pyrad_med is not None else "n/a",
        radix_sum="%.1f s" % timing["radix_sum_s"],
        pyrad_sum="%.1f s" % timing["pyradiomics_sum_s"],
        roi_med="%.0f" % timing["roi_voxels_median"] if timing.get("roi_voxels_median") else "n/a",
        roi_min=str(timing.get("roi_voxels_min") or "n/a"),
        roi_max=str(timing.get("roi_voxels_max") or "n/a"),
        speedup=speedup_s,
        speedup_sum=speedup_sum_s,
        fail_table=_md_table(["Feature", "n cases", "% compared"], fail_rows),
        family_table=_md_table(["Family", "n", "note"], family_rows),
        slow_table=_md_table(
            ["Patient", "ROI voxels", "Radix s", "PyRadiomics s"], slow_rows
        ),
        error_table=_md_table(["Patient", "Error"], error_rows),
    )


def _limit_from_args(ns: argparse.Namespace) -> int | None:
    if ns.limit is not None:
        return ns.limit
    raw = os.environ.get("LUNG1_LIMIT", "").strip()
    if raw:
        return int(raw)
    return None


def main() -> int:
    if os.environ.get("LUNG1_WORKER") == "1":
        return _worker_main()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="first N GTV pairs")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild metrics + markdown from checkpoints; do not extract",
    )
    args = parser.parse_args()
    limit = _limit_from_args(args)

    if not PAIRS_CSV.is_file():
        raise SystemExit("missing %s — run scripts/fetch_lung1.py" % PAIRS_CSV)

    pairs = _load_pairs(limit)
    if not pairs:
        raise SystemExit("pairs.csv is empty")

    radix_py = os.environ.get("RADIX_PYTHON", sys.executable)
    pyrad_py = "" if args.report_only else str(require_pyrad_python())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {p["patient_id"] for p in pairs}
    _truncate_stale(CHECKPOINT_RADIX, keep)
    _truncate_stale(CHECKPOINT_PYRAD, keep)

    pairs_path = OUT_DIR / "pairs_selected.json"
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

    radix_meta = {"package": "pyradix", "version": "unknown"}
    pyrad_meta = {"package": "radiomics", "version": "unknown"}

    if not args.report_only:
        print("radix via %s" % radix_py, file=sys.stderr)
        print("pyradiomics via %s" % pyrad_py, file=sys.stderr)
        print("pairs: %d" % len(pairs), file=sys.stderr)
        radix_meta = _run_engine_supervised(radix_py, "radix", pairs, CHECKPOINT_RADIX) or radix_meta
        pyrad_meta = (
            _run_engine_supervised(pyrad_py, "pyradiomics", pairs, CHECKPOINT_PYRAD)
            or pyrad_meta
        )
    else:
        radix_done = _read_jsonl(CHECKPOINT_RADIX)
        pyrad_done = _read_jsonl(CHECKPOINT_PYRAD)
        if radix_done:
            sample = next(iter(radix_done.values()))
            radix_meta = {
                "package": sample.get("package", "pyradix"),
                "version": sample.get("version"),
            }
        if pyrad_done:
            sample = next(iter(pyrad_done.values()))
            pyrad_meta = {
                "package": sample.get("package", "radiomics"),
                "version": sample.get("version"),
            }

    radix_blob = _read_jsonl(CHECKPOINT_RADIX)
    pyrad_blob = _read_jsonl(CHECKPOINT_PYRAD)
    # last-write-wins already in _read_jsonl
    report, rows = build_report(pairs, radix_blob, pyrad_blob, radix_meta, pyrad_meta)
    write_outputs(report, rows)
    agr = report["agreement"]
    print(
        "compared %d/%d  pass %d  allowed %d  unexpected %d  errors %d"
        % (
            report["cohort"]["n_compared"],
            report["cohort"]["n_pairs"],
            agr["n_full_pass"],
            agr["n_allowed_only"],
            agr["n_unexpected_fail"],
            report["cohort"]["n_error"],
        ),
        file=sys.stderr,
    )
    if report["cohort"]["n_incomplete"]:
        return 2
    if agr["n_unexpected_fail"] or report["cohort"]["n_error"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
