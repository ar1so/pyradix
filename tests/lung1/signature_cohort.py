"""Lung1 full signature versus installed PyRadiomics 3.1.0.

Requirement 8.3: pyradiomics mode, bin width 25, the criterion-1 signature,
native grid and 2 mm isotropic. Gradient, LBP2D, and LBP3D are the task 18 pilot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS_CSV = ROOT / "data" / "lung1" / "pairs.csv"
OUT_DIR = ROOT / "tests" / "references" / "lung1" / "signature"
REPORT_PATH = ROOT / "tests" / "references" / "lung1" / "signature_report.json"

TOL = 1e-4
CONFIG_ID = "full-signature-v1"
GRIDS = ("native", "resample_2mm")
SIGNATURE = (
    "Original",
    "LoG",
    "Wavelet",
    "Square",
    "SquareRoot",
    "Logarithm",
    "Exponential",
)
LOG_SIGMA = [1.0, 2.0, 3.0, 4.0, 5.0]
KNOWN_NO_TUMOR = ("LUNG1-128",)
WORKER_READY_S = 180.0
DEFAULT_TIMEOUT_S = 3600.0


def agrees(actual: float, expected: float, tol: float = TOL) -> bool:
    """True when |actual-expected| <= max(tol, tol*|expected|)."""
    if math.isnan(actual) and math.isnan(expected):
        return True
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return abs(actual - expected) <= max(tol, tol * abs(expected))


def tumor_present(mask, label: int = 1) -> bool:
    import numpy as np

    return bool(np.any(np.asanyarray(mask) == label))


def mismatches(patient_id: str, got: dict, reference: dict) -> list[dict]:
    found = []
    for key, expected in reference.items():
        if key not in got or not agrees(float(got[key]), float(expected)):
            found.append(
                {
                    "patient_id": patient_id,
                    "feature": key,
                    "radix": got.get(key),
                    "reference": expected,
                }
            )
    return found


def build_report(
    pairs: list[dict],
    skips: list[dict],
    by_grid: dict[str, dict[str, dict]],
    engines: dict,
) -> dict:
    skip_ids = {row["patient_id"] for row in skips}
    grids = {}
    for grid, records in by_grid.items():
        n_pass = n_fail = n_error = n_skip = 0
        failures: list[dict] = []
        extract_errors: list[dict] = []
        for pair in pairs:
            pid = pair["patient_id"]
            if pid in skip_ids:
                n_skip += 1
                continue
            sides = records.get(pid) or {}
            radix = sides.get("radix")
            pyrad = sides.get("pyradiomics")
            problem = _engine_problem(radix, pyrad)
            if problem is not None:
                n_error += 1
                extract_errors.append({"patient_id": pid, "error": problem})
                continue
            found = mismatches(pid, radix["values"], pyrad["values"])
            if found:
                n_fail += 1
                failures.extend(found)
                continue
            n_pass += 1
        grids[grid] = {
            "n_pairs": len(pairs),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_extract_error": n_error,
            "n_skip": n_skip,
            "failures": failures,
            "extract_errors": extract_errors,
        }
    return {
        "config_id": CONFIG_ID,
        "settings": {
            "mode": "pyradiomics",
            "bin_width": 25.0,
            "label": 1,
            "interpolator": "sitkBSpline",
            "image_types": list(SIGNATURE),
            "log_sigma": LOG_SIGMA,
            "wavelet": "coif1",
            "wavelet_level": 1,
            "grids": list(GRIDS),
            "tol": TOL,
        },
        "engines": engines,
        "skips": skips,
        "grids": grids,
    }


def _engine_problem(radix: dict | None, pyrad: dict | None) -> str | None:
    if not radix or not pyrad:
        return "missing engine output"
    if radix.get("error") or pyrad.get("error"):
        return "radix=%s; pyradiomics=%s" % (radix.get("error"), pyrad.get("error"))
    if not pyrad.get("values"):
        return "reference returned no features"
    return None


def _pyrad_python() -> Path:
    raw = os.environ.get("PYRAD_PYTHON", "").strip()
    if not raw:
        raise SystemExit(
            "Set PYRAD_PYTHON to the interpreter that has PyRadiomics 3.1.0 installed."
        )
    return Path(raw)


def _timeout_s() -> float:
    raw = os.environ.get("LUNG1_SIG_TIMEOUT", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_TIMEOUT_S


def _jobs(requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    raw = os.environ.get("LUNG1_JOBS", "").strip()
    if raw:
        return max(1, int(raw))
    return 2


def _load_pairs(limit: int | None) -> list[dict]:
    import csv

    rows = []
    with PAIRS_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pid = row["patient_id"].strip()
            rows.append(
                {
                    "patient_id": pid,
                    "image": str(ROOT / row["image"]),
                    "mask": str(ROOT / row["mask"]),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _mask_label(pair: dict) -> tuple[str, bool]:
    import nibabel as nib
    import numpy as np

    mask = Path(pair["mask"])
    if not mask.is_file():
        return pair["patient_id"], False
    img = nib.load(os.fspath(mask.resolve()))
    try:
        present = tumor_present(np.array(np.asanyarray(img.dataobj), copy=True))
    finally:
        img.uncache()
    return pair["patient_id"], present


def _scan_skips(pairs: list[dict]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    cache_path = OUT_DIR / "mask_labels.json"
    cache: dict[str, bool] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    pair_ids = {pair["patient_id"] for pair in pairs}
    skips = [
        {"patient_id": pid, "reason": "no tumor mask"}
        for pid in KNOWN_NO_TUMOR
        if pid not in pair_ids
    ]
    missing = [pair for pair in pairs if pair["patient_id"] not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (pid, present) in enumerate(pool.map(_mask_label, missing), start=1):
                cache[pid] = present
                if index % 25 == 0 or index == len(missing):
                    print("scanned %d/%d masks" % (index, len(missing)), file=sys.stderr)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    for pair in pairs:
        if not cache[pair["patient_id"]]:
            skips.append({"patient_id": pair["patient_id"], "reason": "no tumor mask"})
    return skips


def _read_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("config_id") != CONFIG_ID:
                continue
            done[rec["patient_id"]] = rec
    return done


def _append(path: Path, rec: dict, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rec) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _readline_timeout(proc: subprocess.Popen, timeout: float) -> str | None:
    found: queue.Queue[str | None] = queue.Queue()

    def _read() -> None:
        try:
            line = proc.stdout.readline() if proc.stdout is not None else ""
        except Exception:
            line = ""
        found.put(line)

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    line = found.get_nowait()
    if not line:
        return None
    return line


def _spawn(python: str, engine: str) -> tuple[subprocess.Popen, dict]:
    env = {**os.environ, "LUNG1_SIG_WORKER": "1", "LUNG1_SIG_ENGINE": engine}
    env.pop("CONDA_PREFIX", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
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
        raise RuntimeError("%s worker did not become ready" % engine)
    ready = json.loads(ready_line)
    if not ready.get("ready"):
        proc.kill()
        raise RuntimeError("%s worker ready payload invalid" % engine)
    return proc, ready.get("versions") or {}


def _stop(proc: subprocess.Popen | None) -> None:
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


def _numeric(raw: dict) -> dict[str, float]:
    values = {}
    for key, value in raw.items():
        if str(key).startswith("diagnostics_"):
            continue
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _image_types() -> tuple:
    return tuple((name, {"sigma": LOG_SIGMA} if name == "LoG" else {}) for name in SIGNATURE)


def _extract_radix(image: str, mask: str, grid: str) -> dict[str, float]:
    from pyradix import Params, RadixExtractor

    spacing = (2.0, 2.0, 2.0) if grid == "resample_2mm" else None
    params = Params(
        mode="pyradiomics",
        bin_width=25.0,
        interpolator="sitkBSpline",
        resampled_pixel_spacing=spacing,
        image_types=_image_types(),
        additional_info=False,
        label=1,
        voxel_array_shift=0.0,
    )
    result = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask)
    return _numeric(result.values)


def _extract_pyradiomics(image: str, mask: str, grid: str) -> dict[str, float]:
    import SimpleITK as sitk
    from radiomics import featureextractor

    settings = {
        "binWidth": 25,
        "label": 1,
        "interpolator": "sitkBSpline",
        "additionalInfo": False,
        "voxelArrayShift": 0,
    }
    if grid == "resample_2mm":
        settings["resampledPixelSpacing"] = [2.0, 2.0, 2.0]
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    for name, custom in _image_types():
        if custom:
            extractor.enableImageTypeByName(name, customArgs=custom)
        else:
            extractor.enableImageTypeByName(name)
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from image_paths import image_io_path

    return _numeric(
        extractor.execute(
            sitk.ReadImage(image_io_path(image)),
            sitk.ReadImage(image_io_path(mask)),
        )
    )


def _versions(engine: str) -> dict:
    if engine == "pyradiomics":
        import radiomics

        return {
            "package": "radiomics",
            "version": str(radiomics.__version__),
            "python": "PYRAD_PYTHON",
        }
    import pyradix

    return {
        "package": "pyradix",
        "version": str(getattr(pyradix, "__version__", "0.1.0")),
        "python": "project virtualenv",
    }


def _worker_main() -> int:
    engine = os.environ["LUNG1_SIG_ENGINE"]
    if engine == "pyradiomics":
        import radiomics

        radiomics.setVerbosity(60)
    versions = _versions(engine)
    extract = _extract_radix if engine == "radix" else _extract_pyradiomics
    print(json.dumps({"ready": True, "versions": versions}), flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        job = json.loads(line)
        rec = {
            "config_id": CONFIG_ID,
            "patient_id": job["patient_id"],
            "grid": job["grid"],
            "engine": engine,
            "package": versions["package"],
            "version": versions["version"],
            "python": versions["python"],
            "error": None,
            "values": {},
            "seconds": None,
        }
        try:
            started = time.perf_counter()
            rec["values"] = extract(job["image"], job["mask"], job["grid"])
            rec["seconds"] = time.perf_counter() - started
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["values"] = {}
        print(json.dumps(rec), flush=True)
    return 0


def _consume(
    python: str,
    engine: str,
    grid: str,
    tasks: queue.Queue,
    checkpoint: Path,
    lock: threading.Lock,
    versions: dict,
    errors: list,
    total: int,
    counter: list[int],
) -> None:
    proc = None
    try:
        proc, info = _spawn(python, engine)
        with lock:
            versions[engine] = info
        timeout = _timeout_s()
        while True:
            job = tasks.get()
            if job is None:
                return
            pid = job["patient_id"]
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(job) + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                _stop(proc)
                proc, info = _spawn(python, engine)
                proc.stdin.write(json.dumps(job) + "\n")
                proc.stdin.flush()
            line = _readline_timeout(proc, timeout)
            if line is None:
                _stop(proc)
                rec = {
                    "config_id": CONFIG_ID,
                    "patient_id": pid,
                    "grid": grid,
                    "engine": engine,
                    "package": info.get("package"),
                    "version": info.get("version"),
                    "python": info.get("python"),
                    "error": "Timeout: extract exceeded %.0fs" % timeout,
                    "values": {},
                    "seconds": None,
                }
                proc, info = _spawn(python, engine)
            else:
                rec = json.loads(line)
            _append(checkpoint, rec, lock)
            with lock:
                counter[0] += 1
                done = counter[0]
            if rec.get("error"):
                print(
                    "%s %s %d/%d %s ERROR %s" % (engine, grid, done, total, pid, rec["error"]),
                    file=sys.stderr,
                )
            else:
                seconds = rec.get("seconds") or 0.0
                print(
                    "%s %s %d/%d %s %.1fs n=%d"
                    % (engine, grid, done, total, pid, seconds, len(rec["values"])),
                    file=sys.stderr,
                )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        errors.append("%s %s: %s: %s" % (engine, grid, type(exc).__name__, exc))
    finally:
        _stop(proc)


def _remember_engines(engines: dict) -> None:
    for grid in GRIDS:
        for engine in ("radix", "pyradiomics"):
            if engines.get(engine, {}).get("version"):
                continue
            done = _read_done(OUT_DIR / grid / ("%s.jsonl" % engine))
            for rec in done.values():
                if rec.get("version"):
                    engines[engine] = {
                        "package": rec.get("package"),
                        "version": rec.get("version"),
                        "python": rec.get("python"),
                    }
                    break


def _run_grid(
    grid: str,
    pairs: list[dict],
    python_for: dict[str, str],
    jobs: int,
    engines: dict,
) -> None:
    lock = threading.Lock()
    errors: list[str] = []
    versions: dict = {}
    threads: list[threading.Thread] = []
    for engine, python in python_for.items():
        checkpoint = OUT_DIR / grid / ("%s.jsonl" % engine)
        done = _read_done(checkpoint)
        remaining = [pair for pair in pairs if pair["patient_id"] not in done]
        print(
            "%s %s already %d/%d"
            % (engine, grid, len(pairs) - len(remaining), len(pairs)),
            file=sys.stderr,
        )
        if not remaining:
            continue
        tasks: queue.Queue = queue.Queue()
        for pair in remaining:
            tasks.put({**pair, "grid": grid})
        workers = min(jobs, len(remaining))
        for _ in range(workers):
            tasks.put(None)
        counter = [len(pairs) - len(remaining)]
        for _ in range(workers):
            threads.append(
                threading.Thread(
                    target=_consume,
                    args=(
                        python,
                        engine,
                        grid,
                        tasks,
                        checkpoint,
                        lock,
                        versions,
                        errors,
                        len(pairs),
                        counter,
                    ),
                )
            )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    for engine, info in versions.items():
        engines[engine] = dict(info)
    _remember_engines(engines)


def _load_grid(grid: str, pairs: list[dict]) -> dict[str, dict]:
    combined = {}
    for pair in pairs:
        combined[pair["patient_id"]] = {}
    for engine in ("radix", "pyradiomics"):
        done = _read_done(OUT_DIR / grid / ("%s.jsonl" % engine))
        for pid, rec in done.items():
            if pid in combined:
                combined[pid][engine] = rec
    return combined


def run_cohort(limit: int | None = None, jobs: int | None = None) -> dict:
    if not PAIRS_CSV.is_file():
        raise SystemExit("Lung1 pairs.csv missing — run scripts/fetch_lung1.py")
    pairs = _load_pairs(limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skips = _scan_skips(pairs)
    skip_ids = {row["patient_id"] for row in skips}
    work = [pair for pair in pairs if pair["patient_id"] not in skip_ids]
    python_for = {"radix": sys.executable, "pyradiomics": str(_pyrad_python())}
    engines: dict = {}
    by_grid = {}
    for grid in GRIDS:
        _run_grid(grid, work, python_for, _jobs(jobs), engines)
        by_grid[grid] = _load_grid(grid, pairs)
        report = build_report(pairs, skips, by_grid, engines)
        _write_report(report)
        block = report["grids"][grid]
        print(
            "%s pass=%d fail=%d extract_error=%d skip=%d"
            % (grid, block["n_pass"], block["n_fail"], block["n_extract_error"], block["n_skip"]),
            file=sys.stderr,
        )
        for failure in block["failures"][:10]:
            print(_failure_line(grid, failure), file=sys.stderr)
        if block["n_fail"] or block["n_extract_error"]:
            break
    report = build_report(pairs, skips, by_grid, engines)
    _write_report(report)
    return report


def _failure_line(grid: str, failure: dict) -> str:
    return "%s %s %s: radix %s reference %s" % (
        failure["patient_id"],
        grid,
        failure["feature"],
        failure["radix"],
        failure["reference"],
    )


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("LUNG1_SIG_WORKER") == "1":
        return _worker_main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)
    run_cohort(limit=args.limit, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
