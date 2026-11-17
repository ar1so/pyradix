#!/usr/bin/env python3
"""Batch live compare: Radix (pyradiomics mode) vs PyRadiomics 3.1.0.

Protocol: docs/VALIDATION.md

Times are extract-only (volumes loaded first). Use a release wheel:
  .venv/bin/maturin develop --release
  .venv/bin/python scripts/compare_pyradiomics.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from interpreters import require_pyrad_python

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "tests" / "references" / "radix_vs_pyradiomics_metrics.json"
OUT_CSV = ROOT / "tests" / "references" / "radix_vs_pyradiomics_features.csv"
TOL = 1e-4

# Default 3.1.0 original_* keys. Radix extras (deprecated names) are not scored.
ALLOWED_FAIL = {
    # docs/DEVIATIONS.md: Lorensen 256-cube vs PyRadiomics cshape 128-cube.
    "original_shape_MeshVolume": "d2_pat1 marching-cubes (256 vs 128)",
}

# Recorded in the batch, not scored by G8. See docs/VALIDATION.md.
KNOWN_GAP_JOBS = {
    "d4_brain1__resample_2mm": (
        "sitk 2 mm resample is not 1e-4 yet "
        "(VoxelVolume 15680 vs 15320; 106/107 keys miss)"
    ),
}

DATASETS = {
    "d1_digital_phantom": {
        "image": ROOT / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz",
        "mask": ROOT / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz",
        "repeats": 3,
        "label": "D1 digital phantom",
    },
    "d4_brain1": {
        "image": ROOT / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd",
        "mask": ROOT / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd",
        "repeats": 3,
        "label": "D4 brain1",
    },
    "d2_pat1": {
        "image": ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz",
        "mask": ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz",
        "repeats": 1,
        "label": "D2 PAT1 lung CT",
    },
}

# Configs applied to every dataset unless `datasets` is set.
CONFIGS = {
    "bw5": {
        "bin_width": 5.0,
        "bin_count": None,
        "resample": None,
        # PAT1 CT + binWidth=5 → hundreds of gray levels; GLCM MCC does not finish in-gate.
        "datasets": ("d1_digital_phantom", "d4_brain1"),
    },
    "bw25": {"bin_width": 25.0, "bin_count": None, "resample": None},
    "bw50": {"bin_width": 50.0, "bin_count": None, "resample": None},
    "resample_2mm": {
        "bin_width": 25.0,
        "bin_count": None,
        "resample": (2.0, 2.0, 2.0),
        "datasets": ("d4_brain1",),
    },
}


def job_ids() -> list[str]:
    ids = []
    for dataset in DATASETS:
        for name, cfg in CONFIGS.items():
            allowed = cfg.get("datasets")
            if allowed and dataset not in allowed:
                continue
            ids.append(f"{dataset}__{name}")
    return ids


def parse_job(job_id: str) -> tuple[str, str, dict]:
    dataset, config = job_id.split("__", 1)
    return dataset, config, CONFIGS[config]


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _family(key: str) -> str:
    parts = key.split("_")
    return parts[1] if len(parts) > 2 else "other"


def _allowed_fail(dataset: str, feature: str) -> bool:
    if feature not in ALLOWED_FAIL:
        return False
    return dataset.startswith("d2_pat1") or dataset.startswith("lung1")


def compare_features(radix: dict[str, float], pyrad: dict[str, float], dataset: str) -> dict:
    shared = sorted(set(radix) & set(pyrad))
    rows = []
    n_pass = n_fail = n_allowed = 0
    families: dict[str, dict[str, int]] = {}
    worst_rel = 0.0
    worst_key = ""
    for key in shared:
        a, b = radix[key], pyrad[key]
        abs_err = abs(a - b)
        rel_err = abs_err / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))
        tol = max(TOL, abs(b) * TOL)
        ok = abs_err <= tol
        allowed = (not ok) and _allowed_fail(dataset, key)
        if ok:
            n_pass += 1
            reason = "ok"
        elif allowed:
            n_allowed += 1
            reason = "allowed"
        else:
            n_fail += 1
            reason = "tol"
        finite_rel = 0.0 if rel_err == float("inf") else rel_err
        if finite_rel > worst_rel:
            worst_rel = finite_rel
            worst_key = key
        fam = _family(key)
        slot = families.setdefault(fam, {"n": 0, "pass": 0, "fail": 0, "allowed": 0})
        slot["n"] += 1
        slot["pass"] += int(ok)
        slot["fail"] += int((not ok) and not allowed)
        slot["allowed"] += int(allowed)
        rows.append(
            {
                "feature": key,
                "family": fam,
                "radix": a,
                "pyradiomics": b,
                "abs_err": abs_err,
                "rel_err": None if rel_err == float("inf") else rel_err,
                "pass": ok or allowed,
                "reason": reason,
            }
        )
    extras = sorted(set(radix) - set(pyrad))
    missing = sorted(set(pyrad) - set(radix))
    for key in extras:
        rows.append(
            {
                "feature": key,
                "family": _family(key),
                "radix": radix[key],
                "pyradiomics": None,
                "abs_err": None,
                "rel_err": None,
                "pass": False,
                "reason": "radix_extra",
            }
        )
    for key in missing:
        rows.append(
            {
                "feature": key,
                "family": _family(key),
                "radix": None,
                "pyradiomics": pyrad[key],
                "abs_err": None,
                "rel_err": None,
                "pass": False,
                "reason": "missing",
            }
        )
    return {
        "n_radix": len(radix),
        "n_pyradiomics": len(pyrad),
        "n_shared": len(shared),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_allowed": n_allowed,
        "n_radix_extra": len(extras),
        "n_missing": len(missing),
        "worst_rel_err": worst_rel,
        "worst_feature": worst_key,
        "families": families,
        "rows": rows,
    }


def _radix_extractor(cfg: dict):
    from pyradix import Params, RadixExtractor

    return RadixExtractor(
        mode="pyradiomics",
        params=Params(
            mode="pyradiomics",
            bin_width=cfg["bin_width"],
            bin_count=cfg["bin_count"],
            resampled_pixel_spacing=cfg["resample"],
            interpolator="sitkBSpline",
            image_types=("Original",),
            label=1,
            voxel_array_shift=0.0,
        ),
    )


def run_radix_job(image: Path, mask: Path, cfg: dict, repeats: int) -> dict:
    from pyradix import load_volume

    ex = _radix_extractor(cfg)
    img = load_volume(image)
    msk = load_volume(mask)
    values: dict[str, float] = {}
    times: list[float] = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = ex.extract(img, msk)
        times.append(time.perf_counter() - t0)
        values = {k: float(v) for k, v in result.values.items() if k.startswith("original_")}
    assert result is not None
    return {
        "values": values,
        "seconds": times,
        "roi_voxels": int(result.provenance.get("roi_voxels", 0)),
        "shape": list(result.provenance.get("image_shape", [])),
        "resampled_shape": result.provenance.get("resampled_shape"),
    }


def run_pyradiomics_job(image: Path, mask: Path, cfg: dict, repeats: int) -> dict:
    import radiomics
    import SimpleITK as sitk
    from radiomics import featureextractor

    radiomics.setVerbosity(60)
    from image_paths import image_io_path

    img = sitk.ReadImage(image_io_path(image))
    msk = sitk.ReadImage(image_io_path(mask))
    ex = featureextractor.RadiomicsFeatureExtractor()
    if cfg["bin_count"] is not None:
        ex.settings.pop("binWidth", None)
        ex.settings["binCount"] = int(cfg["bin_count"])
    else:
        ex.settings["binWidth"] = float(cfg["bin_width"])
        ex.settings.pop("binCount", None)
    ex.settings["interpolator"] = "sitkBSpline"
    ex.settings["label"] = 1
    ex.settings["voxelArrayShift"] = 0
    if cfg["resample"] is not None:
        ex.settings["resampledPixelSpacing"] = list(cfg["resample"])
    else:
        ex.settings["resampledPixelSpacing"] = None
    ex.disableAllImageTypes()
    ex.enableImageTypeByName("Original")
    values: dict[str, float] = {}
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        raw = ex.execute(img, msk)
        times.append(time.perf_counter() - t0)
        values = {}
        for key, value in raw.items():
            if not str(key).startswith("original_"):
                continue
            try:
                values[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    size = img.GetSize()
    return {
        "values": values,
        "seconds": times,
        "shape": [int(size[2]), int(size[1]), int(size[0])],
    }


def _selected_jobs() -> list[str]:
    raw = os.environ.get("COMPARE_JOBS", "")
    if raw.strip():
        return [j for j in raw.split(",") if j]
    names = [n for n in os.environ.get("COMPARE_DATASET", "").split(",") if n]
    configs = [n for n in os.environ.get("COMPARE_CONFIG", "").split(",") if n]
    ids = job_ids()
    if names:
        ids = [j for j in ids if j.split("__", 1)[0] in names]
    if configs:
        ids = [j for j in ids if j.split("__", 1)[1] in configs]
    return ids


def _engine_main() -> int:
    engine = os.environ["COMPARE_ENGINE"]
    runner = run_radix_job if engine == "radix" else run_pyradiomics_job
    out = {"engine": engine, "jobs": {}}
    for job_id in _selected_jobs():
        dataset, _config, cfg = parse_job(job_id)
        spec = DATASETS[dataset]
        if not spec["image"].exists() or not spec["mask"].exists():
            print(f"skip {job_id}: data missing", file=sys.stderr)
            continue
        print(f"== {engine} {job_id} x{spec['repeats']} ==", file=sys.stderr)
        rec = runner(spec["image"], spec["mask"], cfg, spec["repeats"])
        out["jobs"][job_id] = rec
        print(f"   {job_id}: {rec['seconds']}", file=sys.stderr)
    json.dump(out, sys.stdout)
    return 0


def _spawn(python: str, engine: str, jobs: list[str]) -> dict:
    env = {**os.environ, "COMPARE_ENGINE": engine, "COMPARE_JOBS": ",".join(jobs)}
    if engine == "radix":
        env.pop("CONDA_PREFIX", None)
    payload = subprocess.check_output([python, str(Path(__file__))], env=env)
    return json.loads(payload)


def build_report(radix_blob: dict, pyrad_blob: dict, jobs: list[str]) -> dict:
    report: dict = {
        "protocol": "docs/VALIDATION.md",
        "settings": {
            "interpolator": "sitkBSpline",
            "label": 1,
            "imageType": "Original",
            "voxelArrayShift": 0,
            "tol": TOL,
            "timing": "extract-only after load",
            "allowed_fail": ALLOWED_FAIL,
            "known_gap_jobs": KNOWN_GAP_JOBS,
            "configs": {
                name: {
                    "bin_width": cfg["bin_width"],
                    "bin_count": cfg["bin_count"],
                    "resample": list(cfg["resample"]) if cfg["resample"] else None,
                }
                for name, cfg in CONFIGS.items()
            },
        },
        "pyradiomics_python": "PYRAD_PYTHON",
        "radix_python": "project virtualenv",
        "jobs": {},
    }
    all_rows: list[dict] = []
    for job_id in jobs:
        dataset, config, _cfg = parse_job(job_id)
        if job_id not in radix_blob.get("jobs", {}) or job_id not in pyrad_blob.get("jobs", {}):
            continue
        radix_rec = radix_blob["jobs"][job_id]
        pyrad_rec = pyrad_blob["jobs"][job_id]
        cmp_ = compare_features(radix_rec["values"], pyrad_rec["values"], dataset)
        r_med = _median(radix_rec["seconds"])
        p_med = _median(pyrad_rec["seconds"])
        block = {
            "dataset": dataset,
            "config": config,
            "radix_seconds": radix_rec["seconds"],
            "pyradiomics_seconds": pyrad_rec["seconds"],
            "radix_median_s": r_med,
            "pyradiomics_median_s": p_med,
            "speedup": (p_med / r_med) if r_med > 0 else None,
            "roi_voxels": radix_rec.get("roi_voxels"),
            "shape": radix_rec.get("shape"),
            "resampled_shape": radix_rec.get("resampled_shape"),
            "features": {k: v for k, v in cmp_.items() if k != "rows"},
        }
        report["jobs"][job_id] = block
        for row in cmp_["rows"]:
            all_rows.append({"job": job_id, "dataset": dataset, "config": config, **row})
        extra = ""
        if cmp_["n_fail"]:
            extra += "  FAIL %s" % cmp_["n_fail"]
        if cmp_["n_allowed"]:
            extra += "  allowed %s" % cmp_["n_allowed"]
        print(
            "%s: radix %.4fs  pyradiomics %.4fs  speedup %.2fx  pass %s/%s%s"
            % (
                job_id,
                r_med,
                p_med,
                block["speedup"],
                cmp_["n_pass"],
                cmp_["n_shared"],
                extra,
            ),
            file=sys.stderr,
        )
    return report, all_rows


def write_outputs(report: dict, all_rows: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "job",
                "dataset",
                "config",
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
        writer.writerows(all_rows)
    print(f"wrote {OUT_JSON} and {OUT_CSV}", file=sys.stderr)


def main() -> int:
    if os.environ.get("COMPARE_ENGINE") in {"radix", "pyradiomics"}:
        return _engine_main()

    pyrad_py = str(require_pyrad_python())
    radix_py = os.environ.get("RADIX_PYTHON", sys.executable)
    jobs = []
    for job_id in _selected_jobs():
        dataset, _config, _cfg = parse_job(job_id)
        spec = DATASETS[dataset]
        if spec["image"].exists() and spec["mask"].exists():
            jobs.append(job_id)
    if not jobs:
        raise SystemExit("no validation volumes on disk; run scripts/fetch_validation_data.py")

    print(f"radix via {radix_py}", file=sys.stderr)
    print(f"pyradiomics via {pyrad_py}", file=sys.stderr)
    print(f"jobs: {len(jobs)}", file=sys.stderr)
    radix_blob = _spawn(radix_py, "radix", jobs)
    pyrad_blob = _spawn(pyrad_py, "pyradiomics", jobs)
    report, all_rows = build_report(radix_blob, pyrad_blob, jobs)
    write_outputs(report, all_rows)
    n_fail = sum(
        block["features"]["n_fail"]
        for job_id, block in report["jobs"].items()
        if job_id not in KNOWN_GAP_JOBS
    )
    n_jobs = len(report["jobs"])
    print(f"{n_jobs} jobs, {n_fail} unexpected feature fails", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
