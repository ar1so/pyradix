#!/usr/bin/env python3
"""Live pyradix pyradiomics-mode versus installed PyRadiomics 3.1.0.

Each tool reads the public file itself. The clock in the child process
covers execute/extract only, after the volume is loaded, with one untimed
warmup. OpenBLAS and ITK are pinned to 4 threads (the whole machine).
"""

from __future__ import annotations

import csv
import sys

from libbench import DATA, RESULTS, TOL, compare_pair, ensure_dirs, run_worker, write_json

JOBS = [
    ("d1_digital_phantom", "bw5", 5.0, None, None, 3, 60),
    ("d1_digital_phantom", "bw25", 25.0, None, None, 3, 60),
    ("d1_digital_phantom", "bw50", 50.0, None, None, 3, 60),
    ("d4_brain1", "bw5", 5.0, None, None, 3, 180),
    ("d4_brain1", "bw25", 25.0, None, None, 3, 180),
    ("d4_brain1", "bw50", 50.0, None, None, 3, 180),
    ("d4_brain1", "resample_2mm", 25.0, None, [2.0, 2.0, 2.0], 2, 240),
    ("d2_pat1", "bw25", 25.0, None, None, 2, 300),
    ("d2_pat1", "bw50", 50.0, None, None, 2, 300),
    ("d2_pat1", "bw5", 5.0, None, None, 1, 120),
]


def _spec(engine: str, dataset: str, bin_width: float, resample, repeats: int) -> dict:
    paths = DATA[dataset]
    return {
        "engine": engine,
        "job_id": f"{dataset}-{engine}",
        "image": str(paths["image"]),
        "mask": str(paths["mask"]),
        "mode": "pyradiomics",
        "bin_width": bin_width,
        "bin_count": None,
        "resample": resample,
        "interpolator": "sitkBSpline",
        "voxel_array_shift": 0.0,
        "label": 1,
        "image_types": [["Original", {}]],
        "feature_classes": None,
        "roi_voxels": None,
        "repeats": repeats,
        "warmup": 1,
        "return_values": True,
        "minimum_roi_dimensions": 2,
    }


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def main() -> int:
    ensure_dirs()
    jobs = []
    feature_rows = []
    for dataset, config, width, _count, resample, repeats, timeout_s in JOBS:
        paths = DATA[dataset]
        if not paths["image"].exists():
            jobs.append({"dataset": dataset, "config": config, "error": "file missing"})
            continue
        print(f"== {dataset} {config} ==", file=sys.stderr)
        radix = run_worker(
            _spec("radix", dataset, width, resample, repeats), 4, timeout_s
        )
        pyrad = run_worker(
            _spec("pyradiomics", dataset, width, resample, repeats), 4, timeout_s
        )
        block = {
            "dataset": dataset,
            "config": config,
            "bin_width": width,
            "resample_mm": resample,
            "repeats": repeats,
            "threads": 4,
            "licence": paths["licence"],
            "radix_error": radix.get("error"),
            "pyradiomics_error": pyrad.get("error"),
            "radix_seconds": radix.get("seconds", []),
            "pyradiomics_seconds": pyrad.get("seconds", []),
            "radix_median_s": _median(radix.get("seconds", [])),
            "pyradiomics_median_s": _median(pyrad.get("seconds", [])),
            "radix_max_rss_kb": radix.get("max_rss_kb"),
            "pyradiomics_max_rss_kb": pyrad.get("max_rss_kb"),
            "radix_roi_voxels": radix.get("roi_voxels_reported"),
            "pyradiomics_roi_voxels": pyrad.get("roi_voxels_reported"),
        }
        if radix.get("seconds") and pyrad.get("seconds") and block["radix_median_s"]:
            block["speedup"] = block["pyradiomics_median_s"] / block["radix_median_s"]
        if radix.get("values") and pyrad.get("values"):
            # Score original_* keys only. Deprecated names are reported as extras.
            radix_orig = {k: v for k, v in radix["values"].items() if k.startswith("original_")}
            pyrad_orig = {k: v for k, v in pyrad["values"].items() if k.startswith("original_")}
            cmp_ = compare_pair(radix_orig, pyrad_orig, TOL)
            block["agreement"] = {k: v for k, v in cmp_.items() if k != "rows"}
            for row in cmp_["rows"]:
                feature_rows.append({"dataset": dataset, "config": config, **row})
            print(
                f"   pass {cmp_['n_pass']}/{cmp_['n_shared']} fail {cmp_['n_fail']} "
                f"speedup {block.get('speedup')}",
                file=sys.stderr,
            )
        else:
            print(
                f"   radix={radix.get('error')} pyrad={pyrad.get('error')}",
                file=sys.stderr,
            )
            if radix.get("stderr_tail"):
                print(radix["stderr_tail"], file=sys.stderr)
            if pyrad.get("stderr_tail"):
                print(pyrad["stderr_tail"], file=sys.stderr)
        jobs.append(block)
    payload = {
        "protocol": {
            "mode": "pyradiomics",
            "image_type": "Original",
            "interpolator": "sitkBSpline",
            "voxelArrayShift": 0,
            "label": 1,
            "tolerance": "abs(a-b) <= max(1e-4, 1e-4*|b|)",
            "timing": "one untimed warmup after load, then perf_counter around extract/execute only",
            "threads": "taskset to 4 CPUs; OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4",
            "readers": "each engine reads the public file with its own loader",
        },
        "jobs": jobs,
    }
    write_json(RESULTS / "pyradiomics_agreement.json", payload)
    csv_path = RESULTS / "pyradiomics_features.csv"
    fields = [
        "dataset",
        "config",
        "feature",
        "family",
        "radix",
        "reference",
        "abs_err",
        "rel_err",
        "pass",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"wrote {csv_path} ({len(feature_rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
