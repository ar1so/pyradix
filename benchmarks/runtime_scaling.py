#!/usr/bin/env python3
"""Runtime and peak RSS against ROI size and thread count.

Size series uses the public PAT1 CT. The mask keeps the N voxels nearest the
centroid (stable mergesort). Thread series uses public brain1. PyRadiomics-mode
LoG (sigma 2 mm, first-order features only) is the path that calls the
multi-threaded recursive Gaussian. Original-feature extracts are included so
the OpenBLAS thread pin is visible on GLCM MCC as well as on the filter.
"""

from __future__ import annotations

import sys

from libbench import DATA, RESULTS, ensure_dirs, run_worker, write_json

SIZE_TARGETS = [2000, 8000, 32000, None]
THREAD_COUNTS = [1, 2, 4]


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _base(engine: str, dataset: str, kind: str, roi, repeats: int) -> dict:
    paths = DATA[dataset]
    if kind == "log_firstorder":
        image_types = [["LoG", {"sigma": [2.0]}]]
        classes = "firstorder"
    else:
        image_types = [["Original", {}]]
        classes = None
    return {
        "engine": engine,
        "job_id": f"{dataset}-{kind}-{engine}-{roi}",
        "image": str(paths["image"]),
        "mask": str(paths["mask"]),
        "mode": "pyradiomics",
        "bin_width": 25.0,
        "bin_count": None,
        "resample": None,
        "interpolator": "sitkBSpline",
        "voxel_array_shift": 0.0,
        "label": 1,
        "image_types": image_types,
        "feature_classes": classes,
        "roi_voxels": roi,
        "repeats": repeats,
        "warmup": 1,
        "return_values": False,
        "minimum_roi_dimensions": 1,
    }


def _one(series: str, dataset: str, kind: str, roi, threads: int, repeats: int, timeout_s: float) -> dict:
    rows = []
    for engine in ("radix", "pyradiomics"):
        print(f"{series} {dataset} {kind} roi={roi} threads={threads} {engine}", file=sys.stderr)
        spec = _base(engine, dataset, kind, roi, repeats)
        out = run_worker(spec, threads, timeout_s)
        row = {
            "series": series,
            "dataset": dataset,
            "kind": kind,
            "requested_roi_voxels": roi,
            "threads": threads,
            "engine": engine,
            "seconds": out.get("seconds", []),
            "median_s": _median(out.get("seconds", [])),
            "max_rss_kb": out.get("max_rss_kb"),
            "roi_voxels_reported": out.get("roi_voxels_reported"),
            "affinity": out.get("affinity"),
            "error": out.get("error"),
        }
        if out.get("error"):
            print(f"  ERROR {out.get('error')}", file=sys.stderr)
            if out.get("stderr_tail"):
                print(out["stderr_tail"], file=sys.stderr)
        else:
            print(f"  median {row['median_s']} rss_kb {row['max_rss_kb']}", file=sys.stderr)
        rows.append(row)
    return rows


def main() -> int:
    ensure_dirs()
    rows: list[dict] = []
    for roi in SIZE_TARGETS:
        # Full PAT1 is the slow end. Smaller masks should be quicker.
        timeout = 300 if roi is None else 180
        repeats = 2 if roi is None else 3
        rows.extend(_one("roi_size", "d2_pat1", "original", roi, 4, repeats, timeout))
    for threads in THREAD_COUNTS:
        rows.extend(_one("threads_original", "d4_brain1", "original", None, threads, 5, 180))
    for threads in THREAD_COUNTS:
        rows.extend(_one("threads_log", "d4_brain1", "log_firstorder", None, threads, 3, 240))
    # One large-image thread pair, original features, to see whether PAT1 moves.
    for threads in (1, 4):
        rows.extend(_one("threads_pat1_original", "d2_pat1", "original", None, threads, 2, 300))
    payload = {
        "protocol": {
            "size_series": (
                "Public IBSI-1 PAT1. Mask keeps the requested number of voxels "
                "nearest the centroid (mergesort). None means the full mask. "
                "Threads pinned to 4."
            ),
            "thread_series": (
                "Public brain1, and PAT1 at 1 and 4 threads. "
                "taskset plus OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, and "
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS."
            ),
            "log_series": "pyradiomics-mode LoG sigma 2 mm, first-order features only",
            "timing": "one untimed warmup after load; perf_counter around extract/execute",
            "memory": "ru_maxrss of the child process in kilobytes",
            "bin_width": 25,
            "random_seed": None,
        },
        "rows": rows,
    }
    write_json(RESULTS / "runtime.json", payload)
    print(f"wrote {RESULTS / 'runtime.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
