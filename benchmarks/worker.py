#!/usr/bin/env python3
"""One extract in a child process. Prints a single JSON object on stdout.

The parent pins CPUs with taskset and sets OpenBLAS/ITK thread variables
before this process starts. Peak RSS is the child process maximum, so it
includes the interpreter and the loaded volume.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _affinity() -> list[int]:
    try:
        return sorted(os_affinity())
    except (AttributeError, OSError):
        return []


def os_affinity() -> set[int]:
    return set(os.sched_getaffinity(0))  # type: ignore[attr-defined]


import os  # noqa: E402  — affinity helper above is clearer with the import kept local to the call


def _subset_mask(mask, n: int | None):
    import numpy as np

    if n is None:
        return mask
    coords = np.argwhere(mask)
    if coords.shape[0] <= n:
        return mask
    center = coords.mean(axis=0)
    distance = np.linalg.norm(coords - center, axis=1)
    order = np.argsort(distance, kind="mergesort")
    keep = coords[order[:n]]
    out = np.zeros(mask.shape, dtype=bool)
    out[tuple(keep.T)] = True
    return out


def _run_radix(spec: dict) -> tuple[list[float], dict[str, float], int]:
    import numpy as np

    from pyradix import Params, RadixExtractor, load_volume
    from pyradix._radix import feature_catalog

    image = load_volume(spec["image"])
    mask = load_volume(spec["mask"])
    roi = mask.array == int(spec.get("label", 1))
    roi = _subset_mask(roi, spec.get("roi_voxels"))
    mask.array = roi.astype(np.float64)
    catalog = feature_catalog()
    classes = spec.get("feature_classes")
    if classes == "firstorder":
        feature_classes = {"firstorder": tuple(catalog["firstorder"]["features"])}
    elif classes == "all":
        feature_classes = {
            name: tuple(entry["features"]) for name, entry in catalog.items()
        }
    else:
        feature_classes = None
    image_types = spec.get("image_types") or [("Original", {})]
    params = Params(
        mode=spec["mode"],
        bin_width=spec.get("bin_width"),
        bin_count=spec.get("bin_count"),
        resampled_pixel_spacing=spec.get("resample"),
        interpolator=spec.get("interpolator", "sitkBSpline"),
        mask_interpolator=spec.get("mask_interpolator"),
        resegment_range=spec.get("resegment_range"),
        outlier_sigma=spec.get("outlier_sigma"),
        image_types=tuple(tuple(item) for item in image_types),
        feature_classes=feature_classes,
        label=int(spec.get("label", 1)),
        voxel_array_shift=float(spec.get("voxel_array_shift", 0.0)),
        minimum_roi_dimensions=int(spec.get("minimum_roi_dimensions", 2)),
        additional_info=False,
    )
    extractor = RadixExtractor(mode=spec["mode"], params=params)
    repeats = int(spec.get("repeats", 1))
    warmup = int(spec.get("warmup", 0))
    values: dict[str, float] = {}
    roi_voxels = 0
    for _ in range(warmup):
        result = extractor.extract(image, mask)
        values = {k: float(v) for k, v in result.values.items()}
        roi_voxels = int(result.provenance.get("roi_voxels", 0))
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = extractor.extract(image, mask)
        times.append(time.perf_counter() - t0)
        values = {k: float(v) for k, v in result.values.items()}
        roi_voxels = int(result.provenance.get("roi_voxels", 0))
    if spec.get("include_provenance"):
        values["__config_hash__"] = float("nan")
        # Hash is returned separately; keep values numeric.
        spec_hash = result.provenance.get("config_hash", "")
        return times, values, roi_voxels, spec_hash  # type: ignore[return-value]
    return times, values, roi_voxels


def _run_pyradiomics(spec: dict) -> tuple[list[float], dict[str, float], int]:
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(60)
    image = sitk.ReadImage(str(spec["image"]))
    mask_img = sitk.ReadImage(str(spec["mask"]))
    if spec.get("roi_voxels") is not None:
        import numpy as np

        arr = sitk.GetArrayFromImage(mask_img) == int(spec.get("label", 1))
        arr = _subset_mask(arr, int(spec["roi_voxels"]))
        rebuilt = sitk.GetImageFromArray(arr.astype(np.uint8))
        rebuilt.CopyInformation(mask_img)
        mask_img = rebuilt
    extractor = featureextractor.RadiomicsFeatureExtractor()
    if spec.get("bin_count") is not None:
        extractor.settings.pop("binWidth", None)
        extractor.settings["binCount"] = int(spec["bin_count"])
    elif spec.get("bin_width") is not None:
        extractor.settings["binWidth"] = float(spec["bin_width"])
        extractor.settings.pop("binCount", None)
    extractor.settings["interpolator"] = spec.get("interpolator", "sitkBSpline")
    extractor.settings["label"] = int(spec.get("label", 1))
    extractor.settings["voxelArrayShift"] = float(spec.get("voxel_array_shift", 0.0))
    extractor.settings["minimumROIDimensions"] = int(spec.get("minimum_roi_dimensions", 2))
    if spec.get("resample") is not None:
        extractor.settings["resampledPixelSpacing"] = list(spec["resample"])
    else:
        extractor.settings["resampledPixelSpacing"] = None
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    kinds = spec.get("image_types") or [("Original", {})]
    names = [name for name, _settings in kinds]
    if spec.get("feature_classes") == "firstorder":
        extractor.enableFeatureClassByName("firstorder")
    else:
        extractor.enableAllFeatures()
    for name, settings in kinds:
        if name == "LoG":
            sigma = settings.get("sigma", [2.0])
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(sigma)})
        elif name == "Original":
            extractor.enableImageTypeByName("Original")
        else:
            extractor.enableImageTypeByName(name)
    repeats = int(spec.get("repeats", 1))
    warmup = int(spec.get("warmup", 0))
    values: dict[str, float] = {}

    def once() -> dict[str, float]:
        raw = extractor.execute(image, mask_img)
        out: dict[str, float] = {}
        for key, value in raw.items():
            text = str(key)
            if text.startswith("diagnostics_"):
                continue
            try:
                out[text] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    for _ in range(warmup):
        values = once()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        values = once()
        times.append(time.perf_counter() - t0)
    import numpy as np

    label = int(spec.get("label", 1))
    roi = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_img) == label))
    return times, values, roi


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        if spec["engine"] == "radix":
            ran = _run_radix(spec)
        elif spec["engine"] == "pyradiomics":
            ran = _run_pyradiomics(spec)
        else:
            raise SystemExit(f"unknown engine {spec['engine']}")
        config_hash = ""
        if len(ran) == 4:
            times, values, roi, config_hash = ran
        else:
            times, values, roi = ran
        values.pop("__config_hash__", None)
        payload = {
            "seconds": times,
            "max_rss_kb": _rss_kb(),
            "affinity": _affinity(),
            "roi_voxels_reported": roi,
            "n_values": len(values),
            "config_hash": config_hash,
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "itk_threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"),
        }
        if spec.get("return_values", True):
            payload["values"] = values
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # noqa: BLE001 — the parent records the failure
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
