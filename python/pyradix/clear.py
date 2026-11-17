"""CLEAR-ready YAML dump and PyRadiomics-style diagnostics."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pyradix._radix import count_components_26
from pyradix._version import __version__ as _SOFTWARE_VERSION

REQUIRED_KEYS = (
    "software",
    "software_version",
    "extraction_mode",
    "discretization",
    "resampling",
    "image_types",
    "config_hash",
    "n_features",
)


def clear_report(result) -> dict[str, Any]:
    prov = dict(result.provenance)
    return {
        "software": "Radix",
        "software_version": _SOFTWARE_VERSION,
        "extraction_mode": prov.get("mode"),
        "discretization": {
            "bin_width": prov.get("bin_width"),
            "bin_count": prov.get("bin_count"),
            "bin_min": prov.get("bin_min"),
        },
        "resampling": {
            "resampled_pixel_spacing": prov.get("resampled_pixel_spacing"),
            "interpolator": prov.get("interpolator"),
            "mask_interpolator": prov.get("mask_interpolator"),
        },
        "image_types": prov.get("image_types"),
        "resegmentation": {
            "resegment_range": prov.get("resegment_range"),
            "outlier_sigma": prov.get("outlier_sigma"),
        },
        "config_hash": prov.get("config_hash"),
        "n_features": len(result.values),
        "label": prov.get("label"),
        "roi_voxels": prov.get("roi_voxels"),
    }


def dump_clear_yaml(result, path: str | Path) -> dict[str, Any]:
    body = clear_report(result)
    missing = [key for key in REQUIRED_KEYS if key not in body or body[key] is None]
    if missing:
        raise ValueError(f"CLEAR dump missing keys: {missing}")
    Path(path).write_text(
        yaml.safe_dump(body, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return body


def diagnostics_map(
    image: np.ndarray,
    roi: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    direction: tuple[float, ...],
    settings: dict[str, Any],
    image_types: dict[str, Any],
    interpolated: tuple[tuple[int, int, int], tuple[float, float, float]] | None = None,
    resegmented: np.ndarray | None = None,
) -> dict[str, Any]:
    """PyRadiomics `diagnostics_` fields. Interpolated and resegmented keys are optional."""
    out: dict[str, Any] = {
        "diagnostics_Versions_Radix": _SOFTWARE_VERSION,
        "diagnostics_Versions_Numpy": np.__version__,
        "diagnostics_Versions_Python": "%i.%i.%i" % sys.version_info[:3],
    }
    _image_fields(out, "original", image, spacing_zyx, include_identity=True)
    _mask_fields(out, "original", image, roi, spacing_zyx, origin_xyz, direction)
    if interpolated is not None:
        dim_zyx, spacing = interpolated
        _grid_fields(out, "interpolated", dim_zyx, spacing)
    if resegmented is not None and np.any(resegmented):
        _mask_fields(out, "resegmented", image, resegmented, spacing_zyx, origin_xyz, direction)
    out["diagnostics_Configuration_Settings"] = settings
    out["diagnostics_Configuration_EnabledImageTypes"] = image_types
    return out


def resegmented_roi(
    image: np.ndarray,
    roi: np.ndarray,
    resegment_range: tuple[float, float] | None,
    resegment_mode: str,
    outlier_sigma: float | None,
) -> np.ndarray | None:
    """Intensity mask after a requested resegmentation. `None` when that step is off."""
    if resegment_range is None and outlier_sigma is None:
        return None
    mask = np.array(roi, dtype=bool, copy=True)
    if resegment_range is not None:
        lo, hi = _thresholds(image, mask, resegment_range, resegment_mode)
        mask &= (image >= lo) & (image <= hi)
    if outlier_sigma is not None and outlier_sigma > 0 and int(mask.sum()) >= 2:
        values = image[mask]
        mean = float(values.mean())
        sigma = float(values.std(ddof=1))
        lo = mean - outlier_sigma * sigma
        hi = mean + outlier_sigma * sigma
        mask &= (image >= lo) & (image <= hi)
    return mask


def _thresholds(image, mask, bounds, mode: str) -> tuple[float, float]:
    lo, hi = (float(bounds[0]), float(bounds[1]))
    if lo > hi:
        lo, hi = hi, lo
    values = image[mask]
    if mode == "relative":
        peak = float(values.max())
        lo, hi = peak * lo, peak * hi
    elif mode == "sigma":
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        lo, hi = mean + std * lo, mean + std * hi
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _xyz(spacing_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))


def _size_xyz(shape_zyx) -> tuple[int, int, int]:
    return (int(shape_zyx[2]), int(shape_zyx[1]), int(shape_zyx[0]))


def _sha1(array: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


def _image_fields(out, prefix: str, image: np.ndarray, spacing_zyx, include_identity: bool) -> None:
    if include_identity:
        out[f"diagnostics_Image-{prefix}_Hash"] = _sha1(image)
        out[f"diagnostics_Image-{prefix}_Dimensionality"] = "3D"
    out[f"diagnostics_Image-{prefix}_Spacing"] = _xyz(spacing_zyx)
    out[f"diagnostics_Image-{prefix}_Size"] = _size_xyz(image.shape)
    out[f"diagnostics_Image-{prefix}_Mean"] = float(image.mean())
    out[f"diagnostics_Image-{prefix}_Minimum"] = float(image.min())
    out[f"diagnostics_Image-{prefix}_Maximum"] = float(image.max())


def _grid_fields(out, prefix: str, dim_zyx, spacing_zyx) -> None:
    spacing = _xyz(spacing_zyx)
    size = _size_xyz(dim_zyx)
    out[f"diagnostics_Image-{prefix}_Spacing"] = spacing
    out[f"diagnostics_Image-{prefix}_Size"] = size
    out[f"diagnostics_Mask-{prefix}_Spacing"] = spacing
    out[f"diagnostics_Mask-{prefix}_Size"] = size


def _mask_fields(out, prefix, image, roi, spacing_zyx, origin_xyz, direction) -> None:
    roi = np.asarray(roi, dtype=bool)
    if prefix == "original":
        out[f"diagnostics_Mask-{prefix}_Hash"] = _sha1(roi.astype(np.uint8))
    out[f"diagnostics_Mask-{prefix}_Spacing"] = _xyz(spacing_zyx)
    out[f"diagnostics_Mask-{prefix}_Size"] = _size_xyz(image.shape)
    coords = np.argwhere(roi)
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    out[f"diagnostics_Mask-{prefix}_BoundingBox"] = (
        int(x.min()),
        int(y.min()),
        int(z.min()),
        int(x.max() - x.min() + 1),
        int(y.max() - y.min() + 1),
        int(z.max() - z.min() + 1),
    )
    out[f"diagnostics_Mask-{prefix}_VoxelNum"] = int(roi.sum())
    out[f"diagnostics_Mask-{prefix}_VolumeNum"] = _volume_count(roi)
    index = (float(x.mean()), float(y.mean()), float(z.mean()))
    out[f"diagnostics_Mask-{prefix}_CenterOfMassIndex"] = index
    out[f"diagnostics_Mask-{prefix}_CenterOfMass"] = _physical(
        origin_xyz, spacing_zyx, direction, index
    )
    values = image[roi]
    out[f"diagnostics_Mask-{prefix}_Mean"] = float(values.mean())
    out[f"diagnostics_Mask-{prefix}_Minimum"] = float(values.min())
    out[f"diagnostics_Mask-{prefix}_Maximum"] = float(values.max())


def _physical(origin_xyz, spacing_zyx, direction, index_xyz) -> tuple[float, float, float]:
    if len(direction) == 9:
        axes = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    else:
        axes = np.diag(_xyz(spacing_zyx))
    origin = np.asarray(origin_xyz, dtype=np.float64)
    index = np.asarray(index_xyz, dtype=np.float64)
    world = origin + index @ axes.T
    return (float(world[0]), float(world[1]), float(world[2]))


def _volume_count(roi: np.ndarray) -> int:
    """Fully connected volumes, 26-neighbourhood (counted in Rust)."""
    roi = np.ascontiguousarray(roi, dtype=bool)
    if roi.ndim != 3:
        roi = roi.reshape((1,) * (3 - roi.ndim) + roi.shape)
    return int(count_components_26(roi))
