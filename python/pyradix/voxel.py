"""Voxel maps through the image-type registry and feature classes.

The kernel is ``2 * kernelRadius + 1``. One batch is at most ``voxelBatch``
voxels, and each voxel is calculated on its own, so a co-occurrence matrix
is not kept for every voxel at once. Shape and diagnostics are not maps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyradix._radix import RadixError, apply_image_type, extract_features
from pyradix.params import Params

DEFAULT_VOXEL_BATCH = 256
_SHAPE = frozenset({"shape", "shape2D"})


def extract_voxel_maps(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    kernel_radius: int | None = None,
    voxel_batch: int | None = None,
    features: tuple[str, ...] | None = None,
    params: Params | None = None,
    mode: str | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    masked_kernel: bool | None = None,
    init_value: float | None = None,
) -> dict[str, np.ndarray]:
    """Per-ROI-voxel maps. Voxels outside the ROI stay at ``initValue``."""
    image = np.ascontiguousarray(image, dtype=np.float64)
    roi = np.ascontiguousarray(mask, dtype=bool)
    if image.shape != roi.shape or image.ndim != 3:
        raise ValueError("image and mask must be 3-D and the same shape")
    radius, batch, masked, fill, use_mode, classes, type_rows, bin_width = _resolve(
        params, kernel_radius, voxel_batch, features, mode, masked_kernel, init_value
    )
    volumes = _derived_volumes(image, spacing, use_mode, type_rows)
    maps = {
        f"{prefix}_{class_name}_{feature}": np.full(image.shape, fill, dtype=np.float64)
        for prefix, _volume in volumes
        for class_name, names in classes
        for feature in names
    }
    coords = np.argwhere(roi)
    for start in range(0, coords.shape[0], batch):
        for z, y, x in coords[start : start + batch]:
            kernel_mask = _kernel_mask(roi, int(z), int(y), int(x), radius, masked)
            if not np.any(kernel_mask):
                continue
            for prefix, volume in volumes:
                kernel = _kernel_image(volume, int(z), int(y), int(x), radius)
                got = _kernel_features(kernel, kernel_mask, spacing, use_mode, bin_width, classes)
                for class_name, feature, value in got:
                    maps[f"{prefix}_{class_name}_{feature}"][z, y, x] = value
    return maps


def _resolve(params, kernel_radius, voxel_batch, features, mode, masked_kernel, init_value):
    setting = {} if params is None else params.voxel_setting
    radius = int(setting.get("kernelRadius", 1) if kernel_radius is None else kernel_radius)
    if radius <= 0:
        raise ValueError("kernel_radius must be > 0")
    batch = voxel_batch
    if batch is None and params is not None:
        batch = params.voxel_batch if params.voxel_batch is not None else setting.get("voxelBatch")
    if batch is None:
        batch = DEFAULT_VOXEL_BATCH
    batch = int(batch)
    if batch <= 0:
        raise ValueError("voxel_batch must be > 0")
    masked = bool(setting.get("maskedKernel", True) if masked_kernel is None else masked_kernel)
    fill = float(setting.get("initValue", 0.0) if init_value is None else init_value)
    use_mode = params.mode if params is not None and mode is None else (mode or "pyradiomics")
    if params is None or features is not None:
        names = ("Mean",) if features is None else tuple(features)
        classes = [("firstorder", list(names))]
        type_rows = [("Original", [])]
    else:
        classes = [
            (name, list(values))
            for name, values in params.enabled_features().items()
            if name not in _SHAPE and values
        ]
        from pyradix.extractor import _image_type_args

        type_rows = _image_type_args(params)
    bin_width = None if params is None else params.bin_width
    if bin_width is None:
        bin_width = 25.0
    return radius, batch, masked, fill, use_mode, classes, type_rows, float(bin_width)


def _derived_volumes(image, spacing, mode, type_rows) -> list[tuple[str, np.ndarray]]:
    volumes: list[tuple[str, np.ndarray]] = []
    for name, settings in type_rows:
        for prefix, volume in apply_image_type(name, mode, image, spacing, list(settings)):
            volumes.append((prefix, np.asarray(volume, dtype=np.float64)))
    return volumes


def _kernel_image(volume: np.ndarray, z: int, y: int, x: int, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size, size), dtype=np.float64)
    nz, ny, nx = volume.shape
    for dz in range(-radius, radius + 1):
        zz = z + dz
        if zz < 0 or zz >= nz:
            continue
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= ny:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if 0 <= xx < nx:
                    kernel[dz + radius, dy + radius, dx + radius] = volume[zz, yy, xx]
    return kernel


def _kernel_mask(roi: np.ndarray, z: int, y: int, x: int, radius: int, masked: bool) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size, size), dtype=bool)
    nz, ny, nx = roi.shape
    for dz in range(-radius, radius + 1):
        zz = z + dz
        if zz < 0 or zz >= nz:
            continue
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= ny:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if not (0 <= xx < nx):
                    continue
                if (not masked) or bool(roi[zz, yy, xx]):
                    kernel[dz + radius, dy + radius, dx + radius] = True
    return kernel


def _kernel_features(
    kernel, mask, spacing, mode, bin_width, classes
) -> list[tuple[str, str, float]]:
    found: list[tuple[str, str, float]] = []
    for class_name, names in classes:
        try:
            values = _extract_kernel(kernel, mask, spacing, mode, bin_width, [(class_name, names)])
        except RadixError:
            values = {}
            for feature in names:
                try:
                    one = _extract_kernel(
                        kernel, mask, spacing, mode, bin_width, [(class_name, [feature])]
                    )
                except RadixError:
                    continue
                values.update(one)
        for key, value in values.items():
            prefix, feature_class, feature = key.split("_", 2)
            if prefix != "original" or feature_class != class_name:
                continue
            found.append((class_name, feature, float(value)))
    return found


def _extract_kernel(kernel, mask, spacing, mode, bin_width, classes) -> dict[str, Any]:
    return extract_features(
        np.ascontiguousarray(kernel),
        np.ascontiguousarray(mask),
        tuple(float(value) for value in spacing),
        mode,
        bin_width,
        None,
        0.0,
        None,
        "linear",
        (0.0, 0.0, 0.0),
        None,
        None,
        None,
        None,
        False,
        1.0,
        0,
        False,
        "absolute",
        False,
        [("Original", [])],
        classes,
        [1],
        None,
        True,
        0,
        False,
        0,
    )
