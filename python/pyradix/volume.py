from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np

from pyradix.paths import native_path

PathLike = str | Path


@dataclass
class Volume:
    """3D image in (z, y, x) voxel order, float64."""

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    path: str | None = None
    integer_pixels: bool = False
    float32_pixels: bool = False

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array, dtype=np.float64)
        if self.array.ndim != 3:
            raise ValueError(f"expected a 3D volume, got shape {self.array.shape}")
        if len(self.spacing) != 3 or any(s <= 0 for s in self.spacing):
            raise ValueError("spacing must be three positive numbers (z, y, x)")
        if len(self.origin) != 3:
            raise ValueError("origin must be (x, y, z) in millimetres")
        if len(self.direction) not in {9, 0}:
            raise ValueError("direction must be 9 affine terms, or empty")

    def check_geometry(self, other: Volume, *, atol: float = 1e-5) -> None:
        if self.array.shape != other.array.shape:
            raise ValueError(
                f"image/mask shape mismatch: {self.array.shape} vs {other.array.shape}"
            )
        if not np.allclose(self.spacing, other.spacing, atol=atol, rtol=0):
            raise ValueError(f"image/mask spacing mismatch: {self.spacing} vs {other.spacing}")

    def geometry_matches(self, other: Volume, *, atol: float = 1e-5) -> bool:
        if self.array.shape != other.array.shape:
            return False
        if not np.allclose(self.spacing, other.spacing, atol=atol, rtol=0):
            return False
        if not np.allclose(self.origin, other.origin, atol=atol, rtol=0):
            return False
        if len(self.direction) == 9 and len(other.direction) == 9:
            return bool(np.allclose(self.direction, other.direction, atol=atol, rtol=0))
        return True


def voxel_axes(volume: Volume) -> np.ndarray:
    """Voxel-index (x, y, z) to world. Columns are the axis steps."""
    if len(volume.direction) == 9:
        return np.asarray(volume.direction, dtype=np.float64).reshape(3, 3)
    spacing_x, spacing_y, spacing_z = volume.spacing[2], volume.spacing[1], volume.spacing[0]
    return np.diag([spacing_x, spacing_y, spacing_z])


def resample_mask_nearest(image: Volume, mask: Volume) -> Volume:
    """Nearest-neighbor mask on the image grid, in physical space."""
    axes = voxel_axes(image)
    mask_axes = voxel_axes(mask)
    try:
        mask_inv = np.linalg.inv(mask_axes)
    except np.linalg.LinAlgError as exc:
        raise ValueError("mask direction is not invertible") from exc
    nz, ny, nx = image.array.shape
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float64)
    world = np.asarray(image.origin, dtype=np.float64) + points @ axes.T
    index = (world - np.asarray(mask.origin, dtype=np.float64)) @ mask_inv.T
    nearest = np.rint(index).astype(np.int64)
    mz, my, mx = mask.array.shape
    # mask array is (z, y, x); continuous index is (x, y, z)
    inside = (
        (nearest[:, 0] >= 0)
        & (nearest[:, 0] < mx)
        & (nearest[:, 1] >= 0)
        & (nearest[:, 1] < my)
        & (nearest[:, 2] >= 0)
        & (nearest[:, 2] < mz)
    )
    out = np.zeros(nz * ny * nx, dtype=np.float64)
    if np.any(inside):
        picked = nearest[inside]
        out[inside] = mask.array[picked[:, 2], picked[:, 1], picked[:, 0]]
    return Volume(
        array=out.reshape(nz, ny, nx),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        path=mask.path,
    )


def label_inside_image(image: Volume, mask: Volume, label: int, *, tolerance: float = 1e-3) -> bool:
    """True when every voxel of `label` falls inside the image index bounds."""
    coords = np.argwhere(mask.array == label)
    if coords.size == 0:
        return False
    mask_axes = voxel_axes(mask)
    image_axes = voxel_axes(image)
    try:
        image_inv = np.linalg.inv(image_axes)
    except np.linalg.LinAlgError:
        return False
    # coords are (z, y, x)
    points = np.stack([coords[:, 2], coords[:, 1], coords[:, 0]], axis=1).astype(np.float64)
    world = np.asarray(mask.origin, dtype=np.float64) + points @ mask_axes.T
    index = (world - np.asarray(image.origin, dtype=np.float64)) @ image_inv.T
    size = np.array(
        [image.array.shape[2], image.array.shape[1], image.array.shape[0]],
        dtype=np.float64,
    )
    lower = -0.5 - tolerance
    upper = size - 0.5 + tolerance
    return bool(np.all(index >= lower) and np.all(index <= upper))


def load_volume(path: PathLike) -> Volume:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return _load_nifti(path)
    if name.endswith(".nrrd") or name.endswith(".nhdr"):
        return _load_nrrd(path)
    raise ValueError(f"unsupported image format: {path}")


def _close_nibabel(img: nib.spatialimages.SpatialImage) -> None:
    """Drop memory maps so Windows can delete or replace the file."""
    try:
        img.uncache()
    except Exception:
        pass
    file_map = getattr(img, "file_map", None) or {}
    for handle in file_map.values():
        fileobj = getattr(handle, "fileobj", None)
        if fileobj is not None and not getattr(fileobj, "closed", True):
            fileobj.close()


def _load_nifti(path: Path) -> Volume:
    filename = native_path(path)
    img = nib.load(filename)
    try:
        source = np.asanyarray(img.dataobj)
        integer_pixels = bool(np.issubdtype(source.dtype, np.integer))
        float32_pixels = bool(source.dtype == np.dtype(np.float32))
        # Always copy. A float64 NIfTI stays a memmap when copy=False, and
        # Windows will not delete or overwrite that file while it is mapped.
        data = np.array(source, dtype=np.float64, copy=True)
        del source
        if data.ndim != 3:
            raise ValueError(f"{path} is not 3D (shape {data.shape})")
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        affine = np.array(img.affine, dtype=np.float64, copy=True)
    finally:
        _close_nibabel(img)
    # nibabel is (x, y, z); store (z, y, x) like SimpleITK/PyRadiomics.
    array = np.transpose(data, (2, 1, 0))
    spacing = (zooms[2], zooms[1], zooms[0])
    origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    direction = tuple(float(v) for v in affine[:3, :3].reshape(-1))
    return Volume(
        array=array,
        spacing=spacing,
        origin=origin,
        direction=direction,
        path=filename,
        integer_pixels=integer_pixels,
        float32_pixels=float32_pixels,
    )


def _load_nrrd(path: Path) -> Volume:
    filename = native_path(path)
    # A path (not a handle) so detached .nhdr data files resolve next to the
    # header. The array is copied so the reader can close the file.
    data, header = nrrd.read(filename)
    integer_pixels = bool(np.issubdtype(np.asarray(data).dtype, np.integer))
    float32_pixels = bool(np.asarray(data).dtype == np.dtype(np.float32))
    array = np.array(data, dtype=np.float64, copy=True)
    del data
    if array.ndim != 3:
        raise ValueError(f"{path} is not 3D (shape {array.shape})")
    array = np.transpose(array, (2, 1, 0))
    spacing = _nrrd_spacing(header)
    origin = _nrrd_origin(header)
    direction = _nrrd_direction(header)
    return Volume(
        array=array,
        spacing=spacing,
        origin=origin,
        direction=direction,
        path=filename,
        integer_pixels=integer_pixels,
        float32_pixels=float32_pixels,
    )


def _nrrd_spacing(header: dict) -> tuple[float, float, float]:
    directions = header.get("space directions")
    if directions is not None:
        lengths = []
        for axis in directions:
            vec = np.asarray(axis, dtype=np.float64).reshape(-1)
            lengths.append(float(np.linalg.norm(vec)))
        if len(lengths) >= 3 and all(v > 0 for v in lengths[:3]):
            # header axes are x, y, z → store z, y, x
            return (lengths[2], lengths[1], lengths[0])
    spacings = header.get("spacings")
    if spacings is not None and len(spacings) >= 3:
        return (float(spacings[2]), float(spacings[1]), float(spacings[0]))
    return (1.0, 1.0, 1.0)


def _nrrd_origin(header: dict) -> tuple[float, float, float]:
    origin = header.get("space origin")
    if origin is None:
        return (0.0, 0.0, 0.0)
    vals = [float(v) for v in origin]
    if len(vals) < 3:
        vals.extend([0.0] * (3 - len(vals)))
    return (vals[0], vals[1], vals[2])


def _nrrd_direction(header: dict) -> tuple[float, ...]:
    directions = header.get("space directions")
    if directions is None:
        return ()
    flat = np.asarray(directions, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in flat[:9])
