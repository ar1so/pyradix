"""Type stubs for the Rust extension ``pyradix._radix``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class RadixError(Exception):
    code: str
    message: str

def engine_info() -> str: ...
def parse_mode(mode: str) -> str: ...
def legal_image_types(mode: str) -> list[str]: ...
def legal_image_types_all() -> list[str]: ...
def legal_feature_classes() -> list[str]: ...
def feature_catalog() -> dict[str, dict[str, list[str]]]: ...
def resample_geometry(
    dim: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    mode: str,
    out_spacing: tuple[float, float, float],
) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]: ...
def extract_features(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: tuple[float, float, float],
    mode: str,
    bin_width: float | None = ...,
    bin_count: int | None = ...,
    voxel_array_shift: float = ...,
    resample_spacing: tuple[float, float, float] | None = ...,
    interpolator: str = ...,
    origin: tuple[float, float, float] | None = ...,
    mask_interpolator: str | None = ...,
    resegment_range: tuple[float, float] | None = ...,
    outlier_sigma: float | None = ...,
    bin_min: float | None = ...,
    normalize: bool = ...,
    normalize_scale: float = ...,
    pad_distance: int = ...,
    pre_crop: bool = ...,
    resegment_mode: str = ...,
    resegment_shape: bool = ...,
    image_types: list[tuple[str, list[tuple[str, str]]]] | None = ...,
    feature_classes: list[tuple[str, list[str]]] | None = ...,
    distances: list[int] | None = ...,
    weighting_norm: str | None = ...,
    symmetrical_glcm: bool = ...,
    gldm_a: int = ...,
    force_2d: bool = ...,
    force_2d_dimension: int = ...,
    integer_pixels: bool | None = ...
) -> dict[str, float]: ...
def apply_image_type(
    name: str,
    mode: str,
    image: NDArray[np.float64],
    spacing: tuple[float, float, float],
    settings: list[tuple[str, str]],
) -> list[tuple[str, NDArray[np.float64]]]: ...
def apply_mean(
    image: NDArray[np.float64],
    support: int,
    padding: str = ...,
    ndims: int = ...,
) -> NDArray[np.float64]: ...
def apply_log(
    image: NDArray[np.float64],
    sigma_mm: float,
    spacing: tuple[float, float, float],
    padding: str = ...,
    ndims: int = ...,
    cutoff: float = ...,
) -> NDArray[np.float64]: ...
def apply_laws(
    image: NDArray[np.float64],
    kernels: str,
    padding: str = ...,
    rotation_invariant: bool = ...,
    energy_delta: int | None = ...,
) -> NDArray[np.float64]: ...
def apply_gabor(
    image: NDArray[np.float64],
    sigma_mm: float,
    lambda_mm: float,
    gamma: float,
    spacing: tuple[float, float, float],
    theta: float = ...,
    padding: str = ...,
    rotation_invariant: bool = ...,
    delta_theta: float | None = ...,
    orthogonal_planes: bool = ...,
) -> NDArray[np.float64]: ...
def apply_wavelet(
    image: NDArray[np.float64],
    family: str,
    combo: str,
    level: int = ...,
    padding: str = ...,
    rotation_invariant: bool = ...,
) -> NDArray[np.float64]: ...

__lib_version__: str
