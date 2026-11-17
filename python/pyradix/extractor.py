from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pyradix._radix import RadixError, extract_features, parse_mode, resample_geometry
from pyradix.params import Params, ParamsError
from pyradix.volume import Volume, label_inside_image, load_volume, resample_mask_nearest

PathLike = str | Path


@dataclass(frozen=True)
class ExtractResult:
    """Feature table plus provenance."""

    values: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    ibsi_report: dict[str, Any] | None = None
    stability: dict[str, float] | None = None


class RadixExtractor:
    def __init__(self, mode: str = "ibsi", params: Params | Mapping[str, Any] | None = None):
        self.mode = parse_mode(mode)
        if params is None:
            self.params = Params(mode=self.mode)
        elif isinstance(params, Params):
            if params.mode != self.mode:
                raise ParamsError(
                    f"params.mode={params.mode!r} does not match extractor mode={self.mode!r}"
                )
            self.params = params
        else:
            self.params = Params.from_mapping(params, default_mode=self.mode)

    @classmethod
    def from_pyradiomics_yaml(cls, path: PathLike) -> RadixExtractor:
        params = Params.from_pyradiomics_yaml(path)
        return cls(mode="pyradiomics", params=params)

    def extract(
        self,
        image: PathLike | Volume,
        mask: PathLike | Volume,
        label: int | None = None,
    ) -> ExtractResult:
        image_vol = image if isinstance(image, Volume) else load_volume(image)
        mask_vol = mask if isinstance(mask, Volume) else load_volume(mask)
        use_label = self.params.label if label is None else int(label)
        mask_vol = self._align_mask(image_vol, mask_vol, use_label)
        roi = mask_vol.array == use_label
        if not np.any(roi):
            _raise_case(_missing_label_message(mask_vol.array, use_label))
        _check_roi_size(roi, self.params.minimum_roi_dimensions, self.params.minimum_roi_size)
        image_c = np.ascontiguousarray(image_vol.array, dtype=np.float64)
        mask_c = np.ascontiguousarray(roi, dtype=bool)
        origin_zyx = (
            float(image_vol.origin[2]),
            float(image_vol.origin[1]),
            float(image_vol.origin[0]),
        )
        resample_spacing = _resolved_spacing(self.params.resampled_pixel_spacing, image_vol.spacing)
        values = extract_features(
            image_c,
            mask_c,
            tuple(float(s) for s in image_vol.spacing),
            self.mode,
            self.params.bin_width,
            None if self.params.bin_count is None else int(self.params.bin_count),
            float(self.params.voxel_array_shift),
            resample_spacing,
            self.params.interpolator,
            origin_zyx,
            self.params.mask_interpolator,
            self.params.resegment_range,
            self.params.outlier_sigma,
            self.params.bin_min,
            bool(self.params.normalize),
            float(self.params.normalize_scale),
            int(self.params.pad_distance),
            bool(self.params.pre_crop),
            self.params.resegment_mode,
            bool(self.params.resegment_shape),
            _image_type_args(
                self.params, image_vol.integer_pixels, image_vol.float32_pixels
            ),
            _feature_class_args(self.params.enabled_features()),
            [int(distance) for distance in self.params.distances],
            self.params.weighting_norm,
            bool(self.params.symmetrical_glcm),
            int(self.params.gldm_a),
            bool(self.params.force_2d),
            int(self.params.force_2d_dimension),
            bool(
                image_vol.integer_pixels
                or np.issubdtype(np.asarray(image_vol.array).dtype, np.integer)
            ),
        )
        extra: dict[str, Any] = {
            "image_shape": list(image_vol.array.shape),
            "mask_shape": list(mask_vol.array.shape),
            "spacing": list(image_vol.spacing),
            "origin": list(image_vol.origin),
            "label": use_label,
            "roi_voxels": int(np.count_nonzero(roi)),
            "n_features": len(values),
            "interpolator": self.params.interpolator,
        }
        interpolated = None
        if resample_spacing is not None:
            dim, spacing, origin = resample_geometry(
                tuple(int(s) for s in image_vol.array.shape),
                tuple(float(s) for s in image_vol.spacing),
                origin_zyx,
                self.mode,
                resample_spacing,
            )
            extra["resampled_shape"] = list(dim)
            extra["resampled_spacing"] = list(spacing)
            extra["resampled_origin_zyx"] = list(origin)
            interpolated = (dim, spacing)
        provenance = self.params.provenance(extra=extra)
        if self.params.additional_info:
            from pyradix.clear import diagnostics_map, resegmented_roi

            provenance["diagnostics"] = diagnostics_map(
                image_c,
                mask_c,
                tuple(float(s) for s in image_vol.spacing),
                tuple(float(v) for v in image_vol.origin),
                tuple(float(v) for v in image_vol.direction),
                _diagnostics_settings(self.params, use_label),
                {name: dict(settings) for name, settings in self.params.image_types},
                interpolated=interpolated,
                resegmented=resegmented_roi(
                    image_c,
                    mask_c,
                    self.params.resegment_range,
                    self.params.resegment_mode,
                    self.params.outlier_sigma,
                ),
            )
        return ExtractResult(values=dict(values), provenance=provenance)

    def _align_mask(self, image_vol: Volume, mask_vol: Volume, label: int) -> Volume:
        if self.params.geometry_tolerance is None:
            tolerance = 1e-5
        else:
            tolerance = float(self.params.geometry_tolerance)
        if image_vol.geometry_matches(mask_vol, atol=tolerance):
            return mask_vol
        detail = _geometry_detail(image_vol, mask_vol)
        if not self.params.correct_mask:
            _raise_case(
                "mask: image and mask geometry disagree and correctMask is false; " + detail
            )
        if not np.any(mask_vol.array == label):
            _raise_case(_missing_label_message(mask_vol.array, label))
        if not label_inside_image(image_vol, mask_vol, label):
            _raise_case("mask: ROI outside the image after mask correction; " + detail)
        return resample_mask_nearest(image_vol, mask_vol)


def _diagnostics_settings(params: Params, label: int) -> dict[str, Any]:
    spacing = params.resampled_pixel_spacing
    resegment = params.resegment_range
    return {
        "binWidth": params.bin_width,
        "binCount": params.bin_count,
        "label": label,
        "resampledPixelSpacing": None if spacing is None else list(spacing),
        "interpolator": params.interpolator,
        "resegmentRange": None if resegment is None else list(resegment),
        "resegmentMode": params.resegment_mode,
        "removeOutliers": params.outlier_sigma,
        "normalize": params.normalize,
        "additionalInfo": params.additional_info,
    }


def _setting_text(value: Any) -> str:
    """Stable setting text. ``repr(float)`` does not follow the numeric locale."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_setting_text(item) for item in value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _image_type_args(
    params: Params, integer_pixels: bool = False, float32_pixels: bool = False
) -> list[tuple[str, list[tuple[str, str]]]]:
    rows: list[tuple[str, list[tuple[str, str]]]] = []
    for name, settings in params.image_types:
        merged = dict(settings)
        if name == "LoG" and "sigma" not in merged and params.sigma is not None:
            merged["sigma"] = list(params.sigma)
        if name == "Wavelet":
            if "wavelet" not in merged and params.wavelet:
                merged["wavelet"] = params.wavelet
            if "start_level" not in merged and params.start_level is not None:
                merged["start_level"] = params.start_level
            if "level" not in merged and params.level is not None:
                merged["level"] = params.level
        if (
            name == "Gradient"
            and "gradientUseSpacing" not in merged
            and params.gradient_use_spacing is not None
        ):
            merged["gradientUseSpacing"] = params.gradient_use_spacing
        if name == "LBP2D":
            if "lbp2DRadius" not in merged and params.lbp2d_radius is not None:
                merged["lbp2DRadius"] = params.lbp2d_radius
            if "lbp2DSamples" not in merged and params.lbp2d_samples is not None:
                merged["lbp2DSamples"] = params.lbp2d_samples
            if "lbp2DMethod" not in merged and params.lbp2d_method is not None:
                merged["lbp2DMethod"] = params.lbp2d_method
            if "force2Ddimension" not in merged:
                merged["force2Ddimension"] = params.force_2d_dimension
        if name == "LBP3D":
            if "lbp3DLevels" not in merged and params.lbp3d_levels is not None:
                merged["lbp3DLevels"] = params.lbp3d_levels
            if "lbp3DIcosphereRadius" not in merged and params.lbp3d_icosphere_radius is not None:
                merged["lbp3DIcosphereRadius"] = params.lbp3d_icosphere_radius
            if (
                "lbp3DIcosphereSubdivision" not in merged
                and params.lbp3d_icosphere_subdivision is not None
            ):
                merged["lbp3DIcosphereSubdivision"] = params.lbp3d_icosphere_subdivision
            if integer_pixels:
                merged["integerVoxels"] = True
            elif float32_pixels:
                merged["float32Samples"] = True
        rows.append((name, [(str(key), _setting_text(value)) for key, value in merged.items()]))
    return rows


def _feature_class_args(enabled: dict[str, tuple[str, ...]]) -> list[tuple[str, list[str]]]:
    return [(name, list(features)) for name, features in enabled.items()]


def _resolved_spacing(requested, original):
    if requested is None:
        return None
    original = tuple(float(s) for s in original)
    return tuple(
        original[i] if float(requested[i]) == 0.0 else float(requested[i]) for i in range(3)
    )


def _raise_case(message: str) -> None:
    err = RadixError(message)
    err.code = "case"
    err.message = message
    raise err


def _missing_label_message(mask: np.ndarray, label: int) -> str:
    present = sorted({int(v) for v in np.unique(mask) if int(v) != 0})
    return f"mask: label {label} is not in the mask; present labels: {present}"


def _geometry_detail(image: Volume, mask: Volume) -> str:
    return (
        f"image shape {list(image.array.shape)} spacing {list(image.spacing)}; "
        f"mask shape {list(mask.array.shape)} spacing {list(mask.spacing)}"
    )


def _check_roi_size(roi: np.ndarray, minimum_dimensions: int, minimum_size: int | None) -> None:
    coords = np.argwhere(roi)
    count = int(coords.shape[0])
    sizes = coords.max(axis=0) - coords.min(axis=0) + 1
    dimensions = int(np.sum(sizes > 1))
    if dimensions < minimum_dimensions:
        _raise_case(
            "mask: ROI has "
            f"{dimensions} dimension(s), minimumROIDimensions is {minimum_dimensions}, "
            f"ROI size {count}"
        )
    if minimum_size is not None and count <= int(minimum_size):
        _raise_case(f"mask: ROI size {count} is below minimumROISize {int(minimum_size)}")
