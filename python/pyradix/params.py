from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from pyradix._radix import (
    RadixError,
    feature_catalog,
    legal_image_types,
    legal_image_types_all,
    parse_mode,
)

_INTERPOLATORS = frozenset(
    {
        "nearest",
        "linear",
        "bspline",
        "sitkNearestNeighbor",
        "sitkLinear",
        "sitkBSpline",
    }
)
_WEIGHTING = frozenset({"euclidean", "manhattan", "infinity", "no_weighting"})
_RESEGMENT_MODES = frozenset({"absolute", "relative", "sigma"})
_LBP_METHODS = frozenset({"default", "ror", "uniform", "var"})
_TYPE_OVERRIDE_BLOCKED = frozenset({"interpolator", "resampledPixelSpacing", "padDistance"})
_FILTER_SETTING_KEYS = frozenset({"support", "padding", "kernel", "lambda", "gamma", "theta"})

# PyRadiomics setting names that belong to the global setting block.
_PYRAD_SETTING_MAP = {
    "binWidth": "bin_width",
    "binCount": "bin_count",
    "resampledPixelSpacing": "resampled_pixel_spacing",
    "interpolator": "interpolator",
    "label": "label",
    "label_channel": "label_channel",
    "voxelArrayShift": "voxel_array_shift",
    "resegmentRange": "resegment_range",
    "removeOutliers": "outlier_sigma",
    "normalize": "normalize",
    "normalizeScale": "normalize_scale",
    "padDistance": "pad_distance",
    "preCrop": "pre_crop",
    "resegmentMode": "resegment_mode",
    "resegmentShape": "resegment_shape",
    "minimumROIDimensions": "minimum_roi_dimensions",
    "minimumROISize": "minimum_roi_size",
    "correctMask": "correct_mask",
    "force2D": "force_2d",
    "force2Ddimension": "force_2d_dimension",
    "distances": "distances",
    "weightingNorm": "weighting_norm",
    "symmetricalGLCM": "symmetrical_glcm",
    "gldm_a": "gldm_a",
    "additionalInfo": "additional_info",
    "geometryTolerance": "geometry_tolerance",
    "sigma": "sigma",
    "wavelet": "wavelet",
    "start_level": "start_level",
    "level": "level",
    "gradientUseSpacing": "gradient_use_spacing",
    "lbp2DRadius": "lbp2d_radius",
    "lbp2DSamples": "lbp2d_samples",
    "lbp2DMethod": "lbp2d_method",
    "lbp3DLevels": "lbp3d_levels",
    "lbp3DIcosphereRadius": "lbp3d_icosphere_radius",
    "lbp3DIcosphereSubdivision": "lbp3d_icosphere_subdivision",
}
_VOXEL_DEFAULTS = {
    "kernelRadius": 1,
    "maskedKernel": True,
    "initValue": 0.0,
}
_VOXEL_KEYS = frozenset({*_VOXEL_DEFAULTS, "voxelBatch"})
_INTERNAL_FIELDS = frozenset(_PYRAD_SETTING_MAP.values()) | {
    "mask_interpolator",
    "image_types",
    "feature_classes",
    "voxel_setting",
    "voxel_batch",
    "bin_min",
    "extra",
    "load_problems",
}


class ParamsError(ValueError):
    """The parameter file could not be read."""


def _raise_settings(problems: list[str]) -> None:
    if not problems:
        return
    message = "; ".join(problems)
    err = RadixError(message)
    err.code = "setting"
    err.message = message
    raise err


@dataclass
class Params:
    mode: str = "ibsi"
    bin_width: float | None = 25.0
    bin_count: int | None = None
    resampled_pixel_spacing: tuple[float, float, float] | None = None
    interpolator: str = "bspline"
    mask_interpolator: str | None = None
    image_types: tuple[Any, ...] = (("Original", {}),)
    label: int = 1
    label_channel: int | None = None
    voxel_batch: int | None = None
    voxel_array_shift: float = 0.0
    resegment_range: tuple[float, float] | None = None
    outlier_sigma: float | None = None
    bin_min: float | None = None
    normalize: bool = False
    normalize_scale: float = 1.0
    pad_distance: int = 5
    pre_crop: bool = False
    resegment_mode: str = "absolute"
    resegment_shape: bool = False
    minimum_roi_dimensions: int = 2
    minimum_roi_size: int | None = None
    correct_mask: bool = False
    force_2d: bool = False
    force_2d_dimension: int = 0
    distances: tuple[int, ...] = (1,)
    weighting_norm: str | None = None
    symmetrical_glcm: bool = True
    gldm_a: int = 0
    additional_info: bool = True
    geometry_tolerance: float | None = None
    sigma: tuple[float, ...] | None = None
    wavelet: str | None = None
    start_level: int | None = None
    level: int | None = None
    gradient_use_spacing: bool | None = None
    lbp2d_radius: float | None = None
    lbp2d_samples: int | None = None
    lbp2d_method: str | None = None
    lbp3d_levels: int | None = None
    lbp3d_icosphere_radius: float | None = None
    lbp3d_icosphere_subdivision: int | None = None
    feature_classes: dict[str, tuple[str, ...]] | None = None
    voxel_setting: dict[str, Any] = field(default_factory=lambda: dict(_VOXEL_DEFAULTS))
    extra: dict[str, Any] = field(default_factory=dict)
    load_problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = list(self.load_problems)
        self.load_problems = ()
        try:
            self.mode = parse_mode(self.mode)
        except ValueError as exc:
            problems.append(str(exc))
        self.image_types = _normalize_image_types(self.image_types, problems)
        self.voxel_setting = _normalize_voxel_setting(self.voxel_setting, problems)
        if self.voxel_setting.get("voxelBatch") is not None and self.voxel_batch is None:
            self.voxel_batch = self.voxel_setting["voxelBatch"]
        problems.extend(_value_problems(self))
        problems.extend(_image_type_problems(self))
        problems.extend(_feature_class_problems(self))
        _raise_settings(problems)

    def enabled_features(self) -> dict[str, tuple[str, ...]]:
        """Class name to the feature names this parameter set turns on."""
        if self.feature_classes is not None:
            return {name: tuple(values) for name, values in self.feature_classes.items()}
        catalog = feature_catalog()
        return {
            name: tuple(
                feature
                for feature in entry["features"]
                if feature not in set(entry["deprecated"])
            )
            for name, entry in catalog.items()
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_mode: str = "ibsi") -> Params:
        payload = dict(data)
        mode = str(payload.pop("mode", default_mode))
        fields, problems = _coerce_fields(payload)
        fields["load_problems"] = tuple(problems)
        return cls(mode=mode, **fields)

    @classmethod
    def from_yaml(cls, path: str | Path, default_mode: str = "ibsi", **overrides: Any) -> Params:
        raw = _read_yaml(path)
        if not isinstance(raw, dict):
            raise ParamsError("YAML root must be a mapping")
        raw = dict(raw)
        raw.update(overrides)
        return cls.from_mapping(raw, default_mode=default_mode)

    @classmethod
    def from_pyradiomics_file(cls, path: str | Path, **overrides: Any) -> Params:
        raw = _read_document(path)
        if not isinstance(raw, dict):
            raise ParamsError("parameter file root must be a mapping")
        mapped, problems = _translate_pyradiomics(raw)
        for key, value in overrides.items():
            if key == "mode":
                continue
            if key not in _INTERNAL_FIELDS:
                problems.append(f"unknown param keys: ['{key}']")
                continue
            mapped[key] = value
        mapped["load_problems"] = tuple(problems)
        return cls(mode="pyradiomics", **mapped)

    @classmethod
    def from_pyradiomics_yaml(cls, path: str | Path, **overrides: Any) -> Params:
        return cls.from_pyradiomics_file(path, **overrides)

    @classmethod
    def from_pyradiomics_json(cls, path: str | Path, **overrides: Any) -> Params:
        return cls.from_pyradiomics_file(path, **overrides)

    def provenance(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "mode": self.mode,
            "bin_width": self.bin_width,
            "bin_count": self.bin_count,
            "resampled_pixel_spacing": self.resampled_pixel_spacing,
            "interpolator": self.interpolator,
            "mask_interpolator": self.mask_interpolator,
            "image_types": [
                {"name": name, "settings": settings} for name, settings in self.image_types
            ],
            "feature_classes": None
            if self.feature_classes is None
            else {name: list(values) for name, values in self.feature_classes.items()},
            "voxel_setting": dict(self.voxel_setting),
            "label": self.label,
            "voxel_batch": self.voxel_batch,
            "voxel_array_shift": self.voxel_array_shift,
            "resegment_range": self.resegment_range,
            "resegment_mode": self.resegment_mode,
            "resegment_shape": self.resegment_shape,
            "outlier_sigma": self.outlier_sigma,
            "bin_min": self.bin_min,
            "normalize": self.normalize,
            "normalize_scale": self.normalize_scale,
            "pad_distance": self.pad_distance,
            "sigma": None if self.sigma is None else list(self.sigma),
            "additional_info": self.additional_info,
        }
        if extra:
            body.update(dict(extra))
        encoded = json.dumps(_plain(body), sort_keys=True, default=str, allow_nan=True).encode(
            "utf-8"
        )
        body["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return body


def _catalog() -> dict[str, dict[str, list[str]]]:
    return feature_catalog()


def _normalize_image_types(
    value: Any, problems: list[str]
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if value is None:
        problems.append("image_types must not be empty")
        return ()
    if isinstance(value, str):
        value = (value,)
    out: list[tuple[str, dict[str, Any]]] = []
    try:
        entries = list(value)
    except TypeError:
        problems.append("imageType must be a mapping or list")
        return ()
    if not entries:
        problems.append("image_types must not be empty")
        return ()
    for entry in entries:
        if isinstance(entry, str):
            out.append((entry, {}))
            continue
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], Mapping)
        ):
            out.append((entry[0], dict(entry[1])))
            continue
        problems.append(f"image type entry {entry!r} must be a name or (name, settings)")
    return tuple(out)


def _normalize_voxel_setting(value: Any, problems: list[str]) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        problems.append("voxelSetting must be a mapping")
        return dict(_VOXEL_DEFAULTS)
    out = dict(_VOXEL_DEFAULTS)
    for key, item in value.items():
        if key not in _VOXEL_KEYS:
            problems.append(f"unknown voxelSetting: {key}")
            continue
        out[key] = item
    batch = out.get("voxelBatch")
    if batch is not None and (not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0):
        problems.append("setting voxelBatch: voxelBatch must be > 0")
    radius = out.get("kernelRadius")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius <= 0:
        problems.append("setting kernelRadius: kernelRadius must be > 0")
    if not isinstance(out.get("maskedKernel"), bool):
        problems.append("setting maskedKernel: maskedKernel must be true or false")
    init = out.get("initValue")
    if isinstance(init, bool) or not isinstance(init, (int, float)):
        problems.append("setting initValue: initValue must be a number")
    return out


def _value_problems(params: Params) -> list[str]:
    problems: list[str] = []
    if params.bin_width is not None and (
        isinstance(params.bin_width, bool)
        or not isinstance(params.bin_width, (int, float))
        or params.bin_width <= 0
    ):
        problems.append("setting binWidth: binWidth must be > 0")
    if params.bin_count is not None and (
        isinstance(params.bin_count, bool)
        or not isinstance(params.bin_count, int)
        or params.bin_count <= 0
    ):
        problems.append("setting binCount: binCount must be > 0")
    if params.bin_width is None and params.bin_count is None:
        problems.append("setting binWidth: set binWidth or binCount")
    if isinstance(params.label, bool) or not isinstance(params.label, int) or params.label <= 0:
        problems.append("setting label: label must be a positive integer")
    if params.voxel_batch is not None and (
        isinstance(params.voxel_batch, bool)
        or not isinstance(params.voxel_batch, int)
        or params.voxel_batch <= 0
    ):
        problems.append("setting voxelBatch: voxelBatch must be > 0")
    if params.interpolator not in _INTERPOLATORS:
        problems.append(f"unknown interpolator: {params.interpolator}")
    if params.mask_interpolator is not None and params.mask_interpolator not in _INTERPOLATORS:
        problems.append(f"unknown mask interpolator: {params.mask_interpolator}")
    if params.outlier_sigma is not None and (
        isinstance(params.outlier_sigma, bool)
        or not isinstance(params.outlier_sigma, (int, float))
        or params.outlier_sigma <= 0
    ):
        problems.append("setting removeOutliers: removeOutliers must be > 0")
    spacing = params.resampled_pixel_spacing
    if spacing is not None:
        values = tuple(spacing)
        bad_spacing = len(values) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in values
        )
        if bad_spacing:
            problems.append(
                "setting resampledPixelSpacing: resampledPixelSpacing must be three numbers "
                "and cannot be negative"
            )
        else:
            params.resampled_pixel_spacing = (float(values[0]), float(values[1]), float(values[2]))
    if params.resegment_range is not None:
        rng = tuple(params.resegment_range)
        if len(rng) != 2:
            problems.append("setting resegmentRange: resegmentRange must be [min, max]")
        else:
            lo, hi = float(rng[0]), float(rng[1])
            if lo > hi:
                problems.append("setting resegmentRange: resegmentRange minimum must be <= maximum")
            else:
                params.resegment_range = (lo, hi)
    if params.resegment_mode not in _RESEGMENT_MODES:
        problems.append(f"unknown resegmentMode: {params.resegment_mode}")
    if params.minimum_roi_dimensions not in (1, 2, 3):
        problems.append("setting minimumROIDimensions: minimumROIDimensions must be 1, 2, or 3")
    if params.minimum_roi_size is not None and (
        isinstance(params.minimum_roi_size, bool)
        or not isinstance(params.minimum_roi_size, int)
        or params.minimum_roi_size <= 0
    ):
        problems.append("setting minimumROISize: minimumROISize must be > 0")
    if params.force_2d_dimension not in (0, 1, 2):
        problems.append("setting force2Ddimension: force2Ddimension must be 0, 1, or 2")
    if params.pad_distance < 0:
        problems.append("setting padDistance: padDistance must be >= 0")
    if (
        isinstance(params.normalize_scale, bool)
        or not isinstance(params.normalize_scale, (int, float))
        or params.normalize_scale <= 0
    ):
        problems.append("setting normalizeScale: normalizeScale must be > 0")
    if params.weighting_norm is not None and params.weighting_norm not in _WEIGHTING:
        problems.append(f"unknown weightingNorm: {params.weighting_norm}")
    if isinstance(params.gldm_a, bool) or not isinstance(params.gldm_a, int) or params.gldm_a < 0:
        problems.append("setting gldm_a: gldm_a must be >= 0")
    distances = params.distances
    if not isinstance(distances, tuple):
        distances = tuple(distances)
    if not distances or any(
        isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in distances
    ):
        problems.append("setting distances: distances must be positive integers")
    else:
        params.distances = distances
    if params.sigma is not None:
        sigma = _positive_floats(params.sigma)
        if sigma is None:
            problems.append("setting sigma: sigma must be a list of numbers > 0")
        else:
            params.sigma = sigma
    if params.lbp2d_method is not None and params.lbp2d_method not in _LBP_METHODS:
        problems.append(f"unknown lbp2DMethod: {params.lbp2d_method}")
    return problems


def _image_type_problems(params: Params) -> list[str]:
    problems: list[str] = []
    known = set(legal_image_types_all())
    try:
        allowed = set(legal_image_types(params.mode))
    except ValueError:
        allowed = set()
    unknown = [name for name, _ in params.image_types if name not in known]
    illegal = [name for name, _ in params.image_types if name in known and name not in allowed]
    if unknown:
        problems.append(f"unknown image types: {unknown}")
    if illegal:
        problems.append(f"image types not available in {params.mode} mode: {illegal}")
    for name, settings in params.image_types:
        for key in settings:
            if key in _TYPE_OVERRIDE_BLOCKED:
                problems.append(f"image type {name}: setting {key} cannot be set on an image type")
            elif key not in _PYRAD_SETTING_MAP and key not in _FILTER_SETTING_KEYS:
                problems.append(f"image type {name}: unknown setting {key}")
        if params.mode == "pyradiomics" and name == "LoG":
            sigma = settings["sigma"] if "sigma" in settings else params.sigma
            if _positive_floats(sigma) is None:
                problems.append("image type LoG: sigma is required in pyradiomics mode")
    return problems


def _feature_class_problems(params: Params) -> list[str]:
    raw = params.feature_classes
    if raw is None:
        legacy = params.extra.get("featureClass") if isinstance(params.extra, Mapping) else None
        if legacy is None:
            return []
        raw = legacy
    if not isinstance(raw, Mapping):
        return ["featureClass must be a mapping"]
    catalog = _catalog()
    problems: list[str] = []
    enabled: dict[str, tuple[str, ...]] = {}
    unknown = sorted(name for name in raw if name not in catalog)
    if unknown:
        problems.append(f"unknown feature classes: {unknown}")
    for name, requested in raw.items():
        if name not in catalog:
            continue
        entry = catalog[name]
        legal = list(entry["features"])
        deprecated = set(entry["deprecated"])
        if requested is None or (isinstance(requested, (list, tuple)) and len(requested) == 0):
            enabled[name] = tuple(feature for feature in legal if feature not in deprecated)
            continue
        if isinstance(requested, str) or not isinstance(requested, (list, tuple)):
            problems.append(f"feature class {name}: feature list must be a list of names")
            continue
        bad = [feature for feature in requested if feature not in legal]
        if bad:
            problems.append(f"feature class {name}: unknown features {bad}")
            continue
        enabled[name] = tuple(str(feature) for feature in requested)
    if not problems:
        params.feature_classes = enabled
    return problems


def _positive_floats(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not numbers or any(item <= 0 for item in numbers):
        return None
    return numbers


def _translate_pyradiomics(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    mapped: dict[str, Any] = {}
    setting = raw.get("setting") or raw.get("settings") or {}
    if setting and not isinstance(setting, dict):
        problems.append("setting must be a mapping")
        setting = {}
    for key, value in dict(setting).items():
        if key not in _PYRAD_SETTING_MAP:
            problems.append(f"unknown PyRadiomics setting: {key}")
            continue
        mapped[_PYRAD_SETTING_MAP[key]] = value
    spacing = mapped.get("resampled_pixel_spacing")
    if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
        # PyRadiomics (SimpleITK) lists spacing as x, y, z; Params keeps z, y, x.
        mapped["resampled_pixel_spacing"] = tuple(reversed(spacing))
    if "sigma" in mapped and mapped["sigma"] is not None:
        mapped["sigma"] = _positive_floats(mapped["sigma"]) or mapped["sigma"]
    image_type = raw.get("imageType", raw.get("image_types", None))
    if image_type is None:
        mapped["image_types"] = (("Original", {}),)
    elif isinstance(image_type, dict):
        pairs = []
        for name, settings in image_type.items():
            if settings is None:
                settings = {}
            if not isinstance(settings, dict):
                problems.append(f"image type {name}: settings must be a mapping")
                settings = {}
            pairs.append((str(name), dict(settings)))
        mapped["image_types"] = tuple(pairs) or (("Original", {}),)
    elif isinstance(image_type, list):
        mapped["image_types"] = tuple(image_type)
    else:
        problems.append("imageType must be a mapping or list")
    if "featureClass" in raw:
        feature_class = raw["featureClass"]
        if feature_class is None:
            mapped["feature_classes"] = None
        elif isinstance(feature_class, dict):
            mapped["feature_classes"] = {
                str(name): (
                    ()
                    if values is None
                    else tuple(values)
                    if isinstance(values, (list, tuple))
                    else values
                )
                for name, values in feature_class.items()
            }
        else:
            problems.append("featureClass must be a mapping")
    voxel = raw.get("voxelSetting")
    if voxel is not None:
        mapped["voxel_setting"] = voxel
    known_roots = {
        "setting",
        "settings",
        "imageType",
        "image_types",
        "featureClass",
        "voxelSetting",
    }
    extra_keys = sorted(set(raw) - known_roots)
    if extra_keys:
        problems.append(f"unknown PyRadiomics YAML keys: {extra_keys}")
    return mapped, problems


def _read_document(path: str | Path) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return _read_json(path)
    return _read_yaml(path)


def _read_text(path: str | Path) -> str:
    """UTF-8 text with a Windows BOM and CR LF normalized away."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _plain(value: Any) -> Any:
    """Turn NumPy scalars and arrays into JSON values.

    ``json.dumps`` would otherwise call ``default=str`` and the hash would
    depend on ``str(np.float64(...))``. Lists and dicts of Python numbers are
    left as they are, so an existing provenance hash does not move.
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, (str, bytes)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and hasattr(value, "dtype"):
        return _plain(tolist())
    return value


def _read_yaml(path: str | Path) -> Any:
    text = _read_text(path)
    if "\t" in text:
        raise ParamsError("YAML must use spaces, not tabs")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParamsError(f"invalid YAML: {exc}") from exc


def _read_json(path: str | Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParamsError(f"invalid JSON: {exc}") from exc


def _coerce_fields(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    unknown = [key for key in payload if key not in _INTERNAL_FIELDS]
    problems = [f"unknown param keys: {unknown}"] if unknown else []
    out = {key: payload[key] for key in payload if key in _INTERNAL_FIELDS}
    if "image_types" in out and isinstance(out["image_types"], list):
        out["image_types"] = tuple(out["image_types"])
    if "resampled_pixel_spacing" in out and out["resampled_pixel_spacing"] is not None:
        out["resampled_pixel_spacing"] = tuple(out["resampled_pixel_spacing"])
    if "resegment_range" in out and out["resegment_range"] is not None:
        out["resegment_range"] = tuple(out["resegment_range"])
    if "distances" in out and out["distances"] is not None:
        out["distances"] = tuple(out["distances"])
    if "sigma" in out and out["sigma"] is not None and not isinstance(out["sigma"], tuple):
        try:
            out["sigma"] = tuple(out["sigma"])
        except TypeError:
            pass
    return out, problems
