"""One full-signature extract. The parent test runs this in a separate process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FULL_TYPES = (
    "Original",
    "LoG",
    "Wavelet",
    "Square",
    "SquareRoot",
    "Logarithm",
    "Exponential",
    "Gradient",
    "LBP2D",
    "LBP3D",
)
LOG_SIGMA = [1.0, 2.0, 3.0, 4.0, 5.0]


def _numeric(raw: dict) -> dict[str, float]:
    values = {}
    for key, value in raw.items():
        if str(key).startswith("diagnostics_"):
            continue
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _radix() -> dict[str, float]:
    from pyradix import Params, RadixExtractor

    grid = os.environ["COMPARE_GRID"]
    if grid == "resample_2mm":
        image_types = (("Original", {}),)
        spacing = (2.0, 2.0, 2.0)
    else:
        image_types = tuple(
            (name, {"sigma": LOG_SIGMA} if name == "LoG" else {}) for name in FULL_TYPES
        )
        spacing = None
    params = Params(
        mode="pyradiomics",
        bin_width=25.0,
        interpolator="sitkBSpline",
        resampled_pixel_spacing=spacing,
        image_types=image_types,
        additional_info=False,
    )
    result = RadixExtractor(mode="pyradiomics", params=params).extract(
        os.environ["COMPARE_IMAGE"],
        os.environ["COMPARE_MASK"],
    )
    return _numeric(result.values)


def _pyradiomics() -> dict[str, float]:
    import SimpleITK as sitk
    from radiomics import featureextractor

    grid = os.environ["COMPARE_GRID"]
    settings = {
        "binWidth": 25,
        "label": 1,
        "interpolator": "sitkBSpline",
        "additionalInfo": False,
    }
    if grid == "resample_2mm":
        settings["resampledPixelSpacing"] = [2.0, 2.0, 2.0]
        names = ("Original",)
    else:
        names = FULL_TYPES
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    for name in names:
        if name == "LoG":
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": LOG_SIGMA})
        else:
            extractor.enableImageTypeByName(name)
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from image_paths import image_io_path

    image = sitk.ReadImage(image_io_path(os.environ["COMPARE_IMAGE"]))
    mask = sitk.ReadImage(image_io_path(os.environ["COMPARE_MASK"]))
    return _numeric(extractor.execute(image, mask))


def main() -> int:
    engine = os.environ["COMPARE_ENGINE"]
    values = _radix() if engine == "radix" else _pyradiomics()
    json.dump(values, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
