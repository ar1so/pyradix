"""PyRadiomics-mode resampling details that SimpleITK decides."""

from __future__ import annotations

import numpy as np

from pyradix import Params, RadixExtractor, Volume


def test_pyradiomics_spacing_is_read_as_xyz(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        "setting:\n  binWidth: 25\n  resampledPixelSpacing: [1.5, 2.0, 4.0]\n"
        "imageType:\n  Original: {}\n"
    )
    assert Params.from_pyradiomics_yaml(path).resampled_pixel_spacing == (4.0, 2.0, 1.5)


def _case(dtype):
    rng = np.random.default_rng(3)
    image = rng.integers(-200, 300, size=(12, 24, 24)).astype(dtype)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[3:9, 6:18, 6:18] = 1
    def vol(a):
        return Volume(
            array=a.astype(np.float64),
            spacing=(3.0, 0.8, 0.8),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            integer_pixels=np.issubdtype(np.dtype(dtype), np.integer),
            float32_pixels=np.dtype(dtype) == np.float32,
        )
    return vol(image), vol(mask)


def _median(dtype):
    params = Params(
        mode="pyradiomics",
        resampled_pixel_spacing=(2.0, 2.0, 2.0),
        interpolator="sitkBSpline",
        feature_classes={"firstorder": ["Median"]},
    )
    image, mask = _case(dtype)
    out = RadixExtractor(mode="pyradiomics", params=params).extract(image, mask)
    return out.values["original_firstorder_Median"]


def test_float_image_with_whole_values_keeps_fractions_after_resampling():
    # SimpleITK keeps the stored pixel type: float32 CT stays fractional.
    assert _median(np.float32) != round(_median(np.float32))


def test_integer_image_truncates_after_resampling():
    assert _median(np.int16) == int(_median(np.int16))
