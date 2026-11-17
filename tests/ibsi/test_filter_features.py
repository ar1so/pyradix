"""IBSI feature columns on Mean, Laws, and Gabor responses.

IBSI published filter images for these kernels and no feature table on the
filtered volumes. The columns still have to exist and be finite.
"""

from __future__ import annotations

import math

import numpy as np

from pyradix import Params, RadixExtractor
from pyradix._radix import feature_catalog
from pyradix.volume import Volume

TEXTURE = ("firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm")


def _volume():
    image = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    image[2:6, 2:6, 2:6] = image[2:6, 2:6, 2:6] * 1.7 + 3.0
    mask = np.zeros(image.shape, dtype=np.float64)
    mask[1:7, 1:7, 1:7] = 1.0
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    return Volume(image, spacing, origin, ()), Volume(mask, spacing, origin, ())


def _columns():
    catalog = feature_catalog()
    columns = []
    for name in TEXTURE:
        entry = catalog[name]
        deprecated = set(entry["deprecated"])
        columns.extend((name, feature) for feature in entry["features"] if feature not in deprecated)
    return columns


def test_ibsi_filter_responses_have_finite_feature_columns():
    image, mask = _volume()
    params = Params(
        mode="ibsi",
        image_types=(
            ("Original", {}),
            ("Mean", {"support": 3, "padding": "mirror"}),
            ("Laws", {"kernel": "E5L5S5,L5E5S5", "padding": "mirror"}),
            (
                "Gabor",
                {
                    "sigma": 1.0,
                    "lambda": 2.0,
                    "gamma": 1.0,
                    "theta": [0.0, math.pi / 4],
                    "padding": "mirror",
                },
            ),
        ),
        feature_classes={name: [] for name in ("shape", *TEXTURE)},
        bin_width=25,
    )
    values = RadixExtractor(mode="ibsi", params=params).extract(image, mask).values
    prefixes = ("mean", "laws-E5L5S5", "laws-L5E5S5", "gabor-0.0", "gabor-0.7853981633974483")
    for prefix in prefixes:
        for class_name, feature in _columns():
            key = f"{prefix}_{class_name}_{feature}"
            assert key in values, key
            assert math.isfinite(values[key]), key
        assert f"{prefix}_shape_MeshVolume" not in values
    assert "original_shape_MeshVolume" in values
    assert math.isfinite(values["original_shape_MeshVolume"])


def test_provenance_records_the_mode():
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    mask = np.ones((2, 2, 2), dtype=np.float64)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    image_vol = Volume(image, spacing, origin, ())
    mask_vol = Volume(mask, spacing, origin, ())
    ibsi = RadixExtractor(
        mode="ibsi",
        params=Params(mode="ibsi", image_types=(("Original", {}),), bin_width=1),
    ).extract(image_vol, mask_vol)
    pyrad = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", image_types=(("Original", {}),), bin_width=1),
    ).extract(image_vol, mask_vol)
    assert ibsi.provenance["mode"] == "ibsi"
    assert pyrad.provenance["mode"] == "pyradiomics"
    assert ibsi.values["original_firstorder_Kurtosis"] != pyrad.values["original_firstorder_Kurtosis"]
