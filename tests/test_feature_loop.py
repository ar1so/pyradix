import numpy as np
import pytest

from pyradix import Params, RadixExtractor
from pyradix.volume import Volume


def _volume(values, spacing=(1.0, 1.0, 1.0)):
    image = Volume(np.asarray(values, dtype=np.float64), spacing, (0.0, 0.0, 0.0), ())
    mask = Volume(np.ones(values.shape, dtype=np.float64), spacing, (0.0, 0.0, 0.0), ())
    return image, mask


def _pattern():
    values = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    values[0, 0, 0] = 0
    return _volume(values)


def test_deprecated_names_are_off_unless_listed():
    image, mask = _pattern()
    plain = RadixExtractor(mode="pyradiomics", params=Params(mode="pyradiomics")).extract(image, mask)
    assert "original_firstorder_StandardDeviation" not in plain.values
    assert "original_shape_Compactness1" not in plain.values
    assert "original_glcm_Dissimilarity" not in plain.values
    assert "original_gldm_DependencePercentage" not in plain.values
    assert "original_firstorder_Mean" in plain.values
    named = RadixExtractor(
        mode="pyradiomics",
        params=Params(
            mode="pyradiomics",
            feature_classes={"firstorder": ["Mean", "StandardDeviation"]},
        ),
    ).extract(image, mask)
    assert set(named.values) == {
        "original_firstorder_Mean",
        "original_firstorder_StandardDeviation",
    }


def test_omitted_class_is_absent():
    image, mask = _pattern()
    result = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", feature_classes={"firstorder": []}),
    ).extract(image, mask)
    assert result.values
    assert all(key.startswith("original_firstorder_") for key in result.values)
    assert "original_firstorder_StandardDeviation" not in result.values


def test_single_slice_enables_shape2d():
    image, mask = _volume(np.arange(16, dtype=np.float64).reshape(1, 4, 4))
    result = RadixExtractor(mode="pyradiomics", params=Params(mode="pyradiomics")).extract(image, mask)
    assert "original_shape2D_PixelSurface" in result.values


def test_force2d_averages_texture_and_enables_shape2d():
    values = np.zeros((4, 4, 4))
    values[0] = 1
    values[1] = 8
    values[2] = 3
    values[3] = 40
    image, mask = _volume(values)
    full = RadixExtractor(mode="pyradiomics", params=Params(mode="pyradiomics")).extract(image, mask)
    flat = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", force_2d=True, force_2d_dimension=0),
    ).extract(image, mask)
    assert "original_shape2D_PixelSurface" not in full.values
    assert "original_shape2D_PixelSurface" in flat.values
    assert flat.values["original_glcm_Contrast"] != pytest.approx(full.values["original_glcm_Contrast"])


def test_distances_weighting_symmetry_and_gldm_change_texture():
    image, mask = _pattern()
    base = Params(mode="pyradiomics")
    reference = RadixExtractor(mode="pyradiomics", params=base).extract(image, mask)
    farther = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", distances=(2,)),
    ).extract(image, mask)
    asymmetric = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", symmetrical_glcm=False),
    ).extract(image, mask)
    weighted = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", weighting_norm="euclidean"),
    ).extract(image, mask)
    wider = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", gldm_a=5),
    ).extract(image, mask)
    assert farther.values["original_glcm_Contrast"] != pytest.approx(
        reference.values["original_glcm_Contrast"]
    )
    assert asymmetric.values["original_glcm_Correlation"] != pytest.approx(
        reference.values["original_glcm_Correlation"]
    )
    assert weighted.values["original_glcm_Contrast"] != pytest.approx(
        reference.values["original_glcm_Contrast"]
    )
    assert wider.values["original_gldm_SmallDependenceEmphasis"] != pytest.approx(
        reference.values["original_gldm_SmallDependenceEmphasis"]
    )


def test_voxel_array_shift_changes_energy():
    image, mask = _pattern()
    plain = RadixExtractor(mode="pyradiomics", params=Params(mode="pyradiomics")).extract(image, mask)
    shifted = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", voxel_array_shift=10),
    ).extract(image, mask)
    assert shifted.values["original_firstorder_Mean"] == pytest.approx(plain.values["original_firstorder_Mean"])
    assert shifted.values["original_firstorder_Energy"] != pytest.approx(plain.values["original_firstorder_Energy"])


def test_mean_image_type_returns_a_table():
    image, mask = _pattern()
    extractor = RadixExtractor(
        mode="ibsi",
        params=Params(mode="ibsi", image_types=(("Original", {}), ("Mean", {}))),
    )
    result = extractor.extract(image, mask)
    assert any(key.startswith("original_firstorder_") for key in result.values)
    assert any(key.startswith("mean_firstorder_") for key in result.values)
