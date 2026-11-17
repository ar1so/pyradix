"""Registry names and extract-path errors."""

import numpy as np
import pytest

from pyradix import Params, RadixError
from pyradix._radix import extract_features, legal_feature_classes, legal_image_types


def test_pyradiomics_registry_excludes_ibsi_only_filters():
    names = set(legal_image_types("pyradiomics"))
    assert {"Original", "LoG", "Wavelet", "Square", "Gradient", "LBP2D", "LBP3D"} <= names
    assert "Mean" not in names
    assert "Laws" not in names
    assert "Gabor" not in names
    assert "Riesz" not in names


def test_ibsi_registry_excludes_pyradiomics_only_filters():
    names = set(legal_image_types("ibsi"))
    assert {"Original", "Mean", "LoG", "Laws", "Gabor", "Wavelet"} <= names
    assert "Square" not in names
    assert "LBP3D" not in names


def test_feature_classes_come_from_the_registry():
    assert set(legal_feature_classes()) == {
        "firstorder",
        "shape",
        "shape2D",
        "glcm",
        "glrlm",
        "glszm",
        "ngtdm",
        "gldm",
    }


def test_unknown_image_type_names_the_type():
    with pytest.raises(RadixError, match="Riesz"):
        Params(mode="pyradiomics", image_types=("Riesz",))


def test_mean_is_rejected_in_pyradiomics_mode():
    with pytest.raises(RadixError, match="Mean"):
        Params(mode="pyradiomics", image_types=("Mean",))


def test_laws_and_gabor_are_rejected_in_pyradiomics_mode():
    with pytest.raises(RadixError, match="Laws"):
        Params(mode="pyradiomics", image_types=("Laws",))
    with pytest.raises(RadixError, match="Gabor"):
        Params(mode="pyradiomics", image_types=("Gabor",))


def test_riesz_is_rejected_in_both_modes():
    for mode in ("ibsi", "pyradiomics"):
        with pytest.raises(RadixError, match="Riesz"):
            Params(mode=mode, image_types=("Riesz",))


def test_square_is_rejected_in_ibsi_mode():
    with pytest.raises(RadixError, match="Square"):
        Params(mode="ibsi", image_types=("Square",))


def test_unknown_feature_class_names_the_class():
    with pytest.raises(RadixError, match="notAClass"):
        Params(mode="ibsi", extra={"featureClass": {"notAClass": []}})


def test_empty_roi_is_a_case_error():
    image = np.zeros((2, 2, 2), dtype=np.float64)
    mask = np.zeros((2, 2, 2), dtype=bool)
    with pytest.raises(RadixError) as exc:
        extract_features(image, mask, (1.0, 1.0, 1.0), "ibsi", 25.0)
    assert exc.value.code == "case"
    assert "empty" in exc.value.message
    assert exc.value.message.startswith("resegment:")
