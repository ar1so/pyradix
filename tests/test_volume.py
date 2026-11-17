import pytest

from pyradix import Volume, load_volume


def test_load_ibsi_digital_phantom(d1_image, d1_mask):
    image = load_volume(d1_image)
    mask = load_volume(d1_mask)
    assert image.array.ndim == 3
    assert image.array.shape == mask.array.shape
    image.check_geometry(mask)
    assert image.array.dtype == "float64"
    assert image.array.size == 5 * 4 * 4


def test_load_pyradiomics_brain1(d4_image, d4_mask):
    image = load_volume(d4_image)
    mask = load_volume(d4_mask)
    assert image.array.ndim == 3
    assert image.array.shape == mask.array.shape
    image.check_geometry(mask)
    assert mask.array.max() >= 1


def test_geometry_mismatch(d1_image):
    image = load_volume(d1_image)
    other = Volume(
        array=image.array[:1],
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        image.check_geometry(other)
