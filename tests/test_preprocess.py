import numpy as np

from pyradix import Params, RadixExtractor
from pyradix.volume import Volume


def _box(values):
    image = Volume(np.asarray(values, dtype=np.float64), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), ())
    mask = Volume(np.ones(values.shape, dtype=np.float64), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), ())
    return image, mask


def test_normalize_sets_the_mean_to_zero():
    values = np.zeros((2, 2, 2))
    values.reshape(-1)[:4] = 4.0
    image, mask = _box(values)
    raw = RadixExtractor(mode="pyradiomics", params=Params(mode="pyradiomics")).extract(image, mask)
    scaled = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", normalize=True, normalize_scale=2.0),
    ).extract(image, mask)
    assert raw.values["original_firstorder_Mean"] == 2.0
    assert abs(scaled.values["original_firstorder_Mean"]) < 1e-9


def test_resegment_shape_changes_voxel_volume():
    values = np.zeros((2, 2, 2))
    values.reshape(-1)[:4] = 10.0
    image, mask = _box(values)
    kept = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", resegment_range=(5.0, 100.0), resegment_shape=False),
    ).extract(image, mask)
    shaped = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", resegment_range=(5.0, 100.0), resegment_shape=True),
    ).extract(image, mask)
    assert kept.values["original_firstorder_Mean"] == 10.0
    assert shaped.values["original_firstorder_Mean"] == 10.0
    assert kept.values["original_shape_VoxelVolume"] == 8.0
    assert shaped.values["original_shape_VoxelVolume"] == 4.0


def test_relative_resegment_uses_the_maximum():
    values = np.zeros((2, 2, 2))
    values.reshape(-1)[:4] = 10.0
    image, mask = _box(values)
    result = RadixExtractor(
        mode="pyradiomics",
        params=Params(
            mode="pyradiomics",
            resegment_range=(0.5, 1.0),
            resegment_mode="relative",
            resegment_shape=True,
        ),
    ).extract(image, mask)
    assert result.values["original_shape_VoxelVolume"] == 4.0


def test_spacing_zero_keeps_the_original_axis():
    image = Volume(np.ones((4, 4, 4)), (3.0, 1.0, 1.0), (0.0, 0.0, 0.0), ())
    mask = Volume(np.ones((4, 4, 4)), (3.0, 1.0, 1.0), (0.0, 0.0, 0.0), ())
    result = RadixExtractor(
        mode="pyradiomics",
        params=Params(
            mode="pyradiomics",
            resampled_pixel_spacing=(0.0, 2.0, 2.0),
            interpolator="linear",
        ),
    ).extract(image, mask)
    assert result.provenance["resampled_spacing"][0] == 3.0
    assert result.provenance["resampled_spacing"][1] == 2.0
    assert result.provenance["resampled_spacing"][2] == 2.0


def test_pre_crop_still_extracts():
    image, mask = _box(np.arange(8.0).reshape(2, 2, 2))
    result = RadixExtractor(
        mode="pyradiomics",
        params=Params(mode="pyradiomics", pre_crop=True, pad_distance=1),
    ).extract(image, mask)
    assert result.values["original_shape_VoxelVolume"] == 8.0
