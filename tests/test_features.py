import numpy as np

from pyradix import RadixExtractor, load_volume

PREFIXES = (
    "original_firstorder_",
    "original_shape_",
    "original_glcm_",
    "original_glrlm_",
    "original_glszm_",
    "original_ngtdm_",
    "original_gldm_",
)

FIRSTORDER = {
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "10Percentile",
    "90Percentile",
    "Maximum",
    "Mean",
    "Median",
    "InterquartileRange",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "Kurtosis",
    "Variance",
    "Uniformity",
}


def test_all_feature_classes_on_digital_phantom(d1_image, d1_mask):
    ex = RadixExtractor(mode="ibsi", params={"bin_width": 1.0})
    result = ex.extract(d1_image, d1_mask, label=1)
    keys = result.values.keys()
    for prefix in PREFIXES:
        assert any(k.startswith(prefix) for k in keys), prefix
    for name in FIRSTORDER:
        assert f"original_firstorder_{name}" in result.values, name
    assert result.provenance["n_features"] >= 100
    assert all(np.isfinite(v) for v in result.values.values())

    image = load_volume(d1_image)
    mask = load_volume(d1_mask)
    roi = image.array[mask.array == 1]
    assert abs(result.values["original_firstorder_Mean"] - float(roi.mean())) < 1e-9
    assert abs(result.values["original_firstorder_Minimum"] - float(roi.min())) < 1e-9
    assert abs(result.values["original_firstorder_Maximum"] - float(roi.max())) < 1e-9
    nv = int(np.count_nonzero(mask.array == 1))
    vox = float(np.prod(image.spacing))
    assert abs(result.values["original_shape_VoxelVolume"] - nv * vox) < 1e-6


def test_extract_no_longer_empty(d1_image, d1_mask):
    ex = RadixExtractor(mode="ibsi")
    result = ex.extract(d1_image, d1_mask, label=1)
    assert result.values
