"""Parameter file: four blocks, one error, keyword override."""

import json
from pathlib import Path

import pytest

from pyradix import Params, RadixError, RadixExtractor


def test_pyradiomics_yaml_keeps_type_settings(examples_dir):
    params = Params.from_pyradiomics_yaml(examples_dir / "pyradiomics.yaml")
    assert params.mode == "pyradiomics"
    assert params.image_types == (("Original", {}),)
    assert params.bin_width == 25.0
    assert params.label == 1


def test_json_file_parses_all_four_blocks(tmp_path: Path):
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "setting": {"binWidth": 10, "label": 2},
                "imageType": {"Original": {}, "LoG": {"sigma": [1, 2]}},
                "featureClass": {"firstorder": []},
                "voxelSetting": {"kernelRadius": 2, "maskedKernel": False},
            }
        ),
        encoding="utf-8",
    )
    params = Params.from_pyradiomics_json(path)
    assert params.mode == "pyradiomics"
    assert params.bin_width == 10
    assert params.label == 2
    assert params.image_types[1][0] == "LoG"
    assert params.image_types[1][1]["sigma"] == [1, 2]
    assert set(params.enabled_features()) == {"firstorder"}
    assert "StandardDeviation" not in params.enabled_features()["firstorder"]
    assert "Mean" in params.enabled_features()["firstorder"]
    assert params.voxel_setting["kernelRadius"] == 2
    assert params.voxel_setting["maskedKernel"] is False


def test_missing_feature_class_enables_non_deprecated():
    params = Params(mode="pyradiomics")
    enabled = params.enabled_features()
    assert "glcm" in enabled
    assert "Dissimilarity" not in enabled["glcm"]
    assert "Contrast" in enabled["glcm"]


def test_named_deprecated_feature_stays_on():
    params = Params(mode="pyradiomics", feature_classes={"firstorder": ("Mean", "StandardDeviation")})
    assert params.enabled_features()["firstorder"] == ("Mean", "StandardDeviation")


def test_keyword_overrides_file(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text("setting:\n  binWidth: 25\n  label: 1\n", encoding="utf-8")
    params = Params.from_pyradiomics_yaml(path, bin_width=5)
    assert params.mode == "pyradiomics"
    assert params.bin_width == 5
    assert params.label == 1


def test_zero_spacing_component_is_kept():
    params = Params(mode="pyradiomics", resampled_pixel_spacing=(0, 2, 2))
    assert params.resampled_pixel_spacing == (0.0, 2.0, 2.0)


def test_image_type_cannot_override_spacing(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text(
        "imageType:\n  Original:\n    interpolator: sitkLinear\n    resampledPixelSpacing: [1, 1, 1]\n",
        encoding="utf-8",
    )
    with pytest.raises(RadixError, match="interpolator") as exc:
        Params.from_pyradiomics_yaml(path)
    assert "resampledPixelSpacing" in exc.value.message
    assert exc.value.code == "setting"


def test_one_error_lists_every_bad_field(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text(
        "\n".join(
            [
                "setting:",
                "  label: 0",
                "  interpolator: not-an-interpolator",
                "  resampledPixelSpacing: [-1, 1, 1]",
                "imageType:",
                "  LoG: {}",
                "  Mean: {}",
                "  Riesz: {}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RadixError) as exc:
        RadixExtractor.from_pyradiomics_yaml(path)
    message = exc.value.message
    assert exc.value.code == "setting"
    for needle in ("label", "interpolator", "resampledPixelSpacing", "LoG", "Mean", "Riesz"):
        assert needle in message
