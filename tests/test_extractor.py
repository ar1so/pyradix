import pytest

from pyradix import RadixError, RadixExtractor


def test_both_modes_construct():
    ibsi = RadixExtractor(mode="ibsi")
    compat = RadixExtractor(mode="pyradiomics")
    assert ibsi.mode == "ibsi"
    assert compat.mode == "pyradiomics"


def test_from_pyradiomics_yaml(examples_dir):
    ex = RadixExtractor.from_pyradiomics_yaml(examples_dir / "pyradiomics.yaml")
    assert ex.mode == "pyradiomics"
    assert ex.params.bin_width == 25


def test_extract_returns_provenance(d1_image, d1_mask):
    ex = RadixExtractor(mode="ibsi")
    result = ex.extract(d1_image, d1_mask, label=1)
    assert result.values
    assert result.provenance["mode"] == "ibsi"
    assert result.provenance["roi_voxels"] > 0
    assert len(result.provenance["config_hash"]) == 64


def test_missing_label_raises(d1_image, d1_mask):
    ex = RadixExtractor(mode="ibsi")
    with pytest.raises(RadixError, match="label 99") as exc:
        ex.extract(d1_image, d1_mask, label=99)
    assert exc.value.code == "case"
    assert "present labels" in exc.value.message
