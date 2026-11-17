from pathlib import Path

import pytest

from pyradix import Params, ParamsError, RadixError, RadixExtractor


def test_valid_ibsi_yaml(examples_dir):
    params = Params.from_yaml(examples_dir / "ibsi.yaml")
    assert params.mode == "ibsi"
    assert params.bin_width == 25.0


def test_invalid_bin_width():
    with pytest.raises(RadixError, match="binWidth"):
        Params(mode="ibsi", bin_width=0)


def test_unknown_key():
    with pytest.raises(RadixError, match="unknown param"):
        Params.from_mapping({"mode": "ibsi", "not_a_field": 1})


def test_tabs_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode:\tibsi\nbin_width: 25\n", encoding="utf-8")
    with pytest.raises(ParamsError, match="tabs"):
        Params.from_yaml(path)


def test_resegment_range_yaml(examples_dir):
    params = Params.from_yaml(examples_dir / "ibsi_pat1.yaml")
    assert params.resegment_range == (-1000.0, 400.0)
    d = Params.from_yaml(examples_dir / "ibsi_pat1_d.yaml")
    assert d.bin_count == 32
    assert d.bin_width is None
    assert d.outlier_sigma == 3.0
    e = Params.from_yaml(examples_dir / "ibsi_pat1_e.yaml")
    assert e.mask_interpolator == "linear"
    assert e.interpolator == "bspline"


def test_unknown_pyradiomics_setting(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("setting:\n  notARealKey: 1\n", encoding="utf-8")
    with pytest.raises(RadixError, match="unknown PyRadiomics setting"):
        RadixExtractor.from_pyradiomics_yaml(path)
