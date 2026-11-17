"""R1 / R2: config hash + CLEAR YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from pyradix import Params, dump_clear_yaml
from pyradix.clear import REQUIRED_KEYS
from pyradix.extractor import ExtractResult


def test_r1_hash_changes_when_bin_width_changes():
    a = Params(mode="ibsi", bin_width=25.0).provenance()
    b = Params(mode="ibsi", bin_width=50.0).provenance()
    assert a["config_hash"] != b["config_hash"]
    assert len(a["config_hash"]) == 64


def test_config_hash_changes_with_mode_image_type_and_sigma():
    base = Params(mode="ibsi", image_types=(("Original", {}),), sigma=(1.0,)).provenance()
    other_mode = Params(mode="pyradiomics", image_types=(("Original", {}),), sigma=(1.0,)).provenance()
    other_type = Params(mode="ibsi", image_types=(("LoG", {"sigma": [1.0]}),), sigma=(1.0,)).provenance()
    other_sigma = Params(mode="ibsi", image_types=(("Original", {}),), sigma=(2.0,)).provenance()
    same = Params(mode="ibsi", image_types=(("Original", {}),), sigma=(1.0,)).provenance()
    assert base["config_hash"] != other_mode["config_hash"]
    assert base["config_hash"] != other_type["config_hash"]
    assert base["config_hash"] != other_sigma["config_hash"]
    assert base["config_hash"] == same["config_hash"]


def test_r2_clear_yaml_has_required_keys(tmp_path: Path):
    params = Params(mode="ibsi", bin_width=25.0)
    result = ExtractResult(values={"original_firstorder_Mean": 1.0}, provenance=params.provenance())
    path = tmp_path / "clear.yaml"
    body = dump_clear_yaml(result, path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in REQUIRED_KEYS:
        assert key in body and body[key] is not None, key
        assert key in loaded, key
