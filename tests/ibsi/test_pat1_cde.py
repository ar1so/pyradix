"""G3: IBSI-1 PAT1 configs C/D/E vs published tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pyradix import Params, RadixExtractor, Volume
from pyradix._radix import feature_catalog

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "tests" / "references" / "ibsi1_pat1_config_cde.csv"
EXAMPLES = ROOT / "examples"
CONFIG_YAML = {
    "C": EXAMPLES / "ibsi_pat1.yaml",
    "D": EXAMPLES / "ibsi_pat1_d.yaml",
    "E": EXAMPLES / "ibsi_pat1_e.yaml",
}


def _rows(config: str) -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return [row for row in csv.DictReader(lines) if row["config"] == config]


def _tol(row: dict[str, str]) -> float:
    raw = (row.get("tol") or "").strip()
    if raw:
        return float(raw)
    text = row["value"].strip().lstrip("-")
    if "." in text:
        decimals = len(text.split(".", 1)[1])
        return 0.5 * (10 ** (-decimals))
    return 0.5


def _close(got: float, expected: float, tol: float) -> bool:
    return abs(got - expected) <= max(tol, 1e-12)


def _extract(config: str, d2_image: Path, d2_mask: Path) -> dict[str, float]:
    if not Path(d2_image).exists():
        pytest.skip("D2 PAT1 not fetched")
    params = Params.from_yaml(CONFIG_YAML[config])
    catalog = feature_catalog()
    params.feature_classes = {name: tuple(entry["features"]) for name, entry in catalog.items()}
    ex = RadixExtractor(mode="ibsi", params=params)
    return ex.extract(d2_image, d2_mask, label=1).values


def _check(config: str, values: dict[str, float]) -> None:
    failures: list[str] = []
    for row in _rows(config):
        key = row["radix_key"]
        expected = float(row["value"])
        tol = _tol(row)
        if key not in values:
            failures.append(f"missing {key}")
            continue
        got = values[key]
        if not _close(got, expected, tol):
            failures.append(f"{key}: got {got:.6g} expected {expected} ± {tol}")
    assert not failures, "\n".join(failures)


@pytest.fixture(scope="module")
def values_c(d2_image, d2_mask):
    return _extract("C", d2_image, d2_mask)


@pytest.fixture(scope="module")
def values_d(d2_image, d2_mask):
    return _extract("D", d2_image, d2_mask)


@pytest.fixture(scope="module")
def values_e(d2_image, d2_mask):
    return _extract("E", d2_image, d2_mask)


def test_g3_config_c(values_c):
    _check("C", values_c)


def test_g3_config_d(values_d):
    _check("D", values_d)


def test_g3_config_e(values_e):
    _check("E", values_e)


def test_resegment_range_intensity_mask_only():
    image = np.zeros((3, 3, 3), dtype=np.float64)
    image[1, 1, 1] = 50.0
    image[1, 1, 0] = 900.0
    mask = np.zeros((3, 3, 3), dtype=np.float64)
    mask[1, 1, 0] = 1
    mask[1, 1, 1] = 1
    vol_i = Volume(array=image, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=())
    vol_m = Volume(array=mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=())
    ex = RadixExtractor(
        mode="ibsi",
        params={"bin_width": 25.0, "resegment_range": (-1000.0, 400.0), "minimum_roi_dimensions": 1},
    )
    result = ex.extract(vol_i, vol_m, label=1)
    assert result.values["original_firstorder_Maximum"] == pytest.approx(50.0)
    assert result.values["original_firstorder_Minimum"] == pytest.approx(50.0)
    assert result.values["original_shape_VoxelVolume"] == pytest.approx(2.0)
