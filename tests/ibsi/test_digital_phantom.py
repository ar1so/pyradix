"""G1 / G2: IBSI-1 digital phantom vs published tables."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pyradix import Params, RadixExtractor
from pyradix._radix import feature_catalog


def _with_deprecated(params: Params) -> Params:
    catalog = feature_catalog()
    params.feature_classes = {
        name: tuple(entry["features"]) for name, entry in catalog.items()
    }
    return params

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "tests" / "references" / "ibsi1_digital_phantom.csv"


def _rows(gate: str) -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return [row for row in csv.DictReader(lines) if row["gate"] == gate]


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


@pytest.fixture(scope="module")
def d1_values(d1_image, d1_mask):
    if not Path(d1_image).exists():
        pytest.skip("D1 digital phantom not fetched")
    ex = RadixExtractor(
        mode="ibsi",
        params=_with_deprecated(Params(mode="ibsi", bin_width=1.0)),
    )
    return ex.extract(d1_image, d1_mask, label=1).values


def _check(gate: str, values: dict[str, float]) -> None:
    failures: list[str] = []
    for row in _rows(gate):
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


def test_g1_firstorder_shape(d1_values):
    _check("G1", d1_values)


def test_g2_texture(d1_values):
    _check("G2", d1_values)
