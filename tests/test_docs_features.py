"""The feature reference documents exactly the features the engine computes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyradix._radix import feature_catalog

DOCS = Path(__file__).resolve().parents[1] / "docsite" / "features"
PAGES = {
    "firstorder": "firstorder.md", "shape": "shape.md", "shape2D": "shape2d.md", "glcm": "glcm.md",
    "glrlm": "glrlm.md", "glszm": "glszm.md", "gldm": "gldm.md", "ngtdm": "ngtdm.md",
}


def _documented(page: str) -> dict[str, bool]:
    """Feature name -> documented as deprecated, from the table rows `| **Name** ...`."""
    text = (DOCS / page).read_text()
    rows = re.findall(r"^\| \*\*(\w+)\*\*( \*\(deprecated\)\*)? \|", text, re.M)
    return {name: bool(dep) for name, dep in rows}


@pytest.mark.parametrize("feature_class", sorted(PAGES))
def test_feature_page_matches_engine(feature_class):
    catalog = feature_catalog()[feature_class]
    documented = _documented(PAGES[feature_class])
    assert set(documented) == set(catalog["features"]), (
        f"missing in docs: {set(catalog['features']) - set(documented)}; "
        f"not in engine: {set(documented) - set(catalog['features'])}")
    deprecated = {name for name, dep in documented.items() if dep}
    assert deprecated == set(catalog["deprecated"])


def test_every_class_has_a_page():
    assert set(feature_catalog()) == set(PAGES)
