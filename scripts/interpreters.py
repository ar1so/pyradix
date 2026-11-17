"""The second interpreter used to run installed PyRadiomics.

Comparison runs launch PyRadiomics 3.1.0 in its own Python so that process
never imports pyradix. Set PYRAD_PYTHON to that interpreter. This repository
does not assume a path on any one machine.
"""

from __future__ import annotations

import os
from pathlib import Path

_MISSING = "Set PYRAD_PYTHON to the interpreter that has PyRadiomics 3.1.0 installed."


def pyrad_python() -> Path | None:
    raw = os.environ.get("PYRAD_PYTHON", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def require_pyrad_python() -> Path:
    path = pyrad_python()
    if path is None:
        raise SystemExit(_MISSING)
    return path
