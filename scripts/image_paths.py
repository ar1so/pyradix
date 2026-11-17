"""Stdlib path helper for scripts that must not import the extension.

The installed package exposes the same function as ``pyradix.paths.image_io_path``.
Keep the two copies identical.
"""

from __future__ import annotations

import os
from pathlib import Path


def image_io_path(path: str | os.PathLike[str]) -> str:
    """Absolute path suitable for SimpleITK ``ReadImage`` / ``WriteImage``."""
    resolved = Path(path).expanduser().resolve()
    if os.name != "nt":
        return os.fspath(resolved)
    text = os.path.normpath(os.fspath(resolved))
    if text.startswith("\\\\?\\"):
        return text
    if len(text) >= 240:
        return "\\\\?\\" + text
    return text.replace("\\", "/")
