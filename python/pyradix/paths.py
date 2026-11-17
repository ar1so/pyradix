"""Filesystem paths for image I/O.

Nibabel and NRRD take a native absolute path. SimpleITK / ITK on Windows
accept forward slashes; a raw backslash path can be read as an escape.
Very long Windows paths need the ``\\\\?\\`` prefix, which requires
backslashes, so those are left in native form.
"""

from __future__ import annotations

import os
from pathlib import Path

PathLike = str | os.PathLike[str]


def native_path(path: PathLike) -> str:
    """Absolute native path. The file does not have to exist yet."""
    return os.fspath(Path(path).expanduser().resolve())


def image_io_path(path: PathLike) -> str:
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
