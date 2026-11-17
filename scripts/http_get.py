"""Download a URL without requiring curl or a shell."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path


def download(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    user_agent: str = "pyradix-fetch/0.1",
) -> None:
    """Write ``url`` to ``dest``.

    Uses the Python standard library. If that fails (some macOS framework
    builds lack a CA bundle), fall back to ``curl`` when it is a real
    executable on ``PATH``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        partial.replace(dest)
        return
    except Exception:
        partial.unlink(missing_ok=True)
        curl = shutil.which("curl")
        if curl is None:
            raise
    subprocess.check_call(
        [curl, "-fsSL", "-A", user_agent, "-o", str(dest), url],
    )
