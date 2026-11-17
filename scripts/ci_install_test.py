#!/usr/bin/env python3
"""Install the one wheel in dist/ and run the test suite with this interpreter."""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    wheels = sorted(glob.glob(str(ROOT / "dist" / "*.whl")))
    if len(wheels) != 1:
        print(f"expected one wheel in dist/, found {wheels}", file=sys.stderr)
        return 1
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", wheels[0]])
    run([sys.executable, "-m", "pip", "install", "pytest>=8", "simpleitk>=2.3"])
    run([sys.executable, str(ROOT / "scripts" / "fetch_validation_data.py")])
    run(
        [
            sys.executable,
            "-c",
            "import pathlib, pyradix; p = pathlib.Path(pyradix.__file__).as_posix();\n"
            "assert 'site-packages' in p, p",
        ]
    )
    run([sys.executable, str(ROOT / "scripts" / "wheel_smoke.py")])
    run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
