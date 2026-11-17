#!/usr/bin/env python3
"""Install every wheel in dist/ and run the suite, or a LAPACK smoke test.

Used inside the manylinux build container. Interpreters are ``python3.10``
or ``/opt/python/cp310-cp310/bin/python``. Set ``WHEEL_TEST=smoke`` to skip
the full suite (aarch64 emulation).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def interpreter_for(wheel: Path) -> str:
    tag = wheel.name.split("-")[2]
    if not tag.startswith("cp"):
        raise SystemExit(f"unrecognized wheel tag in {wheel.name}")
    digits = tag.removeprefix("cp")
    dotted = f"{digits[0]}.{digits[1:]}"
    candidates = [
        f"python{dotted}",
        f"/opt/python/cp{digits}-cp{digits}/bin/python",
    ]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(f"no interpreter for {wheel.name}; tried {candidates}")


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env, cwd=ROOT)


def main() -> int:
    wheels = sorted(Path(path) for path in glob.glob(str(DIST / "*.whl")))
    if not wheels:
        print("no wheels in dist/", file=sys.stderr)
        return 1
    smoke_only = os.environ.get("WHEEL_TEST", "pytest") == "smoke"
    for wheel in wheels:
        py = interpreter_for(wheel)
        venv = Path("/tmp") / f"pyradix-{wheel.name.split('-')[2]}"
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.check_call([py, "-m", "venv", str(venv)])
        python = venv / "bin" / "python"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        run([str(python), "-m", "pip", "install", "--force-reinstall", str(wheel)], env)
        run([str(python), str(ROOT / "scripts" / "wheel_smoke.py")], env)
        if smoke_only:
            continue
        run([str(python), "-m", "pip", "install", "pytest>=8", "simpleitk>=2.3"], env)
        run([str(python), str(ROOT / "scripts" / "fetch_validation_data.py")], env)
        run([str(python), "-m", "pytest", "-q", str(ROOT / "tests")], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
