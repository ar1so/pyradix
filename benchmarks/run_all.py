#!/usr/bin/env python3
"""Run the benchmarks in order and write results under benchmarks/results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from libbench import RESULTS, ensure_dirs, environment, write_json

HERE = Path(__file__).resolve().parent


def _run(script: str) -> int:
    print(f"\n=== {script} ===", file=sys.stderr)
    proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE.parents[1], check=False)
    return proc.returncode


def main() -> int:
    ensure_dirs()
    write_json(RESULTS / "environment.json", environment())
    print(f"wrote {RESULTS / 'environment.json'}", file=sys.stderr)
    codes = {
        "ibsi_agreement.py": _run("ibsi_agreement.py"),
        "pyradiomics_agreement.py": _run("pyradiomics_agreement.py"),
        "runtime_scaling.py": _run("runtime_scaling.py"),
        "make_figures.py": _run("make_figures.py"),
    }
    write_json(RESULTS / "run_status.json", {"returncodes": codes})
    failed = {name: code for name, code in codes.items() if code != 0}
    if failed:
        print(f"stages with non-zero status: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
