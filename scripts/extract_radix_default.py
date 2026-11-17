#!/usr/bin/env python3
"""Run Radix with default extractor settings (IBSI mode, binWidth=25, Original).

Usage:
  .venv/bin/python scripts/extract_radix_default.py data/lung1/LUNG1-005/image.nii.gz data/lung1/LUNG1-005/mask.nii.gz
"""

from __future__ import annotations

import sys
import time

from pyradix import RadixExtractor


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_radix_default.py IMAGE MASK", file=sys.stderr)
        return 2
    image, mask = sys.argv[1], sys.argv[2]
    ex = RadixExtractor()
    t0 = time.perf_counter()
    out = ex.extract(image, mask)
    dt = time.perf_counter() - t0
    print("engine radix")
    print("mode %s" % ex.mode)
    print("seconds %.3f" % dt)
    print("n_features %d" % len(out.values))
    print("roi_voxels %s" % out.provenance.get("roi_voxels"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
