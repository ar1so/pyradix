#!/usr/bin/env python3
"""Run installed PyRadiomics with default RadiomicsFeatureExtractor settings.

This is the `radiomics` package, not Radix. Run it with the PyRadiomics
interpreter:

  PYRAD_PYTHON=path/to/python scripts/extract_pyradiomics_default.py IMAGE MASK
"""

from __future__ import annotations

import sys
import time

import radiomics
import SimpleITK as sitk
from radiomics import featureextractor


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_pyradiomics_default.py IMAGE MASK", file=sys.stderr)
        return 2
    from image_paths import image_io_path

    image, mask = image_io_path(sys.argv[1]), image_io_path(sys.argv[2])
    radiomics.setVerbosity(60)
    ex = featureextractor.RadiomicsFeatureExtractor()
    img = sitk.ReadImage(image)
    msk = sitk.ReadImage(mask)
    t0 = time.perf_counter()
    raw = ex.execute(img, msk)
    dt = time.perf_counter() - t0
    n = 0
    for key, value in raw.items():
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        n += 1
    print("engine pyradiomics")
    print("package radiomics")
    print("version %s" % radiomics.__version__)
    print("seconds %.3f" % dt)
    print("n_features %d" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
