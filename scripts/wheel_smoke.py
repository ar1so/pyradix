"""Smoke test for an installed pyradix wheel.

Runs GLCM MCC, which calls LAPACK dgeev. An import-only check would pass on a
build that fell back to a different eigensolver.
"""

import numpy as np

import pyradix


def main() -> None:
    print(pyradix.engine_info())
    image = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    mask = np.ones((4, 4, 4), dtype=bool)
    out = pyradix.extract_features(
        image,
        mask,
        (1.0, 1.0, 1.0),
        "pyradiomics",
        bin_width=25.0,
        feature_classes=[("glcm", [])],
    )
    mcc = out["original_glcm_MCC"]
    if mcc != mcc:
        raise SystemExit("original_glcm_MCC was NaN")
    print(f"original_glcm_MCC={mcc}")


if __name__ == "__main__":
    main()
