---
description: "IBSI or PyRadiomics mode in pyradix: which radiomics standard to choose, how feature definitions, binning and filters differ, and how to match PyRadiomics 3.1.0."
---

# IBSI or PyRadiomics?

| | `mode="ibsi"` (default) | `mode="pyradiomics"` |
|---|---|---|
| Follows | IBSI 1 and IBSI 2 | PyRadiomics 3.1.0 |
| Choose it for | New studies, multi-centre work, anything you want to call standardised | Repeating or extending a study that used PyRadiomics |
| Feature names | `original_glcm_Contrast` style | Identical to PyRadiomics |
| Filters | Mean, LoG, Wavelet, Laws, Gabor | LoG, Wavelet, Square, Square Root, Logarithm, Exponential, Gradient, LBP 2D/3D |

**Feature classes in both modes:** First Order, Shape (2D and 3D), GLCM, GLRLM, GLSZM, GLDM, NGTDM.

## Why the numbers differ

PyRadiomics predates the IBSI standard and differs from it in several places. A few examples:

- **Discretization.** Fixed-bin-size bins start at different values.
- **Kurtosis.** PyRadiomics reports raw kurtosis; IBSI reports excess kurtosis (minus 3).
- **Resampling grid.** The two align the new grid differently.
- **LoG and wavelet.** The filters are built differently.

So the same image and mask give different values in the two modes. Neither is a bug. Report which mode you used: it is in `result.provenance["mode"]` and in the configuration hash.

The full list, with what pyradix does in each mode, is in [DEVIATIONS.md](https://github.com/ar1so/pyradix/blob/main/docs/DEVIATIONS.md). Feature definitions are in [FEATURES.md](https://github.com/ar1so/pyradix/blob/main/docs/FEATURES.md), and preprocessing in [PREPROCESS.md](https://github.com/ar1so/pyradix/blob/main/docs/PREPROCESS.md).
