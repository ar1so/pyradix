---
description: "Radiomics feature reference: first-order, shape, GLCM, GLRLM, GLSZM, GLDM and NGTDM features in pyradix, with names identical to PyRadiomics."
---

# Feature reference

pyradix computes the same feature classes as PyRadiomics, with the same names, in both modes. A result key reads
`<image type>_<class>_<feature>`, for example `original_glcm_Contrast` or `wavelet-LLH_firstorder_Mean`.

| Class | Key | Features | Describes |
|---|---|---|---|
| [First order](firstorder.md) | `firstorder` | 19 | Distribution of intensities inside the region |
| [Shape (3D)](shape.md) | `shape` | 17 | Size and form of the region in 3D |
| [Shape (2D)](shape2d.md) | `shape2D` | 10 | Size and form in a single slice (`force2D`) |
| [GLCM](glcm.md) | `glcm` | 28 | How often pairs of grey levels occur next to each other |
| [GLRLM](glrlm.md) | `glrlm` | 16 | Runs of consecutive voxels with the same grey level |
| [GLSZM](glszm.md) | `glszm` | 16 | Connected zones of voxels with the same grey level |
| [GLDM](gldm.md) | `gldm` | 16 | How many neighbours depend on (resemble) each voxel |
| [NGTDM](ngtdm.md) | `ngtdm` | 5 | Difference between each grey level and its neighbourhood |

Features marked **deprecated** are computed for compatibility with PyRadiomics but are disabled by default there,
usually because they duplicate another feature.

## Notation

| Symbol | Meaning |
|---|---|
| \(\mathbf{X}\) | the \(N_p\) voxel intensities inside the region of interest (ROI) |
| \(N_p\) | number of voxels in the ROI |
| \(c\) | `voxelArrayShift`, added before energy-type features (default 0) |
| \(N_g\) | number of grey levels after discretisation (binning) |
| \(P(i)\), \(p(i)\) | first-order histogram of discretised grey levels, and \(p(i) = P(i)/N_p\) |
| \(\epsilon\) | machine epsilon (\(\approx 2.2\times10^{-16}\)), used inside logarithms |
| \(V_{voxel}\) | volume of one voxel in mm³ |

Texture classes first **discretise** the intensities into grey levels \(1 \dots N_g\), using a fixed bin width
(`binWidth`) or a fixed number of bins (`binCount`). How the bins are placed differs between the two standards:
see [IBSI or PyRadiomics?](../modes.md) and [Mode differences](../deviations.md).

## Differences between the modes

The formulas on these pages are the same in both modes unless a page says otherwise. What changes between `ibsi` and
`pyradiomics` is mostly *how the inputs are prepared* (resampling grid, discretisation, the surface mesh, the shape
covariance), not the feature formulas. The full list is on [Mode differences](../deviations.md).

The definitions follow the IBSI reference manual (Zwanenburg et al., *Radiology* 2020) and the PyRadiomics 3.1.0
documentation (van Griethuysen et al., *Cancer Research* 2017), which give full derivations.
