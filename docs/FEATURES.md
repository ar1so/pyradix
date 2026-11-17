# Feature catalog (quoted)

Source: [PyRadiomics Radiomic Features](https://pyradiomics.readthedocs.io/en/latest/features.html) (current docs). IBSI notes are the docs’ own notes, not ours.

PyRadiomics classes (quote):

> They are subdivided into the following classes:
>
> - First Order Statistics (19 features)
> - Shape-based (3D) (16 features)
> - Shape-based (2D) (10 features)
> - Gray Level Co-occurrence Matrix (24 features)
> - Gray Level Run Length Matrix (16 features)
> - Gray Level Size Zone Matrix (16 features)
> - Neighbouring Gray Tone Difference Matrix (5 features)
> - Gray Level Dependence Matrix (14 features)

> Most features defined below are in compliance with feature definitions as described by the Imaging Biomarker Standardization Initiative (IBSI) […] Where features differ, a note has been added specifying the difference.

Radix implements **every named getter below**. Result keys: `original_<class>_<Name>` (same as PyRadiomics).

Deprecated GLCM getters (`Dissimilarity`, `Homogeneity1`, `Homogeneity2`, `SumAverage`, `SumVariance`) are computed and prefixed the same way; they are still in the official page.

---

## First Order (`firstorder`) — 19

Quote: “First-order statistics describe the distribution of voxel intensities within the image region defined by the mask through commonly used and basic metrics.”

| # | Name | Quote (docs) |
|---|---|---|
| 1 | Energy | “Energy is a measure of the magnitude of voxel values in an image.” \(\sum (X(i)+c)^2\) |
| 2 | TotalEnergy | “Total Energy is the value of Energy feature scaled by the volume of the voxel in cubic mm.” **Not present in IBSI.** |
| 3 | Entropy | “Entropy specifies the uncertainty/randomness in the image values.” IBSI: Intensity Histogram Entropy. |
| 4 | Minimum | \(\min(X)\) |
| 5 | 10Percentile | “The 10th percentile of \(X\)” |
| 6 | 90Percentile | “The 90th percentile of \(X\)” |
| 7 | Maximum | “The maximum gray level intensity within the ROI.” |
| 8 | Mean | “The average gray level intensity within the ROI.” |
| 9 | Median | “The median gray level intensity within the ROI.” |
| 10 | InterquartileRange | \(P_{75}-P_{25}\) |
| 11 | Range | \(\max(X)-\min(X)\) |
| 12 | MeanAbsoluteDeviation | “mean distance of all intensity values from the Mean Value” |
| 13 | RobustMeanAbsoluteDeviation | MAD on intensities between the 10th and 90th percentile |
| 14 | RootMeanSquared | “square-root of the mean of all the squared intensity values” (uses \(c\)) |
| 15 | StandardDeviation | “not enabled by default”; **“Not present in IBSI feature definitions (correlated with variance)”** |
| 16 | Skewness | \(\mu_3/\sigma^3\); flat ROI → 0 |
| 17 | Kurtosis | \(\mu_4/\sigma^4\). Quote: “The IBSI feature definition implements excess kurtosis […] The PyRadiomics kurtosis is not corrected, yielding a value 3 higher than the IBSI kurtosis.” |
| 18 | Variance | population variance \(\frac{1}{N_p}\sum (X-\bar X)^2\) |
| 19 | Uniformity | \(\sum p(i)^2\). IBSI: Intensity Histogram Uniformity. |

\(c\) = `voxelArrayShift`. Quote: “In the IBSI feature definitions, no correction for negative gray values is implemented. To achieve similar behaviour in PyRadiomics, set `voxelArrayShift` to 0.” Radix default \(c=0\).

---

## Shape 3D (`shape`) — 16 listed (+ Compactness 1/2 / SphericalDisproportion on the page)

Quote: “descriptors of the three-dimensional size and shape of the ROI. These features are independent from the gray level intensity distribution […] only calculated on the non-derived image and mask.”

| # | Name | Quote (docs) |
|---|---|---|
| 1 | MeshVolume | tetrahedra from mesh faces; IBSI `Volume` |
| 2 | VoxelVolume | \(N_v \times V_k\); IBSI `Approximate Volume` |
| 3 | SurfaceArea | sum of triangle areas; IBSI `Surface Area` |
| 4 | SurfaceVolumeRatio | \(A/V\) |
| 5 | Sphericity | \(\sqrt[3]{36\pi V^2}/A\) |
| 6 | Compactness1 | not default (redundant) |
| 7 | Compactness2 | not default (redundant) |
| 8 | SphericalDisproportion | inverse of sphericity; not default |
| 9 | Maximum3DDiameter | “largest pairwise Euclidean distance between tumor surface mesh vertices” |
| 10 | Maximum2DDiameterSlice | axial (row-column) plane |
| 11 | Maximum2DDiameterColumn | coronal (row-slice) plane |
| 12 | Maximum2DDiameterRow | sagittal (column-slice) plane |
| 13 | MajorAxisLength | \(4\sqrt{\lambda_{major}}\) from voxel-center PCA |
| 14 | MinorAxisLength | \(4\sqrt{\lambda_{minor}}\) |
| 15 | LeastAxisLength | \(4\sqrt{\lambda_{least}}\) |
| 16 | Elongation | \(\sqrt{\lambda_{minor}/\lambda_{major}}\) |
| 17 | Flatness | \(\sqrt{\lambda_{least}/\lambda_{major}}\) |

The page numbers 16 features in the intro and then lists Compactness/SphericalDisproportion as extra named getters. Radix computes all of them.

---

## Shape 2D (`shape2D`) — 10

Quote: “This class can only be calculated for truly 2D masks.” (`force2D` / size-1 out-of-plane). Radix runs this class when any array axis has size 1.

| # | Name | Quote |
|---|---|---|
| 1 | MeshSurface | signed mesh area |
| 2 | PixelSurface | \(N_p \times\) pixel area |
| 3 | Perimeter | mesh circumference |
| 4 | PerimeterSurfaceRatio | \(P/A\) |
| 5 | Sphericity | \(2\sqrt{\pi A}/P\) |
| 6 | SphericalDisproportion | inverse; not default |
| 7 | MaximumDiameter | largest pairwise vertex distance |
| 8 | MajorAxisLength | \(4\sqrt{\lambda_{major}}\) |
| 9 | MinorAxisLength | \(4\sqrt{\lambda_{minor}}\) |
| 10 | Elongation | \(\sqrt{\lambda_{minor}/\lambda_{major}}\) |

---

## GLCM (`glcm`) — 24 on the page (incl. deprecated)

Quote: “A Gray Level Co-occurrence Matrix (GLCM) […] The \((i,j)^{\text{th}}\) element […] number of times the combination of levels \(i\) and \(j\) occur in two pixels […] separated by a distance of \(\delta\) pixels along angle \(\theta\).”

> Note that pyradiomics by default computes symmetrical GLCM!

| # | Name | Notes from docs |
|---|---|---|
| 1 | Autocorrelation | \(\sum p(i,j)\,ij\) |
| 2 | JointAverage | \(\mu_x=\sum p(i,j)\,i\) |
| 3 | ClusterProminence | \((i+j-\mu_x-\mu_y)^4\) |
| 4 | ClusterShade | cubed |
| 5 | ClusterTendency | squared |
| 6 | Contrast | \((i-j)^2 p(i,j)\) |
| 7 | Correlation | 1 if flat ROI |
| 8 | DifferenceAverage | from \(p_{x-y}\) |
| 9 | DifferenceEntropy | |
| 10 | DifferenceVariance | |
| — | Dissimilarity | **deprecated** (equals DifferenceAverage) |
| 11 | JointEnergy | \(\sum p(i,j)^2\) (Angular Second Moment) |
| 12 | JointEntropy | |
| — | Homogeneity1 | **deprecated** |
| — | Homogeneity2 | **deprecated** (equals IDM) |
| 13 | Imc1 | informational measure of correlation 1 |
| 14 | Imc2 | informational measure of correlation 2 |
| 15 | Idm | Inverse Difference Moment |
| 16 | MCC | \(\sqrt{\lambda_2(Q)}\) (2nd-largest eigenvalue of Q) |
| 17 | Idmn | IDM normalized |
| 18 | Id | Inverse Difference |
| 19 | Idn | Inverse Difference Normalized |
| 20 | InverseVariance | |
| 21 | MaximumProbability | |
| — | SumAverage | **deprecated** (2 × JointAverage if symmetric) |
| — | SumVariance | **deprecated** |
| 22 | SumEntropy | |
| 23 | SumSquares | |

(The “24 features” count in the intro is the non-deprecated set.)

---

## GLSZM (`glszm`) — 16

Quote: size-zone matrix: zones of connected voxels sharing a gray level (26-connected in 3D).

SmallAreaEmphasis, LargeAreaEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, SizeZoneNonUniformity, SizeZoneNonUniformityNormalized, ZonePercentage, GrayLevelVariance, ZoneVariance, ZoneEntropy, LowGrayLevelZoneEmphasis, HighGrayLevelZoneEmphasis, SmallAreaLowGrayLevelEmphasis, SmallAreaHighGrayLevelEmphasis, LargeAreaLowGrayLevelEmphasis, LargeAreaHighGrayLevelEmphasis.

---

## GLRLM (`glrlm`) — 16

Quote: run-length matrix along 13 directions in 3D (default), then features averaged.

ShortRunEmphasis, LongRunEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, RunLengthNonUniformity, RunLengthNonUniformityNormalized, RunPercentage, GrayLevelVariance, RunVariance, RunEntropy, LowGrayLevelRunEmphasis, HighGrayLevelRunEmphasis, ShortRunLowGrayLevelEmphasis, ShortRunHighGrayLevelEmphasis, LongRunLowGrayLevelEmphasis, LongRunHighGrayLevelEmphasis.

---

## NGTDM (`ngtdm`) — 5

Quote: neighbouring gray-tone difference matrix (Amadasun and King).

Coarseness, Contrast, Busyness, Complexity, Strength.

---

## GLDM (`gldm`) — 14 (+ DependencePercentage on the page)

SmallDependenceEmphasis, LargeDependenceEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, DependenceNonUniformity, DependenceNonUniformityNormalized, GrayLevelVariance, DependenceVariance, DependenceEntropy, DependencePercentage, LowGrayLevelEmphasis, HighGrayLevelEmphasis, SmallDependenceLowGrayLevelEmphasis, SmallDependenceHighGrayLevelEmphasis, LargeDependenceLowGrayLevelEmphasis, LargeDependenceHighGrayLevelEmphasis.

---

## Not features (filters)

Same docs index: LoG, Wavelet, Square, Square Root, Logarithm, Exponential, Gradient, LBP 2D/3D. Those **derive images**; then the classes above are computed again. Filter image types are a later board column (`F1`–`F6`). This file is the **feature class** set.

## Dual-mode notes we honour

1. Kurtosis: excess in `ibsi`, raw in `pyradiomics` (docs quote above).
2. Fixed-bin-size edges: IBSI from ROI min; PyRadiomics “equally spaced from 0” (FAQ).
3. `voxelArrayShift` default 0.
4. Resample grid, intensity rounding, mask interpolator: [DEVIATIONS.md](DEVIATIONS.md) and [PREPROCESS.md](PREPROCESS.md).
