# Mode differences

Everything that `ibsi` and `pyradiomics` mode do differently, with the source of each rule. Feature names are the same in both modes.

| Policy | `ibsi` (default) | `pyradiomics` | Source |
|---|---|---|---|
| Resample grid | Centre of the original stack; \(n'=\lceil n s / s'\rceil\) | Origin of voxel 0 kept (corner-aligned) | IBSI: “aligning the interpolation grid by the center of the original image stack.” PAT1 notes: ceiling grid size. PyRadiomics FAQ: grid follows SimpleITK origin. |
| After interpolate | Round integer-valued images to nearest integer | No rounding | IBSI: “interpolated intensities should be rounded to the nearest integer.” |
| Mask resample | Same interpolator (or `mask_interpolator`), then partial volume \(\ge 0.5\) | Always nearest neighbour | IBSI morphological mask note; PyRadiomics FAQ. |
| Resegmentation | Dual masks: morph unchanged; intensity = range \(\cap\) \(k\sigma\) | Optional `resegmentRange` / `removeOutliers`; shape stays morph unless `resegmentShape` | IBSI: “The morphological mask is not re-segmented […] The intensity mask can be re-segmented.” |
| Fixed-bin-size edges | From resegmentation min | From 0: \(\lfloor X/w_b\rfloor-\lfloor X_\min/w_b\rfloor+1\) | IBSI: FBS “starting at […] the lower bound of the re-segmentation range.” PyRadiomics FAQ “equally spaced from 0”. |
| Bin assignment | IBSI formulas: FBS \(\lfloor (X-X_\min)/w_b\rfloor+1\); FBN \(\lfloor N_g (X-X_\min)/(X_\max-X_\min)\rfloor+1\) | The same edges and steps as PyRadiomics 3.1.0: `numpy.arange` (bin width) or `numpy.histogram` (bin count), then `numpy.digitize`. A voxel exactly on a bin edge goes where numpy puts it (since 0.1.1). | IBSI reference manual, discretisation; PyRadiomics `imageoperations.getBinEdges` / `binImage`. |
| Kurtosis | Excess (\(-3\); Gaussian \(=0\)) | Raw (Gaussian \(=3\)) | PyRadiomics first-order docs: “The PyRadiomics kurtosis is not corrected, yielding a value 3 higher than the IBSI kurtosis.” |
| PCA covariance | Divide by \(N-1\) | Divide by \(N\) (`physicalCoordinates /= sqrt(Np)` then Gram) | PyRadiomics `shape.py` 3.1.0. IBSI digital phantom matches \(N-1\). |
| Surface mesh | Lorensen 256-cube, 0.5-iso | PyRadiomics `cshape.c` 128-cube (symmetry-folded) | PyRadiomics `cshape.c` `calculate_coefficients`. |
| 2D diameter | Same-slice: pair mesh vertices only when the dropped-axis coordinate is equal. IBSI does not define this feature | same rule, on the PyRadiomics mesh | PyRadiomics `calculate_meshDiameter`. |
| Default wavelet | IBSI-2 Haar/db2/coif1, odd-pad, Appendix A average | `coif1` SWT, no Appendix A pooling | IBSI-2 §4.6; PyRadiomics `getWaveletImage`. |
| LoG | IBSI-2 Eq. 4.2 discrete kernel; \(\sigma=\sigma^{*}/s\); cutoff \(4\sigma\) | SimpleITK `LaplacianRecursiveGaussian` + `NormalizeAcrossScale` | IBSI-2 L6PA; PyRadiomics `imageoperations.getLoGImage`. |
| Laws | IBSI-2 JTXT kernels + Appendix A max pooling + PQSD energy | not implemented | PyRadiomics 3.1.0 has no Laws image type. |
| Gabor | IBSI-2 Eq. 4.3 2-D modulus; \(\Delta\theta\) average; optional three-plane average | not implemented | PyRadiomics 3.1.0 has no Gabor image type. |
| NGTDM neighbourhood | Count a voxel only when it has at least one ROI neighbour (\(W_k \neq 0\)) | C code still counts a voxel with no ROI neighbour and adds a difference of 0 | IBSI: “we consider a valid neighbourhood to exist if there is at least one neighbouring voxel included in the ROI mask.” PyRadiomics `ngtdm.py` documents the same \(N_{v,p}\) rule; the C matrix does not follow it. |
| NGTDM coarseness | \(1/\sum p_i s_i\), capped at \(10^6\) | same cap | IBSI QCDE: “Because \(\sum p_i s_i\) potentially evaluates to 0, the maximum coarseness value is set to an arbitrary number of \(10^6\).” PyRadiomics `ngtdm.py` uses the same cap. |

`voxelArrayShift` default is 0 in both modes (IBSI has no negative-value shift). Set it in YAML to match an old PyRadiomics run.
