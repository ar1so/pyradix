# Pre-processing (quoted)

Sources: IBSI reference manual / [ibsi-doc](https://github.com/theibsi/ibsi-doc) image-processing chapter; PAT1 notes in [Reference data sets](https://ibsi.readthedocs.io/en/latest/05_Reference_data_sets.html); [PyRadiomics FAQ](https://pyradiomics.readthedocs.io/en/latest/faq.html). Radix dual-mode table: [SYSTEM_DESIGN.md](internal/SYSTEM_DESIGN.md), [DEVIATIONS.md](DEVIATIONS.md).

## Interpolation grid (IBSI)

Quote (PAT1 notes):

> The size of the interpolation grid is determined by rounding the fractional grid size towards infinity, i.e. a ceiling operation.

Quote (interpolation chapter, align-grid-centers):

> We recommend aligning the interpolation grid by the center of the original image stack.

\[
n_b = \left\lceil \frac{n_a s_a}{s_b}\right\rceil
\]

\[
x_{b,w} = x_{a,w} + \frac{s_a (n_a - 1) - s_b (n_b - 1)}{2}
\]

Voxel centres of the new grid share the geometric centre of the original voxel-centre cloud. `origin` is the centre of voxel 0 (NIfTI / SimpleITK convention). Intensities outside the original grid use the nearest boundary voxel (IBSI: pad with nearest boundary intensity).

PAT1 (D2) at \(s'=2\,\mathrm{mm}\) isotropic: original shape \((n_z,n_y,n_x)=(60,201,204)\), spacing \(\approx(3.0, 0.977, 0.977)\,\mathrm{mm}\) → **(90, 99, 100)**.

## Intensity rounding (IBSI)

Quote:

> If the original image intensities are integer-valued, interpolated intensities should be rounded to the nearest integer.

CT HU on PAT1 is integer; after interpolate, Radix `ibsi` mode rounds. `pyradiomics` mode does not.

## Mask (IBSI vs PyRadiomics)

IBSI quote:

> The morphological mask is interpolated using the same interpolation algorithm as the image, after which voxels with a partial volume of at least 0.5 are included in the ROI.

Config E is the exception: image interpolator is tricubic spline, ROI interpolator is trilinear (`mask_interpolator: linear`). PyRadiomics FAQ: the mask is resampled with **nearest neighbour**; the resampling grid keeps the **original origin** (corner / first-voxel alignment), not the image centre.

## Re-segmentation (intensity vs morphological mask)

Quote:

> Re-segmentation entails updating the ROI mask based on corresponding voxel intensities. Re-segmentation […] is performed after interpolation.

Quote:

> The morphological mask is not re-segmented and maintains the original morphology […] The intensity mask can be re-segmented and will contain only the selected voxels.

Range (closed interval). IBSI config C/E: \([-1000, 400]\) HU. Outlier filter:

> The mean \(\mu\) and standard deviation \(\sigma\) of grey levels of voxels assigned to the ROI are calculated. Voxels outside the range \([\mu - 3\sigma, \mu + 3\sigma]\) are subsequently excluded from the intensity mask.

When both are set, Radix applies range first, then \(3\sigma\) on the remaining intensity mask (intersection of the two intensity ranges). Shape uses the morphological mask; first-order and texture use the intensity mask.

## Interpolators

`nearest`, `linear` (`sitkLinear`), `bspline` (`sitkBSpline`: interpolating cubic B-spline, Unser poles). IBSI config C/D use trilinear; config E uses tricubic spline on the image.

## Fixed bin size (IBSI)

Quote:

> Fixed bin size discretisation […] starting at a minimum \(X_{gl,\min}\). The minimum intensity may be a user-set value as defined by the lower bound of the re-segmentation range […] we strongly recommend to always set the same minimum value for all samples as defined by the lower bound of the re-segmentation range.

Config C: \(w_b = 25\) HU, \(X_{gl,\min} = -1000\).

## Mean filter (IBSI-2 S60F)

Quote (IBSI-2 reference manual, §4.2):

> One of the simplest existing kernels is the mean filter that computes the average intensity over its \(M\times\dots\times M\) spatial support. We restrict the definition to odd values of \(M\).

\[
g[\boldsymbol{k}]=\frac{1}{M^{D}}\quad\text{if }k_{1},\dots,k_{D}\in\bigl[-\lfloor M/2\rfloor,\lfloor M/2\rfloor\bigr]
\]

Support \(M\) is in **voxel** units (YNOF). Table 6.1 phase-1 mean tests: \(M=15\), 3D on the checkerboard (1.a.1–4) and 2D axial on the impulse (1.b.1). Padding (IBSI §2.2): zero / constant \(C=0\) (Z3VE), nearest (SIJG), periodic (Z7YO), mirror with boundary pixels included (ZDTV).

## LoG filter (IBSI-2 L6PA)

Quote (IBSI-2 reference manual, §4.3):

> The Laplacian of Gaussian (LoG) filter is rotationally invariant and commonly used to enhance blob-like structures.

\[
g_{\sigma}[\boldsymbol{k}]=-\frac{1}{\sigma^{2}}\left(\frac{1}{\sqrt{2\pi}\,\sigma}\right)^{D}\left(D-\frac{\lVert\boldsymbol{k}\rVert^{2}}{\sigma^{2}}\right)\exp\left(-\frac{\lVert\boldsymbol{k}\rVert^{2}}{2\sigma^{2}}\right)
\]

\(\sigma=\sigma^{*}/s\) in voxel units. Support \(M=1+2\lfloor d\sigma+0.5\rfloor\) with default cutoff \(d=4\). The kernel is **not** separable. Table 6.1 phase-1 LoG tests: 2.a impulse 3D \(\sigma^{*}=3.0\,\mathrm{mm}\) zero; 2.b checkerboard 3D \(\sigma^{*}=5.0\,\mathrm{mm}\) mirror; 2.c checkerboard 2D \(\sigma^{*}=5.0\,\mathrm{mm}\) mirror. IBSI troubleshooting: the LoG kernel sums to approximately 0.

PyRadiomics `getLoGImage` uses SimpleITK `LaplacianRecursiveGaussian` with `SetSigma` in millimetres and `NormalizeAcrossScale=True` — a different discrete operator. That path is not F2 (D3 / IBSI-2).

## Laws filter (IBSI-2 JTXT / PQSD)

Quote (IBSI-2 reference manual, §4.4):

> Laws kernels are a collection of five types of small 1D filters \(g[k]\). They are combined using outer products to obtain 2D and 3D filters.

Normalised 1-D kernels (deviate from Laws 1980 by the \(L^2\) factor):

\[
g_{L3}=\tfrac{1}{\sqrt{6}}[1,2,1],\quad g_{L5}=\tfrac{1}{\sqrt{70}}[1,4,6,4,1]
\]
\[
g_{E3}=\tfrac{1}{\sqrt{2}}[-1,0,1],\quad g_{E5}=\tfrac{1}{\sqrt{10}}[-1,-2,0,2,1]
\]
\[
g_{S3}=\tfrac{1}{\sqrt{6}}[-1,2,-1],\quad g_{S5}=\tfrac{1}{\sqrt{6}}[-1,0,2,0,-1]
\]
\[
g_{W5}=\tfrac{1}{\sqrt{10}}[-1,2,0,-2,1],\quad g_{R5}=\tfrac{1}{\sqrt{70}}[1,-4,6,-4,1]
\]

A name such as \(h_{E5L5S5}\) is \(g_{E5}\) along \(k_1=x\), \(g_{L5}\) along \(k_2=y\), \(g_{S5}\) along \(k_3=z\) (XKP5). Discrete convolution is \(\langle f(\boldsymbol{k}_0-\cdot),g\rangle\). Texture energy (PQSD):

\[
h_{\mathrm{energy}}[\boldsymbol{k}]=\frac{1}{W}\sum_{k_{0,1}=-\delta}^{\delta}\cdots\sum_{k_{0,d}=-\delta}^{\delta}\bigl|h[\boldsymbol{k}+\boldsymbol{k}_0]\bigr|,\quad W=(2\delta+1)^{N}
\]

i.e. a mean filter of support \(2\delta+1\) on \(|h|\) after optional orientation pooling. Rotation invariance uses Appendix A permutations and 1-D flips, then max pooling; energy is computed once, after pooling. Table 6.1: 3.a.1–3 impulse E5L5S5 zero; 3.b.1–3 checkerboard E3W5R5 mirror; \(\delta=7\) when energy is on.

PyRadiomics has no Laws image type.

## Gabor filter (IBSI-2 Q88H)

Quote (IBSI-2 reference manual, §4.5):

> In the spatial domain, 2D Gabor kernels are complex Gaussian-windowed oscillatory functions

\[
g_{\sigma,\lambda,\gamma,\theta}[\boldsymbol{k}]=\exp\left(-\frac{\tilde{k}_{1}^{2}+\gamma^{2}\tilde{k}_{2}^{2}}{2\sigma^{2}}+\mathrm{j}\frac{2\pi\tilde{k}_{1}}{\lambda}\right)
\]

with \((\tilde{k}_{1},\tilde{k}_{2})=\mathrm{R}_{\theta}\boldsymbol{k}\) and

\[
\mathrm{R}_{\theta}=\begin{pmatrix}\cos\theta&\sin\theta\\ \sin\theta&-\cos\theta\end{pmatrix}
\]

\(\theta\) turns clockwise in the axial plane \((k_1,k_2)\). \(\sigma=\sigma^{*}/s\) and \(\lambda=\lambda^{*}/s\) in voxel units. Because the kernel is complex, the published response is the modulus \(\lvert h\rvert=\lvert g\ast f\rvert\). Spatial crop (Eq. 4.5): \(M_i=N_i\) if \(N_i\) is odd, else \(N_i+1\). Rotation invariance steps \(\theta\) by \(\Delta\theta\) on \([0,\pi)\) (modulus is \(\pi\)-periodic) and average-pools. Approximate 3-D analysis averages the 2-D modulus maps over the three orthogonal planes of the image frame.

Table 6.1: 4.a.1 impulse zero \(\sigma^{*}=10\,\mathrm{mm}\), \(\lambda^{*}=4\,\mathrm{mm}\), \(\gamma=1/2\), \(\theta=\pi/3\); 4.a.2 same plus \(\Delta\theta=\pi/4\) and orthogonal-plane average; 4.b.1 sphere mirror \(\sigma^{*}=20\,\mathrm{mm}\), \(\lambda^{*}=8\,\mathrm{mm}\), \(\gamma=5/2\), \(\theta=5\pi/4\); 4.b.2 same plus \(\Delta\theta=\pi/8\) and orthogonal planes.

PyRadiomics 3.1.0 has no Gabor image type.

## Wavelets (IBSI-2 25BO / CVCQ)

Quote (IBSI-2 reference manual, §4.6.2–4.6.3):

> The undecimated transform, also called stationary transform, yields a translation-invariant decomposition by obviating the downsampling steps.

Haar (UOUE):

\[
g_{H,\mathrm{Haar}}=\Bigl[-\tfrac{1}{\sqrt{2}},\tfrac{1}{\sqrt{2}}\Bigr],\quad g_{L,\mathrm{Haar}}=\Bigl[\tfrac{1}{\sqrt{2}},\tfrac{1}{\sqrt{2}}\Bigr]
\]

Daubechies 2 and Coiflet 1 kernels are the PyWavelets analysis pair (IBSI: “from the PyWavelets website”). Even-length kernels are appended with a trailing \(0\) so they are odd (Appendix A). À trous uses the “second alternative” (zeros after every tap, matching MATLAB / PyWavelets). \(h_{LHL}\) is \(g_L\) along \(k_1=x\), \(g_H\) along \(k_2=y\), \(g_L\) along \(k_3=z\). Level \(j>1\) filters the previous LLL approximation with the dilated kernels. Rotation invariance: Appendix A, then average pooling. Table 6.1: 5.a db2 impulse LHL level 1; 6.a coif1 sphere HHL level 1; 7.a Haar checkerboard LLL/HHH level 2.

PyRadiomics `getWaveletImage` defaults to `coif1` stationary wavelet without Appendix A pooling. That path is not the D3 / IBSI-2 gate.
