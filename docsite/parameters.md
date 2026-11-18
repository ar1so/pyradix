# Parameters

pyradix reads its settings from a **parameter file** (YAML or JSON) or from a `Params` object in Python. Two file
formats are accepted:

- **PyRadiomics parameter files**, unchanged: `setting:`, `imageType:` and `featureClass:` blocks, PyRadiomics names.
  Load them with `RadixExtractor.from_pyradiomics_yaml(path)` or `pyradix extract … --params path`.
- **pyradix parameter files**: flat keys with the pyradix names below, plus `mode:`. Load them with
  `Params.from_yaml(path)`.

```yaml
# PyRadiomics file
setting:
  binWidth: 25
  resampledPixelSpacing: [1, 1, 1]   # x, y, z
  interpolator: sitkBSpline
  label: 1
imageType:
  Original: {}
  LoG:
    sigma: [1.0, 2.0, 3.0]
featureClass:
  firstorder: []                     # [] = every feature of the class
  glcm: [Contrast, Correlation]
```

```yaml
# pyradix file
mode: ibsi
bin_width: 25.0
resampled_pixel_spacing: [2.0, 2.0, 2.0]   # z, y, x
interpolator: linear
resegment_range: [-1000, 400]
image_types: [Original]
label: 1
```

!!! warning "Axis order of `resampledPixelSpacing`"
    PyRadiomics files list spacing as **x, y, z** (SimpleITK order); pyradix's own files and `Params` use
    **z, y, x**. pyradix converts PyRadiomics files for you.

## All settings

| PyRadiomics name | pyradix name | Default | Meaning |
|---|---|---|---|
| `binWidth` | `bin_width` | `25.0` | Width of each intensity bin (fixed bin size). Ignored when `binCount` is set. |
| `binCount` | `bin_count` | not set | Number of intensity bins (fixed bin number). Overrides `binWidth`. Common for MRI. |
| `resampledPixelSpacing` | `resampled_pixel_spacing` | not set | Resample image and mask to this voxel spacing in mm. A `0` keeps that axis. Not set = no resampling. |
| `interpolator` | `interpolator` | `bspline` | Image interpolator for resampling: `sitkBSpline`, `sitkLinear`, `sitkNearestNeighbor` (also `bspline`, `linear`, `nearest`). |
| `label` | `label` | `1` | Mask value that marks the region of interest. |
| `label_channel` | `label_channel` | not set | Channel to read the label from, for multi-channel (vector) masks. |
| `voxelArrayShift` | `voxel_array_shift` | `0.0` | Constant added to intensities before Energy, TotalEnergy and RootMeanSquared. |
| `resegmentRange` | `resegment_range` | not set | `[min, max]`: keep only ROI voxels with intensities in this range (one bound may be omitted). |
| `removeOutliers` | `outlier_sigma` | not set | Remove ROI voxels further than this many standard deviations from the ROI mean. |
| `normalize` | `normalize` | `false` | Z-score normalise the image before extraction. |
| `normalizeScale` | `normalize_scale` | `1.0` | Multiply the normalised image by this factor. |
| `padDistance` | `pad_distance` | `5` | Voxels of padding kept around the ROI when cropping (needed by filters). |
| `preCrop` | `pre_crop` | `false` | Crop the image to the ROI before applying filters (faster, same results with enough padding). |
| `resegmentMode` | `resegment_mode` | `absolute` | How `resegmentRange` is read: `absolute` values, `relative` to the ROI, or `sigma` (standard deviations). |
| `resegmentShape` | `resegment_shape` | `false` | Also apply resegmentation to the mask used for shape features. |
| `minimumROIDimensions` | `minimum_roi_dimensions` | `2` | The ROI must span at least this many dimensions (1–3), or extraction stops with an error. |
| `minimumROISize` | `minimum_roi_size` | not set | The ROI must contain at least this many voxels. |
| `correctMask` | `correct_mask` | `false` | If image and mask geometry differ slightly, resample the mask onto the image grid instead of failing. |
| `force2D` | `force_2d` | `false` | Compute texture features slice by slice in 2D, and 2D shape features. |
| `force2Ddimension` | `force_2d_dimension` | `0` | Axis the 2D slices are taken along: `0` axial, `1` coronal, `2` sagittal. |
| `distances` | `distances` | `[1]` | Neighbour distances for the texture matrices. |
| `weightingNorm` | `weighting_norm` | not set | Merge GLCM directions with a distance weighting (`manhattan`, `euclidean`, `infinity`, `no_weighting`) instead of averaging features over directions. |
| `symmetricalGLCM` | `symmetrical_glcm` | `true` | Count every voxel pair in both directions (symmetrical GLCM). |
| `gldm_a` | `gldm_a` | `0` | GLDM tolerance α: neighbours within this grey-level difference count as dependent. |
| `additionalInfo` | `additional_info` | `true` | Add diagnostic entries (versions, settings, image and mask statistics) to the result. |
| `geometryTolerance` | `geometry_tolerance` | not set | Tolerance when checking that image and mask have the same geometry. |
| `sigma` | `sigma` | not set | LoG filter: list of σ values in mm, one filtered image each. |
| `wavelet` | `wavelet` | not set | Wavelet filter: wavelet family (default `coif1`). |
| `start_level` | `start_level` | not set | Wavelet filter: first decomposition level returned (default 0). |
| `level` | `level` | not set | Wavelet filter: number of decomposition levels (default 1). |
| `gradientUseSpacing` | `gradient_use_spacing` | not set | Gradient filter: use the voxel spacing in the gradient (default true). |
| `lbp2DRadius` | `lbp2d_radius` | not set | LBP2D filter: radius in voxels (default 1). |
| `lbp2DSamples` | `lbp2d_samples` | not set | LBP2D filter: number of samples on the circle (default 8). |
| `lbp2DMethod` | `lbp2d_method` | not set | LBP2D filter: method, e.g. `uniform` (default), `default`, `ror`, `var`. |
| `lbp3DLevels` | `lbp3d_levels` | not set | LBP3D filter: number of spherical-harmonic levels (default 2). |
| `lbp3DIcosphereRadius` | `lbp3d_icosphere_radius` | not set | LBP3D filter: icosphere radius in voxels (default 1). |
| `lbp3DIcosphereSubdivision` | `lbp3d_icosphere_subdivision` | not set | LBP3D filter: icosphere subdivisions (default 1). |

## pyradix-only settings

| pyradix name | Default | Meaning |
|---|---|---|
| `mode` | `ibsi` | The standard to follow: `ibsi` or `pyradiomics`. See [IBSI or PyRadiomics?](modes.md). |
| `mask_interpolator` | depends on mode | Interpolator for the mask when resampling. Default: nearest neighbour in `pyradiomics` mode, the image interpolator (then partial volume ≥ 0.5) in `ibsi` mode. |
| `bin_min` | not set | IBSI fixed bin size: lower edge of the first bin (usually the lower bound of the resegmentation range). |
| `feature_classes` | all | Which feature classes and features to compute, e.g. `{"glcm": ["Contrast"], "firstorder": []}`. |
| `voxel_setting` | `kernelRadius: 1`, `maskedKernel: true`, `initValue: 0` | Settings for [voxel-based feature maps](usage.md#voxel-based-feature-maps). |
| `voxel_batch` | not set | Voxel maps: number of voxels computed per batch, to limit memory. |

## Image types

`imageType` (PyRadiomics files) or `image_types` (pyradix files) lists the images features are computed on: the
original image and any filtered images. See [Image filters](filters.md).
