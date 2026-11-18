# Changelog

## 0.1.1

- `pyradiomics` mode now bins intensities exactly as PyRadiomics 3.1.0 does: bin
  edges from `numpy.histogram` (fixed bin count) or `numpy.arange` (fixed bin
  width), voxels assigned with `numpy.digitize`. The closed-form formula used
  before put voxels lying exactly on a bin edge one bin higher. This happens
  often on integer images after normalisation: on BraTS 2021 MRI with
  `binCount: 32`, about 18% of regions differed. They now match PyRadiomics
  3.1.0 on every feature. `ibsi` mode keeps the IBSI formula, unchanged.

## 0.1.0

First packaged release.

- Wheels for CPython 3.10–3.13 on Linux x86_64, Linux aarch64 (manylinux_2_28), macOS x86_64 and arm64 (universal2), and Windows x64, plus an sdist.
- Linux links OpenBLAS for GLCM MCC (`dgeev`); macOS links Accelerate. Windows wheels use faer for MCC eigenvalues (no OpenBLAS/vcpkg build).
- NIfTI and NRRD loads copy the array and close the file, so a temp path can be deleted on Windows. SimpleITK paths go through `image_io_path`.
- Parameter files accept a UTF-8 BOM and CR LF. CLEAR YAML is written with LF line endings.
- Feature extraction releases the GIL so native work does not hold the interpreter lock.
- `pyradix` command line (`pyradix extract IMAGE MASK --out features.csv`) and a Dockerfile.
- Large masks are much faster: the 26-connected volume count runs in Rust, and mesh diameters skip vertex pairs that cannot be the farthest. Full IBSI lung CT mask (125,256 voxels): 2.0 s to 0.18 s, with identical values.
- `pyradiomics` mode matches PyRadiomics 3.1.0 resampling on anisotropic grids (`resampledPixelSpacing` is read as x, y, z) and keeps fractional values for float images, as SimpleITK does.
- NaN or infinite voxels inside the ROI raise a clear error instead of a panic.
