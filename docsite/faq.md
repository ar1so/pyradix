---
description: "pyradix FAQ: PyRadiomics compatibility, IBSI, Python 3.13 support, DICOM input, binning, speed and common radiomics extraction errors."
---

# FAQ

## Is pyradix a drop-in replacement for PyRadiomics?

For your settings and results, yes; for your code, almost. `RadixExtractor.from_pyradiomics_yaml("params.yaml")` reads
an existing PyRadiomics parameter file unchanged, and in `pyradiomics` mode the feature names and values match
PyRadiomics 3.1.0. You change the import and one call:

```python
from pyradix import RadixExtractor

extractor = RadixExtractor.from_pyradiomics_yaml("params.yaml")   # was RadiomicsFeatureExtractor("params.yaml")
features = extractor.extract("image.nii.gz", "mask.nii.gz").values  # was extractor.execute(...)
```

## Does pyradix support Python 3.13?

Yes. Ready-made wheels exist for CPython 3.10, 3.11, 3.12 and 3.13 on Linux (x86_64, aarch64), macOS (Intel and
Apple silicon) and Windows x64, so `pip install pyradix-radiomics` needs no compiler.

## Is pyradix IBSI-compliant?

The default `ibsi` mode follows the IBSI reference manual. Every change is tested against the published IBSI 1
feature values and IBSI 2 filter responses; see [Validation](validation.md).

## Does pyradix run on the GPU?

Not yet. pyradix runs on the CPU today. GPU acceleration (CUDA and Apple Metal) is planned; see the
[Roadmap](roadmap.md).

## Which mode should I use?

The default, `ibsi`, for new studies and anything multi-centre. Use `pyradiomics` when you must match numbers from
a study that used PyRadiomics 3.1.0. Do not mix modes within one study. See [IBSI or PyRadiomics?](modes.md).

## Do I need PyRadiomics installed for `mode="pyradiomics"`?

No. pyradix computes PyRadiomics 3.1.0 values itself and never imports PyRadiomics.

## Will I get exactly the same numbers as PyRadiomics?

In `pyradiomics` mode, yes, to within \(10^{-4}\) and usually bit-for-bit: on 1,251 BraTS 2021 brain MRI cases,
every feature of every tumour region matched PyRadiomics 3.1.0. Two things outside pyradix can still change values:

- **Your preprocessing.** If you normalise or resample images yourself, do it once and give every tool the same
  saved files. Even different NumPy versions can change the last digit of a normalised image, enough to move a voxel
  into a neighbouring bin.
- **Settings.** Check `binWidth`/`binCount`, `voxelArrayShift`, resampling and `label`; they must be the same.

## Can pyradix read DICOM?

Not directly. Convert the image series to NIfTI first, for example with
[dcm2niix](https://github.com/rordenlab/dcm2niix). Masks stored as DICOM RTSTRUCT can be converted with
[plastimatch](https://plastimatch.org) or [rt-utils](https://github.com/qurit/rt-utils).

## I get "label 2 is not in the mask"

Your region uses a different value than the one requested. The message lists the labels that are present; pass one
of them with `label=` (Python) or `--label` (command line).

## I get "image and mask geometry disagree"

The mask has a different size, spacing or position than the image. Use the mask saved from the same scan, resample
it onto the image grid, or set `correctMask: true` for small differences.

## I get "ROI has 0 dimension(s), minimumROIDimensions is 2"

The region is a single voxel or a line. Check the mask. For single-slice masks, use `force2D: true`.

## Some voxels are NaN

pyradix stops with a clear message when voxels inside the ROI are NaN or infinite. Fill them, or remove them from the
mask, before extraction.

## Should I use `binWidth` or `binCount`?

For CT (Hounsfield units), a fixed bin width such as 25 HU is common. For MRI and other images without absolute units,
normalise the image and use a fixed bin count such as 32. Report your choice: it changes texture values.

## How fast is pyradix?

The engine is written in Rust. In `pyradiomics` mode it is typically 2–3× faster than PyRadiomics 3.1.0 on brain MRI
regions; `ibsi` mode does more work per region and is currently about as fast as PyRadiomics. Speed depends on the size
of the region and the settings.

## How do I report the settings I used?

Every result has `result.provenance` (all settings and a configuration hash), and `dump_clear_yaml(result, path)`
writes them in the structure of the CLEAR reporting checklist. See [Citing](citing.md).
