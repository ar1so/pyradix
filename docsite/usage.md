# Usage

## Your input files

- NIfTI (`.nii`, `.nii.gz`) or NRRD. DICOM is not read directly: convert it first, for example with [dcm2niix](https://github.com/rordenlab/dcm2niix).
- The mask must line up with the image (same size, spacing and position). The region is the voxels with value `1`, or the `label` you pass.

## Try it on a sample case

```bash
curl -LO https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/master/data/brain1_image.nrrd
curl -LO https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/master/data/brain1_label.nrrd
pyradix extract brain1_image.nrrd brain1_label.nrrd --out features.csv
```

## Many patients, one table

```python
from pathlib import Path
import pandas as pd
from pyradix import RadixExtractor

extractor = RadixExtractor(mode="ibsi")
rows = [
    {"case": case.name, **extractor.extract(case / "image.nii.gz", case / "mask.nii.gz").values}
    for case in sorted(Path("cases").iterdir())
]
pd.DataFrame(rows).set_index("case").to_csv("features.csv")
```

## Command line

```bash
pyradix extract IMAGE MASK [--mode ibsi|pyradiomics] [--params FILE] [--label N] [--out FILE.csv|FILE.json] [--provenance]
```

## Extract features

```python
from pyradix import RadixExtractor

result = RadixExtractor(mode="ibsi").extract("image.nrrd", "mask.nrrd")
result.values        # {feature name: float}, 108 features with the defaults
result.provenance    # mode, settings, filters, config_hash, ROI size …
```

`image` and `mask` can be NIfTI (`.nii`, `.nii.gz`) or NRRD paths, or `Volume` objects. Use `label=` on `extract` when the mask holds more than one label.

## Use a PyRadiomics parameter file

```python
extractor = RadixExtractor.from_pyradiomics_yaml("params.yaml")
result = extractor.extract("image.nrrd", "mask.nrrd")
```

This sets `mode="pyradiomics"`. JSON parameter files load with `Params.from_pyradiomics_json`.

## Choose settings in code

```python
from pyradix import Params, RadixExtractor

params = Params(
    mode="pyradiomics",
    bin_width=25.0,
    label=1,
    feature_classes={"glcm": ["Contrast", "MCC"], "firstorder": []},  # [] = whole class
)
result = RadixExtractor(mode="pyradiomics", params=params).extract("image.nrrd", "mask.nrrd")
```

## Load volumes yourself

```python
from pyradix import load_volume

image = load_volume("image.nii.gz")
image.array, image.spacing, image.origin, image.direction
```

## Voxel-based feature maps

```python
from pyradix import extract_voxel_maps

maps = extract_voxel_maps(
    image.array, mask.array,
    mode="pyradiomics", kernel_radius=1, spacing=image.spacing,
    features=("Mean", "Entropy"),   # first-order feature names
)
```

For texture classes, pass a `Params` with `feature_classes` as `params=`. `voxel_batch` limits memory on large images.

## Report the run

```python
from pyradix import dump_clear_yaml

dump_clear_yaml(result, "radiomics_report.yaml")
```

The YAML lists software and version, mode, discretization, resampling and filters, so a reviewer can see exactly how the numbers were made.

## Feature stability under perturbation

`perturbation_table(extractor, image, mask, n=5)` extracts `n` perturbed copies of the case (an integer shift of the image plus optional noise) and returns one row per feature with the mean, standard deviation and coefficient of variation. Use the coefficient of variation to drop features that move under small perturbations.

## Windows paths and SimpleITK

pyradix reads paths with spaces on every platform. If you also give the same path to SimpleITK, pass it through `pyradix.image_io_path(path)` first.
