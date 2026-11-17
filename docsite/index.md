# pyradix

**Radiomics feature extraction for Python: standardised, reproducible and fast.**

You give pyradix a medical image (CT, MRI, PET…) and a **mask** that marks a region in it, usually a tumour. It returns a few hundred numbers, **radiomics features**, describing that region's size and shape, intensity and texture. Researchers use them in statistics and machine learning, for example to predict treatment response.

```python
from pyradix import RadixExtractor

result = RadixExtractor().extract("image.nii.gz", "mask.nii.gz")
result.values["original_shape_VoxelVolume"]   # one number per feature
result.provenance["config_hash"]              # fingerprint of the settings used
```

- **Two standards in one package.** IBSI by default; `mode="pyradiomics"` gives the same numbers as PyRadiomics 3.1.0, without installing it.
- **Checked against references** on Linux, macOS and Windows for every change.
- **Fast:** a Rust engine 🦀, typically 2–6× faster than PyRadiomics.
- **Reproducible:** every result records its settings.

**Start here:** [Install](install.md) · [Usage](usage.md) (sample data, many patients, command line) · [IBSI or PyRadiomics?](modes.md) · [Validation](validation.md) · [API reference](api.md)

!!! warning "Research software"
    Not intended for clinical use.
