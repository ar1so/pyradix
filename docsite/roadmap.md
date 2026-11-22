---
description: "pyradix roadmap: planned GPU acceleration for radiomics feature extraction with CUDA and Apple Metal, batch extraction and PyTorch tensors. Not yet available."
---

# Roadmap: GPU acceleration

!!! warning "Planned — not yet available"
    pyradix runs on the CPU today. Nothing on this page is in a release yet. We will publish speed numbers only
    once they are measured, with the hardware and settings used.

## Why

Radiomics studies are growing: thousands of patients, several regions each, and dozens of filtered images per region
(LoG, wavelets, Laws, Gabor). Texture matrices are the slowest part, and they are a good fit for a GPU. The goal is to
make such studies take minutes rather than hours, without changing a single feature value.

## What we plan

| Area | Plan |
|---|---|
| **Backends** | CUDA for NVIDIA GPUs; Apple Metal for Apple-silicon Macs. The CPU engine stays the default and the reference. |
| **Batch extraction** | One call for many patients and regions of interest, so the GPU stays busy. |
| **Texture matrices** | GLCM, GLRLM, GLSZM, GLDM and NGTDM built on the GPU. |
| **Filters and maps** | Image filters and voxel-wise feature maps on the GPU. |
| **PyTorch** | Optional tensor input and output, so features can be computed inside a deep-learning pipeline without copying data back to the CPU. |
| **Install** | An optional extra; `pip install pyradix-radiomics` keeps working exactly as it does now. |

## The rule: identical numbers

A GPU result is only useful if it is the same result. Every GPU backend must pass the same checks as the CPU engine
before it is released:

- IBSI 1 feature values and IBSI 2 filter responses ([Validation](validation.md));
- PyRadiomics 3.1.0 values in `pyradiomics` mode;
- the CPU engine on the same inputs, within the tolerances used today.

The provenance of every result will record which backend computed it.

## Help shape it

Tell us your use case (study size, modality, filters) or offer hardware to test on: open an
[issue](https://github.com/ar1so/pyradix/issues).
