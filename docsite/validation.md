---
description: "How pyradix radiomics features are validated against IBSI 1, IBSI 2 and PyRadiomics 3.1.0 reference values on Linux, macOS and Windows."
---

# Validation

Every CI run, on Linux, macOS and Windows, downloads small public datasets and compares pyradix with reference values.

| Check | Data | Result |
|---|---|---|
| IBSI 1 features | IBSI digital phantom; IBSI lung CT, configs C, D, E | Match the published IBSI tables |
| IBSI 2 filters | IBSI 2 digital phantoms | Match the IBSI 2 reference responses |
| PyRadiomics 3.1.0, frozen vector | brain1 | All 107 default features within 1e-4 |
| PyRadiomics 3.1.0, live | brain1, IBSI digital phantom, IBSI lung CT (PAT1) | 107 / 107 within 1e-4 on each |
| PyRadiomics 3.1.0, live, with filters | brain1, PAT1: Original + LoG (σ 1, 2, 3 mm) + 8 wavelet bands | 1,209 / 1,209 on each |
| PyRadiomics 3.1.0, live, with resampling | brain1, PAT1 at 1, 2, 3 mm and anisotropic grids | brain1 944 / 944; PAT1 944 / 944 except a few wavelet values within 5e-4 at 3 mm |

Integer features must match exactly. Other values must agree within 1e-4, as an absolute or relative difference, whichever is looser.

Run the checks yourself:

```bash
python scripts/fetch_validation_data.py
python -m pytest -q tests
```

## Speed

One extraction after loading, release build, same settings in both libraries, Apple silicon.

| Volume | ROI voxels | pyradix | PyRadiomics 3.1.0 |
|---|---|---|---|
| brain1, `binWidth` 25 | 4,137 | 73 ms | 194 ms |
| brain1, `binWidth` 5 | 4,137 | 167 ms | 864 ms |
| IBSI lung CT (PAT1), full mask | 125,256 | 0.18 s | 1.1–1.3 s |
| Lung1 cohort, 421 tumours, median | 13,616 | 0.85 s | 2.94 s |

## Known gaps

- **Lung1 cohort comparison needs a re-run.** The September 2026 run used an older shape mesh, which put `Sphericity` and `SurfaceVolumeRatio` just over the tolerance in about 60% of tumours. The current mesh matches PyRadiomics 3.1.0 on every test case; the 421-tumour run has not been repeated yet.
- **Wavelet features after coarse resampling.** On PAT1 at 3 mm or 1.5 × 1.5 × 4 mm, 6–11 of 944 wavelet values differ by 2–5e-4.

Method details: [VALIDATION.md](https://github.com/ar1so/pyradix/blob/main/docs/VALIDATION.md).
