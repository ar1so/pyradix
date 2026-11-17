# Validation

Nothing is “done” without a gate here.

## Datasets (small, public)

Fetched by `python3 scripts/fetch_validation_data.py`. Binaries live in `data/` (gitignored). Hashes: [datasets/MANIFEST.md](datasets/MANIFEST.md).

| ID | What | Local path | Used for |
|---|---|---|---|
| D1 | IBSI-1 digital phantom | `data/ibsi1/digital_phantom/` | Config A/B features |
| D2 | IBSI-1 lung CT PAT1 | `data/ibsi1/lung_ct_pat1/` | Config C/D/E (real CT, resample) |
| D3 | IBSI-2 digital filter phantoms | `data/ibsi2/digital_phantom/` | Filter response images |
| D4 | PyRadiomics `brain1` | `data/pyradiomics/brain1/` | Compat mode vs frozen 3.1.0 |
| D5 | Aerts 2014 Lung1 (NSCLC-Radiomics) | `data/lung1/` | Paper cohort: CT + GTV-1 + clinical ([LUNG1.md](datasets/LUNG1.md)) |

D1–D4 stay the small v1 gates. D5 is the real study (422 NSCLC, CC BY-NC 3.0). Fetch: `python3 scripts/fetch_lung1.py`. The 51-patient IBSI STS pack is still not a v1 gate.

## Gates

| Gate | Mode | Data | Pass if |
|---|---|---|---|
| G1 | `ibsi` | D1 | First-order + shape match IBSI published values |
| G2 | `ibsi` | D1 | Texture families match IBSI published values |
| G3 | `ibsi` | D2 | Configs C, D, E match IBSI tables |
| G4 | `ibsi` | D3 | Filter images match IBSI-2 references (where published) |
| G5 | `pyradiomics` | D4 | Features within `1e-4` of frozen PyRadiomics 3.1.0 |
| G6 | either | D1 | Voxel maps finish; `voxel_batch` memory cap holds |
| G7 | `ibsi` | D1 | Perturbation run writes an ICC table (no crash) |
| G8 | `pyradiomics` | D1, D2, D4 × batch | Live values within `1e-4` of installed 3.1.0 on the job matrix in `scripts/compare_pyradiomics.py`; extract times recorded. Allowed: `d2_pat1` `MeshVolume`. Recorded gap: D4 2 mm resample. |
| G9 | Radix vs installed `radiomics` 3.1.0 | D5 Lung1 (421 GTV) | Research-method cohort: feature agreement + extract time vs the **real PyRadiomics package** (not Radix compat as the baseline). Allowed: `MeshVolume`. |

IBSI integer features must match exactly. Floating features use the IBSI reported tolerance when one exists, else relative `1e-5` or absolute `1e-4`, whichever is looser.

## Golden files

Committed under `tests/references/` (text only):

- IBSI published feature tables (CSV)
- Frozen PyRadiomics 3.1.0 `brain1` vector (CSV), generated once and never edited by hand

## How to run (once the engine exists)

```bash
python3 scripts/fetch_validation_data.py
pytest tests/ibsi tests/pyradiomics_parity
# live batch vs installed PyRadiomics 3.1.0
.venv/bin/python scripts/compare_pyradiomics.py
# Lung1 research cohort vs installed PyRadiomics 3.1.0 (report written to data/lung1/)
.venv/bin/python scripts/compare_lung1.py
pytest tests/lung1/test_lung1_cohort.py
```
