# D5 — Aerts 2014 Lung1 (NSCLC-Radiomics)

The cohort for documentation and a later extractor paper. Not a v1 phantom gate.

## Study

Aerts, H. J. W. L. et al. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. *Nat Commun* **5**, 4006 (2014). https://doi.org/10.1038/ncomms5006

Lung1 (n=422, MAASTRO) was the **training** set that built the four-feature radiomic signature. Independent validation in that paper used Lung2, H&N1, H&N2. We replay **Lung1** because TCIA ships CT + GTV + clinical for every subject.

## What we keep

| Piece | File | Notes |
|---|---|---|
| Pretreatment CT | `data/lung1/<PatientID>/image.nii.gz` | Converted from DICOM CT |
| Tumor mask | `data/lung1/<PatientID>/mask.nii.gz` | DICOM **SEG** GTV-1 (not v2 RTSTRUCT) |
| Clinical table | `data/lung1/Lung1.clinical.csv` | Survival days from start of treatment; `deadstatus.event` 1=death, 0=censored |

Patient IDs match the DICOM PatientName / CSV (`LUNG1-xxx`).

## Do not use

- **Version 2 RTSTRUCT** — TCIA removed it; ROIs were not vertically aligned with the CT.
- **LUNG1-128** — post-operative, no GTV-1 (TCIA note).
- **LUNG1-014, LUNG1-021, LUNG1-085** — missing CT slices (not through the GTV). Flag, do not drop unless a geometry check fails.

## License and citation (required)

CC BY-NC 3.0. Non-commercial. Cite **both**:

1. Aerts et al., *Nat Commun* 2014. https://doi.org/10.1038/ncomms5006
2. Aerts et al. Data From NSCLC-Radiomics (version 4). TCIA. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

Collection page: https://www.cancerimagingarchive.net/collection/nsclc-radiomics/

## Why this study (not the other public packs)

| Candidate | n | Image+mask+clinical | Why not first |
|---|---|---|---|
| **Lung1 / NSCLC-Radiomics** | 422 | yes (CT, SEG GTV-1, CSV) | **picked** — the paper everyone cites |
| Head-Neck-Radiomics-HN1 | 137 | yes (same paper’s H&N1) | secondary validation later; images listed NIH-controlled on one TCIA page |
| Soft-tissue sarcoma / IBSI STS | 51 | yes (Vallières 2015 *Phys Med Biol*) | IBSI filter-repro pack, not the clinical radiomics story |
| PyRadiomics `brain1` | 1 | no outcomes | already D4 / G5 |

## Fetch

```bash
# clinical table only (~23 kB)
.venv/bin/python scripts/fetch_lung1.py --clinical

# CT + SEG → NIfTI (skips RTSTRUCT). Needs ~disk for one patient at a time.
.venv/bin/python scripts/fetch_lung1.py
.venv/bin/python scripts/fetch_lung1.py --limit 10   # smoke
```

TCIA’s full DICOM dump is 35.78 GB (CT+SEG+RTSTRUCT). This script downloads **CT + SEG only** and deletes DICOM after NIfTI conversion so a ~33 GB free disk can hold the cohort.

## Use with Radix / PyRadiomics

Same settings as the live compare in [VALIDATION.md](../VALIDATION.md) (`binWidth=25`, Original, no resample) unless the paper replay needs Aerts’ original preprocess (document that separately). Pair list: `data/lung1/pairs.csv`.

Research-method extractor compare (Radix vs **installed** PyRadiomics 3.1.0, not Radix compat as the PyRadiomics arm):

```bash
.venv/bin/python scripts/compare_lung1.py
pytest tests/lung1/test_lung1_cohort.py
```

Report: `scripts/compare_lung1.py` writes it to `data/lung1/LUNG1_STUDY.md`. Gate **G9** (open — 10 Radix hangs). Default extractors: `scripts/extract_radix_default.py`, `scripts/extract_pyradiomics_default.py`.
