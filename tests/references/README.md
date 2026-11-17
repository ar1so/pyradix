# Reference tables (text, committed)

Golden CSVs live here. Images stay in `data/` (gitignored).

- `ibsi1_digital_phantom.csv` — IBSI-1 published digital-phantom values (G1/G2)
- `ibsi1_pat1_config_cde.csv` — PAT1 configs C/D/E published values (G3)
- `pyradiomics_3.1.0_brain1.csv` — frozen compat baseline (task M3)
- `radix_vs_pyradiomics_metrics.json` — live batch times + pass counts vs installed 3.1.0 (M5/G8); regenerate with `scripts/compare_pyradiomics.py` (release wheel). Protocol: `docs/COMPARE.md`.
- `radix_vs_pyradiomics_features.csv` — per-feature abs/rel errors from that batch
- `lung1/metrics.json` — D5 Lung1 research compare vs **installed** PyRadiomics 3.1.0 (M6/G9). Regenerate with `scripts/compare_lung1.py`. Protocol: `docs/LUNG1_STUDY.md`. Checkpoints and the long `features.csv` are local artifacts.

IBSI source: [ibsi-doc `03_Image_features.rst`](https://github.com/theibsi/ibsi-doc/blob/master/docs/03_Image_features.rst). Texture rows use 3D averaged (GLCM/GLRLM) or 3D (GLSZM/NGTDM/NGLDM) aggregation.
