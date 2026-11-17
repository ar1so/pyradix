# Benchmarks

These scripts call the repository's own code. They do not contain typed-in results. Figures are drawn from the JSON files in `results/`.

## What is measured

- IBSI-1 digital phantom and PAT1 configurations C, D, and E against `tests/references/`
- IBSI-2 filter tests under `tests/ibsi/`, plus the Laplacian-of-Gaussian impulse error
- Live original features against PyRadiomics 3.1.0 on the digital phantom, brain1, and PAT1
- Extract time and child-process peak RSS against ROI size and a thread pin of 1, 2, or 4

No random generator is used. Scaling masks keep the voxels nearest the centroid.

## How to repeat

Public files:

```bash
python3 scripts/fetch_validation_data.py
```

pyradix, release build, from the repository root. Linux needs OpenBLAS and a gfortran library path if the linker does not search the GCC directory:

```bash
python3 -m venv .venv
.venv/bin/pip install "maturin>=1.14,<1.16" numpy nibabel pynrrd pyyaml pytest matplotlib
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/13${LIBRARY_PATH:+:$LIBRARY_PATH}"
.venv/bin/maturin develop --release
```

PyRadiomics 3.1.0 publishes manylinux wheels through CPython 3.9 only. The run in this folder used Python 3.9.25:

```bash
python3.9 -m venv .venv-pyrad
.venv-pyrad/bin/pip install "numpy==1.26.4" "SimpleITK==2.3.1" pykwalify pywavelets six
.venv-pyrad/bin/pip install --no-deps \
  "https://files.pythonhosted.org/packages/93/37/a1a66918ee9f8f1e8339a9a203317d20e478ecb11f5a350e31ab7a318a04/pyradiomics-3.1.0-cp39-cp39-manylinux1_x86_64.whl"
```

Then:

```bash
.venv/bin/python benchmarks/run_all.py
```

`PYRAD_PYTHON` can point at another interpreter. The default is `.venv-pyrad/bin/python`.

## Outputs

- `results/environment.json` — machine, versions, and the statement that no seed was used. The hostname is not recorded
- `results/ibsi.json`
- `results/pyradiomics_agreement.json` and `results/pyradiomics_features.csv`
- `results/runtime.json`
- `results/summary.json` — the numbers the manuscript quotes
- `../figures/fig1` … `fig6` as PDF, SVG, and 300 dpi PNG
- `../softwarex/generated_results.tex`

If a stage cannot run, the JSON records the error string from the child process. Do not replace that string with an estimated time.
