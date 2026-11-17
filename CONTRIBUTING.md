# Contributing to pyradix

Thank you for helping. pyradix is research software that people will cite, so every change that can move a feature value has to come with evidence that the new value is right. This guide says what evidence is enough.

- [Ways to contribute](#ways-to-contribute)
- [Set up and run the tests](#set-up-and-run-the-tests)
- [Standards and modes](#standards-and-modes)
- [Contributing a mode](#contributing-a-mode)
- [Validation rules](#validation-rules)
- [Contributing a feature, filter or standard](#contributing-a-feature-filter-or-standard)
- [Pull request checklist](#pull-request-checklist)
- [Release](#release)

## Ways to contribute

| You want to… | Start with |
|---|---|
| Report a wrong value, a crash or an install problem | A **Bug report** issue with the image/mask (or a way to make them), the settings and `pyradix --version` |
| Fix docs or examples | A pull request |
| Add a **mode** (a named, validated way to extract features) | A **New mode proposal** issue, then the steps below |
| Add a feature, a filter or a **standard** | An issue first; these change the engine |

## Set up and run the tests

You need Rust 1.85+ and Python 3.10–3.13. Linux also needs `pkg-config`, OpenBLAS (`libopenblas-dev`) and `patchelf`; macOS uses Accelerate; Windows needs only the Visual Studio C++ Build Tools.

Test the installed wheel, not the source tree. That is what users run.

```bash
python -m pip install "maturin>=1.14,<1.16"
python -m maturin build --release -o dist
python -m pip install --force-reinstall dist/*.whl
python -m pip install "pytest>=8" "simpleitk>=2.3"
python scripts/fetch_validation_data.py      # public reference data, checked against hashes
python -m pytest -q
```

The same commands work in PowerShell on Windows. CI runs exactly this on Linux, macOS and Windows for every push and pull request.

Rust unit tests: `cargo test -p radix-core`. Style: `ruff check python tests`, `cargo fmt --all`, `cargo clippy`.

Layout: `crates/radix-core` is the engine, `crates/radix-py` the Python binding, `python/pyradix` the package, `tests/references` the reference values, `examples` the parameter files. The version in `pyproject.toml`, `Cargo.toml` and `python/pyradix/_version.py` must match.

## Standards and modes

Two words have one meaning each in this project:

- A **standard** defines the math: which formula a feature uses, how intensities are discretized, how a filter is built. pyradix has two: **IBSI** (IBSI 1 features and IBSI 2 filters) and **PyRadiomics 3.1.0**. Adding a standard means new engine code and is rare.
- A **mode** is what a user picks. It is one standard plus everything needed to get reproducible numbers from it: fixed settings, the features it computes, which images it accepts, and the evidence that it is right. `ibsi` and `pyradiomics` are the first two modes. Future modes might target a modality (`us-2d`), a protocol (`lung-screening`) or a speed budget (`ct-fast`).

**A new mode almost never needs new feature code.** It is a choice of settings and features on top of a standard that is already validated, plus its own validation.

> The mode catalogue (`pyradix.modes()`, `RadixExtractor(mode="us-2d")`, the mode version and hash in every result) is planned for **0.2**. Until it lands, a mode is contributed as a parameter file and a validation pack, as described below, and the maintainers register it when the catalogue ships. Nothing in your pack has to change then.

## Contributing a mode

### 1. Propose it

Open a **New mode proposal** issue. Say:

- **Purpose:** who uses it and for what.
- **Base standard:** `ibsi` or `pyradiomics`.
- **Input contract:** modality, 2D or 3D, intensity units (HU, raw, normalized), allowed spacing.
- **Settings and features:** what is fixed and what is computed, and why.
- **Reference source:** where the expected values will come from (see [Validation rules](#validation-rules)).
- **Data:** which public cases you will use and their license.

Wait for agreement on the name and scope before writing the pack. Names are lowercase with hyphens, modality first when there is one: `us-2d`, `ct-fast`.

### 2. Write the validation pack

One folder per mode:

```
validation/<mode>/
  mode.yaml       # the mode itself: name, version, base standard, input contract, features
  params.yaml     # the settings as a pyradix parameter file (what RadixExtractor loads)
  cases.csv       # case_id, image, mask, sha256 of each file, source, license
  expected.csv    # case_id, feature, value, tol (blank tol = the default rule)
  SOURCE.md       # where every expected value came from, exactly how it was produced
```

- **`mode.yaml`** follows the example below. Spacing uses named axes (`x`, `y`, `z`), never an unnamed list.
- **`cases.csv`** points at data; it does not contain it. Images are downloaded by `scripts/fetch_validation_data.py` and checked against the hashes. Add your files there and to `docs/datasets/MANIFEST.md`.
- **`expected.csv`** uses the same columns as `tests/references/ibsi1_pat1_config_cde.csv`.
- **`SOURCE.md`** must let someone else regenerate `expected.csv` without asking you: tool name and exact version, the exact command or script, the date, the machine.

```yaml
# validation/us-2d/mode.yaml (example)
name: us-2d
version: "1.0"
base: pyradiomics
summary: 2D B-mode ultrasound frames; drops 3D shape.
input:
  modality: US
  dimensions: 2
  units: raw
settings_file: params.yaml
features:
  firstorder: all
  shape2D: all
  glcm: all
```

### 3. Write the test

Add `tests/modes/test_<mode>.py`. It must:

1. skip cleanly, with a message saying how to fetch the data, when the data is not on disk;
2. load `params.yaml`, extract every case in `cases.csv`, and compare **every** row of `expected.csv` using the tolerance rule below;
3. check the input contract: an input the mode does not accept (wrong dimensionality, wrong units) raises a clear `RadixError` and is not silently processed;
4. run in under about a minute on a laptop, so CI stays fast.

`tests/ibsi/test_pat1_cde.py` is a good starting point: it already reads a reference CSV and applies per-row tolerances.

### 4. Open the pull request

CI must be green on Linux, macOS and Windows. A maintainer reruns `SOURCE.md` independently for at least one case before merging. A mode that passes is listed as **verified**.

## Validation rules

**Where expected values may come from**, in order of preference:

1. **Published reference values**, for example the IBSI tables.
2. **An independent, pinned implementation**, for example PyRadiomics 3.1.0 run in its own environment (pyradix never imports it), with the exact version in `SOURCE.md`.
3. **Hand calculation** on a small synthetic phantom, with the working in `SOURCE.md`.

**Never from pyradix itself.** Comparing pyradix with its own earlier output proves only that nothing changed, not that anything is right. Such regression snapshots are welcome as extra tests, but they do not validate a mode.

**Tolerance:** integer-valued features must match exactly. Other features must agree within `1e-4`, absolute or relative, whichever is looser. A wider tolerance needs a reason in `SOURCE.md` (for example a reference published to three significant digits) and the reviewer's agreement.

**Data:** public, with a license that allows redistribution or a stable download URL; no patient-identifiable information; small (a few MB per case). Synthetic phantoms are welcome for edge cases.

**Changing a mode later:** any change that can move an output value is a new major version (`us-2d@2.0`), with the reason in `SOURCE.md`. Expected values are never edited just to make a test pass.

## Contributing a feature, filter or standard

These change the engine, so open an issue first.

- **Feature or filter:** implement it in `crates/radix-core`, register it in `registry.rs` with the standards it supports, add a Rust unit test, and add reference values and a Python test exactly as for a mode. If it differs between the standards, add the difference to `docs/DEVIATIONS.md`.
- **New standard:** needs reference values published outside this project, a written comparison with the existing standards, and agreement from the maintainers before work starts.

## Pull request checklist

- [ ] `python -m pytest -q` passes against an installed wheel, and `cargo test -p radix-core` passes.
- [ ] `ruff check python tests` and `cargo fmt --all -- --check` are clean for the files you touched.
- [ ] Any change that can move a feature value has validation evidence from an independent source.
- [ ] New behaviour has a test; a fixed bug has a test that failed before the fix.
- [ ] User-visible changes are in `CHANGELOG.md` and, where relevant, the README or `docsite/`.

## Docs

```bash
python -m pip install "mkdocs>=1.6" "mkdocs-material>=9.5" "mkdocstrings[python]>=0.26"
python -m mkdocs build --strict
```

The site source is `docsite/`; it builds to `site-build/`.

## Release

Pushing a tag `v*` runs `.github/workflows/release.yml`: it builds the wheels and the sdist on Linux, macOS and Windows and publishes to PyPI with Trusted Publishing (GitHub environment `pypi`, no stored password). A manual run publishes nothing unless asked. Never upload a release by hand.
