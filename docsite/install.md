---
description: "Install pyradix (pyradix-radiomics on PyPI) with pip, uv or Docker. Ready-made wheels for Python 3.10–3.13 on Linux, macOS and Windows."
---

# Install

Ready-made packages exist for Python 3.10–3.13 on Linux, macOS and Windows; no Rust or compiler needed. On PyPI the package is **`pyradix-radiomics`**; in code you `import pyradix`.

## pip

```bash
pip install pyradix-radiomics
```

## uv

```bash
uv add pyradix-radiomics                     # in a project
uvx --from pyradix-radiomics pyradix --help  # run the command line without installing
```

## Docker

Use Docker if you do not want to install Python, or need the same environment on every machine. You only need
[Docker](https://docs.docker.com/get-docker/); no Python or Rust.

**1. Build the image** once. It compiles pyradix from the GitHub source, so the first build takes several minutes:

```bash
docker build -t pyradix https://github.com/ar1so/pyradix.git
```

**2. Check it works:**

```bash
docker run --rm pyradix --version
```

**3. Extract features.** Run this from the folder that holds your image and mask. `-v "$PWD:/data"` makes that
folder visible inside the container, and `features.csv` is written back into it:

```bash
docker run --rm -v "$PWD:/data" pyradix extract image.nii.gz mask.nii.gz --out features.csv
```

The container runs the `pyradix` command line, so every option of `pyradix extract --help` works the same way. On
Windows PowerShell, use `${PWD}` instead of `$PWD`. On Linux, if you get "Permission denied" writing the output,
add `--user "$(id -u):$(id -g)"` after `docker run`.

## From source

Only needed if you want to change pyradix itself. You need **Rust 1.85+** ([rustup.rs](https://rustup.rs)) and Python 3.10–3.13.

## macOS

```bash
git clone https://github.com/ar1so/pyradix && cd pyradix
python -m pip install "maturin>=1.14,<1.16"
python -m maturin build --release -o dist
python -m pip install dist/*.whl
```

To build one wheel for both Apple silicon and Intel, add `--target universal2-apple-darwin` to the build command.

## Linux

Install pkg-config and OpenBLAS first, then run the macOS commands:

```bash
sudo apt-get install pkg-config libopenblas-dev patchelf   # Debian / Ubuntu
```

## Windows

Install the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++". No OpenBLAS is needed: on Windows, the one eigenvalue step (GLCM MCC) uses the pure-Rust `faer` library. In PowerShell:

```powershell
git clone https://github.com/ar1so/pyradix; cd pyradix
python -m pip install "maturin>=1.14,<1.16"
python -m maturin build --release -o dist
python -m pip install (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
```

## Check the install

```bash
pyradix --version
```

Runtime dependencies (installed by pip): numpy, nibabel, pynrrd, PyYAML.
