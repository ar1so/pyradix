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

```bash
docker build -t pyradix https://github.com/ar1so/pyradix.git
docker run --rm -v "$PWD:/data" pyradix extract image.nii.gz mask.nii.gz --out features.csv
```

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
