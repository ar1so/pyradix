"""Path, line-ending, and file-lock behavior that breaks on Windows."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
import pytest

from pyradix import Params, RadixExtractor, image_io_path, load_volume
from pyradix.extractor import _setting_text
from pyradix.paths import native_path


def _nifti(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    nib.save(nib.Nifti1Image(np.transpose(values, (2, 1, 0)), affine), os.fspath(path))


def test_nifti_and_numpy_roundtrip_releases_the_file(tmp_path: Path):
    folder = tmp_path / "patient 01" / "café"
    image = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    nifti_path = folder / "image.nii.gz"
    _nifti(nifti_path, image)
    loaded = load_volume(nifti_path)
    assert loaded.array.shape == image.shape
    assert loaded.integer_pixels is False
    assert np.allclose(loaded.array, image)
    # A remaining memmap makes this unlink raise PermissionError on Windows.
    nifti_path.unlink()
    assert not nifti_path.exists()

    npy_path = folder / "values.npy"
    np.save(npy_path, image)
    restored = np.load(npy_path)
    assert np.array_equal(restored, image)
    npy_path.unlink()


def test_integer_nifti_flag_and_spaces(tmp_path: Path):
    path = tmp_path / "roi data" / "mask.nii.gz"
    values = np.zeros((2, 2, 2), dtype=np.int16)
    values[0, 0, 0] = 1
    _nifti(path, values.astype(np.float64))
    # Rewrite as int16 so the integer-pixel flag is meaningful.
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), affine), os.fspath(path))
    loaded = load_volume(path)
    assert loaded.integer_pixels
    path.unlink()


def test_nrrd_roundtrip_releases_the_file(tmp_path: Path):
    path = tmp_path / "scan data" / "brain.nrrd"
    path.parent.mkdir(parents=True)
    data = np.arange(24, dtype=np.int16).reshape(4, 3, 2)
    nrrd.write(os.fspath(path), data)
    loaded = load_volume(path)
    assert loaded.array.ndim == 3
    assert loaded.integer_pixels
    path.unlink()
    assert not path.exists()


def test_crlf_and_bom_yaml(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_bytes(b"\xef\xbb\xbfmode: ibsi\r\nbin_width: 25\r\n")
    params = Params.from_yaml(path)
    assert params.mode == "ibsi"
    assert params.bin_width == 25.0
    extracted = RadixExtractor(mode="ibsi", params=params)
    assert extracted.mode == "ibsi"


def test_clear_yaml_uses_lf(tmp_path: Path):
    from pyradix import dump_clear_yaml
    from pyradix.extractor import ExtractResult

    params = Params(mode="ibsi", bin_width=25.0)
    result = ExtractResult(values={"original_firstorder_Mean": 1.0}, provenance=params.provenance())
    path = tmp_path / "clear.yaml"
    dump_clear_yaml(result, path)
    raw = path.read_bytes()
    assert b"\r" not in raw


def test_setting_text_is_locale_independent():
    assert _setting_text(0.1) == "0.1"
    assert _setting_text(1.0) == "1.0"
    assert _setting_text(np.float64(1.5)) == "1.5"
    assert "," not in _setting_text(1234.5)


def test_image_io_path_is_absolute(tmp_path: Path):
    path = tmp_path / "a b" / "scan.nii.gz"
    path.parent.mkdir()
    path.write_bytes(b"")
    got = image_io_path(path)
    assert Path(native_path(path)).is_absolute()
    if os.name == "nt":
        assert "\\" not in got or got.startswith("\\\\?\\")
        assert "a b" in got
    else:
        assert got.endswith("scan.nii.gz")
        assert "a b" in got


def test_simpleitk_roundtrip_on_spaced_path(tmp_path: Path):
    sitk = pytest.importorskip("SimpleITK")
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 1.0, 2.0))
    path = tmp_path / "series 1" / "scan.nii.gz"
    path.parent.mkdir()
    sitk.WriteImage(image, image_io_path(path), True)
    back = sitk.ReadImage(image_io_path(path))
    assert tuple(int(v) for v in back.GetSize()) == (4, 3, 2)
    loaded = load_volume(path)
    assert loaded.array.shape == (2, 3, 4)
    path.unlink()
