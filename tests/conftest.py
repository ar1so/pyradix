from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    return repo_root / "examples"


@pytest.fixture(scope="session")
def d1_image(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz"


@pytest.fixture(scope="session")
def d1_mask(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz"


@pytest.fixture(scope="session")
def d2_image(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz"


@pytest.fixture(scope="session")
def d2_mask(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz"


@pytest.fixture(scope="session")
def d3_impulse(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi2" / "digital_phantom" / "impulse.nii.gz"


@pytest.fixture(scope="session")
def d3_checkerboard(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi2" / "digital_phantom" / "checkerboard.nii.gz"


@pytest.fixture(scope="session")
def d3_sphere(repo_root: Path) -> Path:
    return repo_root / "data" / "ibsi2" / "digital_phantom" / "sphere.nii.gz"


@pytest.fixture(scope="session")
def d4_image(repo_root: Path) -> Path:
    return repo_root / "data" / "pyradiomics" / "brain1" / "brain1_image.nrrd"


@pytest.fixture(scope="session")
def d4_mask(repo_root: Path) -> Path:
    return repo_root / "data" / "pyradiomics" / "brain1" / "brain1_label.nrrd"
