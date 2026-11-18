from importlib.metadata import version

from pyradix import __version__, engine_info, parse_mode


def test_import_and_rust_engine():
    assert __version__ == "0.1.1"
    assert version("pyradix-radiomics") == __version__  # PyPI name; the import is pyradix
    info = engine_info()
    assert info.startswith("radix-core")
    assert info.split()[-1] == __version__
    assert parse_mode("IBSI") == "ibsi"
    assert parse_mode("pyradiomics") == "pyradiomics"
