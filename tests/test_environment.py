import sys
from pathlib import Path


def test_runtime_and_source_files() -> None:
    """Catches execution under the wrong Python or missing immutable inputs."""
    assert sys.version_info[:2] == (3, 12)
    assert Path("/Users/um-yunsang/Downloads/open.zip").is_file()
    assert Path("/Users/um-yunsang/Downloads/baseline.ipynb").is_file()


def test_package_is_importable() -> None:
    """Catches an unconfigured src-layout package."""
    import baram

    assert baram.__version__ == "0.1.0"
