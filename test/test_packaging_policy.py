from pathlib import Path


def test_package_tree_does_not_include_model_sources():
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "faxarray" / "_vendor" / "ifsaux").exists()


def test_pyproject_does_not_package_model_sources():
    repo = Path(__file__).resolve().parents[1]
    pyproject = (repo / "pyproject.toml").read_text()
    forbidden = ["_vendor", "ifsaux", "grib_mf/*", "tool.setuptools.package-data"]
    for token in forbidden:
        assert token not in pyproject


def test_legacy_codec_does_not_require_external_model_sources():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "faxarray" / "backends" / "grib_mf_codec.py").read_text()
    forbidden = [
        "FAXARRAY_IFSAUX_ROOT",
        "FAXARRAY_ROOTPACK_TARBALL",
        "FAXARRAY_GRIB_MF_LIBRARY",
        "ctypes",
        "gfortran",
        "tarfile",
    ]
    for token in forbidden:
        assert token not in source
