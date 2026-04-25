"""Tests for from-scratch FA file creation."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.fa_writer import (
    FAFieldData,
    FAGlobalGaussGeometry,
    FARegularLonLatGeometry,
    FAValidityInput,
    FAVerticalInput,
    create_fa_file,
)
from faxarray.backends.lfi_writer import LFIWriter
from faxarray.backends.native_fa import NativeFAResource
from faxarray.backends.native_lfi import LFIFile


def test_lfi_writer_round_trip_single_index(tmp_path):
    path = tmp_path / "single.lfi"
    writer = LFIWriter(str(path), page_size_bytes=4096)
    writer.add_article("FOO", b"\x00" * 24)
    writer.add_article("BAR", b"\xff" * 80)
    writer.write()

    lfi = LFIFile(str(path))
    assert [a.name for a in lfi.articles] == ["FOO", "BAR"]
    assert lfi.read_article_bytes("FOO") == b"\x00" * 24
    assert lfi.read_article_bytes("BAR") == b"\xff" * 80


def test_lfi_writer_handles_extra_index_sections(tmp_path):
    """A small page size forces multiple index sections."""

    path = tmp_path / "multi.lfi"
    writer = LFIWriter(str(path), page_size_bytes=512)  # 32 entries / index
    n_articles = 80  # needs 3 index sections
    payload = (b"\x01" * 8)
    for i in range(n_articles):
        writer.add_article(f"ART{i:03d}", payload)
    writer.write()

    lfi = LFIFile(str(path))
    assert len(lfi.articles) == n_articles
    # Check non-zero ioffi entries (extra-index pointers).
    nonzero = [w for w in lfi.header_words[22:] if w != 0]
    assert len(nonzero) == 2  # 3 index sections => 2 extra
    # Confirm round-trip on first/middle/last article.
    for name in ("ART000", "ART040", "ART079"):
        assert lfi.read_article_bytes(name) == payload


def test_create_regular_lonlat_round_trip(tmp_path):
    geometry = FARegularLonLatGeometry(nx=20, ny=15, lon0=2.0, lat0=45.5, dx=0.1, dy=0.1)
    rng = np.random.default_rng(0)
    surfind = rng.random((15, 20))
    surftemp = rng.random((15, 20)) * 30 + 250

    output = tmp_path / "regll.fa"
    create_fa_file(
        str(output),
        geometry=geometry,
        fields=[
            FAFieldData("SURFIND.TERREMER", surfind),
            FAFieldData("SURFTEMPERATURE", surftemp),
        ],
        validity=FAValidityInput(
            base_time=datetime(2024, 6, 1, 12), lead_time=timedelta(hours=6)
        ),
    )

    resource = NativeFAResource(str(output))
    assert resource.geometry.name == "regular_lonlat"
    assert resource.geometry.shape == (15, 20)
    assert set(resource.list_h2d_fields()) == {"SURFIND.TERREMER", "SURFTEMPERATURE"}
    np.testing.assert_allclose(resource.readfield("SURFIND.TERREMER"), surfind)
    np.testing.assert_allclose(resource.readfield("SURFTEMPERATURE"), surftemp)
    # Validity decoded back from DATX seconds.
    validity = resource.validity
    assert validity.base_time == np.datetime64(datetime(2024, 6, 1, 12))
    assert validity.lead_time == np.timedelta64(6 * 3600, "s")


def test_create_global_reduced_gauss_round_trip(tmp_path):
    sin_lat_nh = np.array([0.99, 0.95, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0])
    lon_number_nh = [16, 24, 28, 32, 32, 36, 40, 40]
    geometry = FAGlobalGaussGeometry(
        sin_lat_nh=sin_lat_nh,
        lon_number_by_lat_nh=lon_number_nh,
        truncation=10,
    )
    total_points = 2 * sum(lon_number_nh)
    rng = np.random.default_rng(1)
    field = rng.random(total_points) * 30 + 250

    output = tmp_path / "gauss.fa"
    create_fa_file(
        str(output),
        geometry=geometry,
        fields=[FAFieldData("SURFTEMPERATURE", field)],
    )

    resource = NativeFAResource(str(output))
    assert resource.geometry.name == "reduced_gauss"
    assert resource.geometry.projection["lat_number"] == 16
    assert resource.geometry.projection["total_points"] == total_points
    assert resource.list_h2d_fields() == ["SURFTEMPERATURE"]
    np.testing.assert_allclose(resource.readfield("SURFTEMPERATURE").ravel(), field)


def test_create_rotated_stretched_gauss_round_trip(tmp_path):
    sin_lat_nh = np.array([0.99, 0.85, 0.5, 0.0])
    geometry = FAGlobalGaussGeometry(
        sin_lat_nh=sin_lat_nh,
        lon_number_by_lat_nh=[16, 16, 16, 16],
        truncation=3,
        pole_sin_lat=0.725,
        pole_cos_lon=0.99899,
        pole_sin_lon=0.04498,
        stretching_factor=2.4,
    )
    rng = np.random.default_rng(2)
    field = rng.random(2 * 16 * 4)

    output = tmp_path / "rotated.fa"
    create_fa_file(str(output), geometry=geometry, fields=[FAFieldData("FOO", field)])

    resource = NativeFAResource(str(output))
    assert resource.geometry.name == "rotated_reduced_gauss"
    proj = resource.geometry.projection
    assert proj["pole_sin_lat"] == pytest.approx(0.725)
    assert proj["stretching_factor"] == pytest.approx(2.4)
    np.testing.assert_allclose(resource.readfield("FOO").ravel(), field)


def test_create_fa_file_rejects_field_with_wrong_shape(tmp_path):
    geometry = FARegularLonLatGeometry(nx=10, ny=10, lon0=0.0, lat0=0.0, dx=0.1, dy=0.1)
    bad = np.zeros((9, 9))
    with pytest.raises(ValueError, match="expected"):
        create_fa_file(
            str(tmp_path / "bad.fa"),
            geometry=geometry,
            fields=[FAFieldData("SURFIND.TERREMER", bad)],
        )


def test_create_fa_from_dataset_high_level(tmp_path):
    xr = pytest.importorskip("xarray")
    import faxarray as fx

    nx, ny = 12, 10
    lon = np.linspace(-1.0, 1.0, nx)
    lat = np.linspace(40.0, 41.0, ny)
    lons_2d, lats_2d = np.meshgrid(lon, lat)
    rng = np.random.default_rng(3)
    data = rng.random((ny, nx))

    ds = xr.Dataset(
        {
            "SURFTEMPERATURE": (("y", "x"), data),
        },
        coords={"lat": (("y", "x"), lats_2d), "lon": (("y", "x"), lons_2d)},
    )

    output = tmp_path / "from_xr.fa"
    fx.create_fa_from_dataset(ds, str(output))

    resource = NativeFAResource(str(output))
    assert resource.geometry.shape == (ny, nx)
    np.testing.assert_allclose(resource.readfield("SURFTEMPERATURE"), data)
