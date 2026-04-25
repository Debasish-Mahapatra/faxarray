from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.native_fa import NativeFAResource, write_fa


SAMPLE = Path("/tmp/EPyGrAM/tests/data/geometries/regLL_small.fa")


pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="EPyGRAM sample FA files not available")


def test_native_backend_lists_fields_and_reads_geometry():
    resource = NativeFAResource(str(SAMPLE))

    assert resource.list_h2d_fields() == ["SURFIND.TERREMER", "SURFGEOPOTENTIEL"]
    assert resource.geometry.name == "regular_lonlat"
    assert resource.geometry.shape == (601, 801)
    assert np.isclose(resource.geometry.lons.min(), -8.0)
    assert np.isclose(resource.geometry.lons.max(), 12.0)


def test_native_backend_reads_uncompressed_gridpoint_field():
    resource = NativeFAResource(str(SAMPLE))
    data = resource.readfield("SURFGEOPOTENTIEL")

    assert data.shape == (601, 801)
    assert np.isfinite(data).all()
    assert data.dtype == np.float64


def test_native_backend_reads_legacy_packed_gridpoint_field():
    resource = NativeFAResource(str(SAMPLE))
    data = resource.readfield("SURFIND.TERREMER")

    assert data.shape == (601, 801)
    assert np.isfinite(data).all()
    assert data.dtype == np.float64
    assert np.isclose(data.min(), 0.0)
    assert np.isclose(data.max(), 1.0)


def test_template_write_replaces_raw_field(tmp_path):
    xr = pytest.importorskip("xarray")
    resource = NativeFAResource(str(SAMPLE))
    source = resource.readfield("SURFGEOPOTENTIEL")
    replacement = source + 1.0
    ds = xr.Dataset({"SURFGEOPOTENTIEL": (["y", "x"], replacement)})

    output = tmp_path / "written.fa"
    write_fa(ds, str(output), template=str(SAMPLE), overwrite=False)

    written = NativeFAResource(str(output)).readfield("SURFGEOPOTENTIEL")
    np.testing.assert_allclose(written, replacement)


def test_template_write_replaces_legacy_packed_field(tmp_path):
    xr = pytest.importorskip("xarray")
    y = np.linspace(0.0, 1.0, 601, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, 801, dtype=np.float64)[None, :]
    replacement = x + y
    ds = xr.Dataset({"SURFIND.TERREMER": (["y", "x"], replacement)})

    output = tmp_path / "written-packed.fa"
    write_fa(ds, str(output), template=str(SAMPLE), overwrite=False)

    written = NativeFAResource(str(output)).readfield("SURFIND.TERREMER")
    np.testing.assert_allclose(written, replacement, atol=2.0e-7)
