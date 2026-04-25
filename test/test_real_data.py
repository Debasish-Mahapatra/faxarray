from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.native_fa import NativeFAResource


REAL_SAMPLE = Path("/Users/dev/PROJECTS/test-data/pfABOFABOF+0012")


pytestmark = pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="local real FA test file not available")


def test_real_file_lists_geometry_and_encodings():
    resource = NativeFAResource(str(REAL_SAMPLE))

    assert resource.geometry.name == "mercator"
    assert resource.geometry.shape == (480, 480)
    assert len(resource.list_h2d_fields()) == 3383
    assert resource.fieldencoding("SURFTEMPERATURE")["spectral"] is False
    assert resource.fieldencoding("S001TEMPERATURE")["spectral"] is True


def test_real_file_reads_packed_and_raw_gridpoint_fields():
    resource = NativeFAResource(str(REAL_SAMPLE))

    temperature = resource.readfield("SURFTEMPERATURE")
    geopotential = resource.readfield("SURFGEOPOTEN")

    assert temperature.shape == (480, 480)
    assert geopotential.shape == (480, 480)
    assert np.isfinite(temperature).all()
    assert np.isfinite(geopotential).all()
    assert 250.0 < float(temperature.min()) < 330.0
    assert float(geopotential.max()) > float(geopotential.min())


def test_real_file_reads_spectral_fields_as_gridpoint():
    resource = NativeFAResource(str(REAL_SAMPLE))

    temperature = resource.readfield("S001TEMPERATURE")
    log_surface_pressure = resource.readfield("SURFPRESSION")
    spectral_coefficients = resource.readfield("S001TEMPERATURE", convert_spectral=False)

    assert temperature.shape == (480, 480)
    assert log_surface_pressure.shape == (480, 480)
    assert spectral_coefficients.shape == (80020,)
    assert np.isfinite(temperature).all()
    assert np.isfinite(log_surface_pressure).all()
    assert 240.0 < float(temperature.min()) < 290.0
    assert 250.0 < float(temperature.max()) < 300.0
    surface_pressure = np.exp(log_surface_pressure)
    assert 90_000.0 < float(surface_pressure.min()) < 105_000.0
    assert 95_000.0 < float(surface_pressure.max()) < 105_000.0


def test_real_file_xarray_netcdf_and_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    xr = pytest.importorskip("xarray")
    import matplotlib.pyplot as plt
    import faxarray as fx

    fa = fx.open_fa(str(REAL_SAMPLE))
    try:
        ds = fa.to_xarray(
            variables=["SURFTEMPERATURE", "SURFGEOPOTEN", "S001TEMPERATURE", "SURFPRESSION"],
            progress=False,
        )
        output = tmp_path / "real-subset.nc"
        ds.to_netcdf(output)
        reopened = xr.open_dataset(output)
        try:
            assert set(reopened.data_vars) >= {
                "SURFTEMPERATURE",
                "SURFGEOPOTEN",
                "S001TEMPERATURE",
                "SURFPRESSION",
            }
            assert reopened["SURFTEMPERATURE"].shape[-2:] == (480, 480)
            assert reopened["S001TEMPERATURE"].shape[-2:] == (480, 480)
        finally:
            reopened.close()

        plot_path = tmp_path / "surftemperature.png"
        fa["SURFTEMPERATURE"].plot(use_cartopy=False)
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
    finally:
        fa.close()
