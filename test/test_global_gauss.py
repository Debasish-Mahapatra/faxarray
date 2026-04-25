"""Tests for global reduced-Gauss FA support.

These exercise the EPyGRAM sample geometries (gaussC1.fa is unstretched
non-rotated; gaussC2.4.fa is C2.4 stretched + rotated). They are skipped
when those sample files are not present locally.
"""

from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.native_fa import NativeFAResource
from faxarray.backends.spectral import (
    gauss_sp2gp,
    gaussian_latitudes_and_weights,
    make_gauss_layout,
)


GAUSS_C1 = Path("/tmp/EPyGrAM/tests/data/geometries/gaussC1.fa")
GAUSS_C24 = Path("/tmp/EPyGrAM/tests/data/geometries/gaussC2.4.fa")


pytestmark = pytest.mark.skipif(
    not GAUSS_C1.exists(), reason="EPyGRAM Gauss sample FA files not available"
)


def test_global_reduced_gauss_geometry_shape():
    resource = NativeFAResource(str(GAUSS_C1))
    geometry = resource.geometry
    assert geometry.name == "reduced_gauss"
    assert geometry.is_global_gauss
    assert geometry.shape[0] == 1
    assert geometry.projection["lat_number"] == 150
    assert geometry.projection["max_lon_number"] == 300
    assert geometry.projection["total_points"] == 33092
    assert geometry.lats.min() < -89.0
    assert geometry.lats.max() > 89.0
    assert 0.0 <= geometry.lons.min() < 1.0
    assert geometry.lons.max() <= 360.0
    assert geometry.projection["stretching_factor"] == pytest.approx(1.0)


def test_global_reduced_gauss_reads_uncompressed_field():
    resource = NativeFAResource(str(GAUSS_C1))
    field = resource.readfield("SURFGEOPOTENTIEL")
    assert field.shape == resource.geometry.shape
    assert field.size == resource.geometry.projection["total_points"]
    assert np.isfinite(field).all()


@pytest.mark.skipif(not GAUSS_C24.exists(), reason="gaussC2.4.fa not available")
def test_rotated_stretched_gauss_geometry():
    resource = NativeFAResource(str(GAUSS_C24))
    geometry = resource.geometry
    assert geometry.name == "rotated_reduced_gauss"
    assert geometry.projection["stretching_factor"] == pytest.approx(2.4)
    assert geometry.projection["pole_sin_lat"] == pytest.approx(0.725, abs=1e-6)
    # Rotation pulls latitudes away from the true poles slightly.
    assert geometry.lats.min() > -90.0
    assert geometry.lats.max() < 90.0
    # Longitudes after rotation can be in [-180, 180].
    assert geometry.lons.min() >= -180.0
    assert geometry.lons.max() <= 360.0


def test_global_spectral_layout_has_canonical_size():
    resource = NativeFAResource(str(GAUSS_C1))
    layout = make_gauss_layout(resource.header.ktronc)
    expected = (resource.header.ktronc + 1) ** 2
    assert layout.total_real_coeffs == expected


def test_global_gauss_spectral_to_gridpoint_runs():
    """The reference Legendre+FFT path executes and returns a finite field."""

    resource = NativeFAResource(str(GAUSS_C1))
    coefficients = resource.readfield("SPECSURFGEOPOTEN", convert_spectral=False)
    gridpoint = resource.readfield("SPECSURFGEOPOTEN", convert_spectral=True)
    assert coefficients.shape == ((resource.header.ktronc + 1) ** 2,)
    assert gridpoint.shape == resource.geometry.shape
    assert np.isfinite(gridpoint).all()


def test_gaussian_quadrature_helper_returns_north_to_south():
    sin_lats, weights = gaussian_latitudes_and_weights(8)
    assert sin_lats.shape == (8,)
    assert weights.shape == (8,)
    assert sin_lats[0] > sin_lats[-1]  # north (positive sin) before south
    assert weights.sum() == pytest.approx(2.0)


def test_global_spectral_dc_term_dominates_for_constant_field():
    """A flat coefficient array with only the (0,0) mode set should give
    a uniform gridpoint field whose value matches the analytic mean."""

    truncation = 3
    layout = make_gauss_layout(truncation)
    coefficients = np.zeros(layout.total_real_coeffs, dtype=np.float64)
    coefficients[0] = 4.2  # only the m=0,n=0 real coefficient
    sin_lats, _ = gaussian_latitudes_and_weights(truncation + 1)
    lon_number = [4 * (truncation + 1)] * len(sin_lats)
    gridpoint = gauss_sp2gp(coefficients, layout, sin_lats, lon_number)
    # The DC mode in the model layout uses Plm[0,0] = 1/sqrt(2),
    # so the gridpoint field is constant at coefficients[0] / sqrt(2).
    assert np.allclose(gridpoint, coefficients[0] / np.sqrt(2.0))
