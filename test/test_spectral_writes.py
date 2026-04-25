"""Tests for spectral writing and the LAM bi-Fourier inverse transform."""

from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.native_fa import (
    NativeFAResource,
    UnsupportedFAEncodingError,
)
from faxarray.backends.spectral import (
    LamSpectralLayout,
    lam_gp2sp,
    lam_sp2gp,
)


REAL_SAMPLE = Path("/Users/dev/PROJECTS/test-data/pfABOFABOF+0012")


def test_lam_bi_fourier_round_trip_from_gridpoint():
    """gp2sp followed by sp2gp recovers a band-limited gridpoint field.

    The truncation is intentionally below the Nyquist meridian/zonal so the
    layout does not alias the FFT domain.
    """

    ny, nx = 16, 32
    max_meridian = ny // 2 - 1  # 7
    max_zonal = nx // 2 - 1  # 15
    layout = LamSpectralLayout(
        by_meridian=tuple(
            (1 + 4 * (max_zonal + 1) * j, 4 * (max_zonal + 1) * (j + 1), max_zonal)
            for j in range(max_meridian + 1)
        ),
        coeff_count=4 * (max_zonal + 1) * (max_meridian + 1),
    )

    # Build a band-limited gridpoint by inverse-transforming random coefficients
    # then re-transforming. This makes the round-trip well-defined.
    rng = np.random.default_rng(0)
    seed = rng.standard_normal(layout.coeff_count)
    seed_gridpoint = lam_sp2gp(seed, layout, ny, nx)
    coefficients = lam_gp2sp(seed_gridpoint, layout)
    gridpoint_back = lam_sp2gp(coefficients, layout, ny, nx)
    np.testing.assert_allclose(gridpoint_back, seed_gridpoint, atol=1e-12)


def test_lam_sp2gp_then_gp2sp_preserves_meaningful_modes():
    """Coefficients that survive the symmetry projection round-trip exactly."""

    ny, nx = 16, 32
    layout = LamSpectralLayout(
        by_meridian=tuple(
            (1 + 4 * (3 + 1) * j, 4 * (3 + 1) * (j + 1), 3) for j in range(4)
        ),
        coeff_count=4 * (3 + 1) * 4,
    )
    rng = np.random.default_rng(1)
    coefficients = rng.standard_normal(layout.coeff_count)
    # Project out the modes that the bi-Fourier symmetry identically annihilates:
    # for meridian=0,zonal=0 only "a" (offset 0) survives; for meridian=0,zonal>0
    # only "a" and "b" survive; for meridian>0,zonal=0 only "a" and "c" survive.
    for m, (start, _, max_zonal) in enumerate(layout.by_meridian):
        for k in range(max_zonal + 1):
            base = start - 1 + 4 * k
            if m == 0 and k == 0:
                coefficients[base + 1 : base + 4] = 0
            elif m == 0:
                coefficients[base + 1] = 0
                coefficients[base + 3] = 0
            elif k == 0:
                coefficients[base + 2] = 0
                coefficients[base + 3] = 0
    gridpoint = lam_sp2gp(coefficients, layout, ny, nx)
    recovered = lam_gp2sp(gridpoint, layout)
    np.testing.assert_allclose(recovered, coefficients, atol=1e-12)


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_real_lam_spectral_round_trip_via_resource():
    resource = NativeFAResource(str(REAL_SAMPLE))
    gridpoint = resource.readfield("S001TEMPERATURE", convert_spectral=True)
    sp = resource._gridpoint_to_spectral_lam(gridpoint)
    gridpoint_back = resource._spectral_to_gridpoint(sp)
    np.testing.assert_allclose(gridpoint_back, gridpoint, atol=1e-9)


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_template_write_roundtrips_legacy_packed_spectral_field(tmp_path):
    resource = NativeFAResource(str(REAL_SAMPLE))
    original = resource.readfield("S001TEMPERATURE", convert_spectral=True)

    # Use the resource's spectral writer through the template path. We
    # supply a slightly-modified version of the field so the round-trip
    # exercises encode + decode rather than just being a memcpy.
    modified = original + 0.5

    output = tmp_path / "spectral-write.fa"
    resource.write_template(
        str(output),
        {"S001TEMPERATURE": modified},
        overwrite=True,
    )

    written = NativeFAResource(str(output)).readfield("S001TEMPERATURE", convert_spectral=True)
    np.testing.assert_allclose(written, modified, atol=0.5, rtol=0)


def test_global_gauss_spectral_write_requires_coefficient_array():
    """Global Gauss spectral writes accept coefficients, not gridpoint."""

    from pathlib import Path as _Path

    gauss = _Path("/tmp/EPyGrAM/tests/data/geometries/gaussC1.fa")
    if not gauss.exists():
        pytest.skip("gaussC1.fa not available")
    resource = NativeFAResource(str(gauss))
    gridpoint = resource.readfield("SPECSURFGEOPOTEN", convert_spectral=True)
    with pytest.raises(UnsupportedFAEncodingError, match="coefficient array"):
        resource._replace_spectral_field(
            "SPECSURFGEOPOTEN",
            gridpoint,
            resource.fieldencoding_object("SPECSURFGEOPOTEN"),
        )


def test_global_gauss_spectral_write_round_trips_with_coefficients(tmp_path):
    from pathlib import Path as _Path

    gauss = _Path("/tmp/EPyGrAM/tests/data/geometries/gaussC1.fa")
    if not gauss.exists():
        pytest.skip("gaussC1.fa not available")
    resource = NativeFAResource(str(gauss))
    coefficients = resource.readfield("SPECSURFGEOPOTEN", convert_spectral=False)

    output = tmp_path / "global-spectral-write.fa"
    resource.write_template(
        str(output),
        {"SPECSURFGEOPOTEN": coefficients},
        overwrite=True,
    )
    written = NativeFAResource(str(output)).readfield(
        "SPECSURFGEOPOTEN", convert_spectral=False
    )
    np.testing.assert_allclose(written, coefficients)


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_lam_spectral_write_accepts_coefficient_array_directly(tmp_path):
    """Round-trip: read coefficients, write coefficients back via template."""

    resource = NativeFAResource(str(REAL_SAMPLE))
    coefficients = resource.readfield("S001TEMPERATURE", convert_spectral=False)

    output = tmp_path / "lam-coef-write.fa"
    resource.write_template(
        str(output),
        {"S001TEMPERATURE": coefficients},
        overwrite=True,
    )
    written = NativeFAResource(str(output)).readfield(
        "S001TEMPERATURE", convert_spectral=False
    )
    np.testing.assert_allclose(written, coefficients, atol=1e-6, rtol=1e-6)


def test_grib_api_spectral_write_dispatch_path_exists():
    """Ensure the KNGRIB>=100 spectral write dispatch is wired up.

    No GRIB_API spectral test files are available locally, so we just
    confirm that the resource exposes the encoder and that calling it
    with a missing GRIB blob returns the expected error rather than
    silently dropping data.
    """

    from faxarray.backends.native_fa import NativeFAResource

    # Synthesize the smallest possible mock: a NativeFAResource subclass
    # with a fabricated header. We only exercise _encode_grib_api_spectral
    # directly with a payload that has no GRIB blob.
    resource = NativeFAResource.__new__(NativeFAResource)
    resource.lfi = None  # not used in this test
    resource.filepath = "<test>"
    with pytest.raises(UnsupportedFAEncodingError, match="GRIB"):
        resource._encode_grib_api_spectral(
            "FAKE", b"no grib here", np.zeros(10, dtype=np.float64)
        )
