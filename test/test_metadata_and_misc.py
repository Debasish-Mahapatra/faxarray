"""Tests for richer metadata accessors and Misc field exposure."""

from pathlib import Path

import numpy as np
import pytest

from faxarray.backends.native_fa import (
    NativeFAResource,
    NativeFAValidity,
    NativeFAVertical,
    UnsupportedFAEncodingError,
)


REAL_SAMPLE = Path("/Users/dev/PROJECTS/test-data/pfABOFABOF+0012")
GAUSS_C1 = Path("/tmp/EPyGrAM/tests/data/geometries/gaussC1.fa")


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_misc_field_lookup_and_read():
    resource = NativeFAResource(str(REAL_SAMPLE))
    misc = resource.list_misc_fields()
    assert "FULLPOS" in misc
    payload = resource.read_misc_field_bytes("FULLPOS")
    assert isinstance(payload, bytes)
    assert len(payload) == 8
    words = resource.read_misc_field_words("FULLPOS")
    assert words.shape == (1,)


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_vertical_coordinates_present():
    resource = NativeFAResource(str(REAL_SAMPLE))
    vertical = resource.vertical
    assert isinstance(vertical, NativeFAVertical)
    assert vertical.reference_pressure == pytest.approx(101325.0)
    assert vertical.n_levels == 87
    assert vertical.a_coefficients.shape == (88,)
    assert vertical.b_coefficients.shape == (88,)
    # Surface half-level is the bottom: A=0, B=1
    assert vertical.b_coefficients[-1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_validity_object_round_trip():
    resource = NativeFAResource(str(REAL_SAMPLE))
    validity = resource.validity
    assert isinstance(validity, NativeFAValidity)
    assert validity.base_time is not None
    assert validity.valid_time is not None
    assert validity.lead_time is not None
    # The legacy dict accessor still works.
    legacy = resource.get_validity()
    assert legacy["base_time"] == validity.base_time
    assert legacy["valid_time"] == validity.valid_time


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_field_encoding_includes_kngrib_label():
    resource = NativeFAResource(str(REAL_SAMPLE))
    encoding = resource.fieldencoding_object("S001TEMPERATURE")
    assert encoding.spectral
    assert encoding.kngrib in (1, 2)
    assert encoding.kngrib_label.startswith("legacy GRIB_MF")


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_metadata_summary_is_serialisable():
    import json

    resource = NativeFAResource(str(REAL_SAMPLE))
    snapshot = resource.metadata_summary()
    encoded = json.dumps(snapshot, default=str)
    decoded = json.loads(encoded)
    assert decoded["geometry"]["name"] == "mercator"
    assert decoded["header"]["nlevels"] == 87
    assert decoded["vertical"]["n_levels"] == 87


@pytest.mark.skipif(not GAUSS_C1.exists(), reason="gaussC1.fa not available")
def test_metadata_summary_for_global_gauss():
    resource = NativeFAResource(str(GAUSS_C1))
    snapshot = resource.metadata_summary()
    assert snapshot["geometry"]["is_global_gauss"]
    assert snapshot["geometry"]["projection"]["lat_number"] == 150


def test_create_fa_from_scratch_rejects_unknown_geometry_type():
    """Projected LAM (Lambert/Mercator/...) geometries are still out of scope."""

    from faxarray.backends.native_fa import create_fa_from_scratch

    class _DummyGeometry:  # not a regular_lonlat or global_gauss dataclass
        pass

    with pytest.raises(NotImplementedError) as exc_info:
        create_fa_from_scratch(
            "/tmp/should-not-exist.fa",
            geometry=_DummyGeometry(),
            fields=[],
        )
    assert "not supported" in str(exc_info.value).lower()


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_misc_field_read_via_h2d_path_raises():
    resource = NativeFAResource(str(REAL_SAMPLE))
    with pytest.raises(UnsupportedFAEncodingError):
        resource.read_misc_field_bytes("S001TEMPERATURE")
