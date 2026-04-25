"""Tests for the per-field metadata descriptor."""

from pathlib import Path

import pytest

from faxarray.backends.native_fa import NativeFAFieldDescriptor, NativeFAResource


REAL_SAMPLE = Path("/Users/dev/PROJECTS/test-data/pfABOFABOF+0012")
GAUSS_C1 = Path("/tmp/EPyGrAM/tests/data/geometries/gaussC1.fa")


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_descriptor_for_surface_field():
    resource = NativeFAResource(str(REAL_SAMPLE))
    descriptor = resource.field_descriptor("SURFTEMPERATURE")
    assert isinstance(descriptor, NativeFAFieldDescriptor)
    assert descriptor.level_type == "surface"
    assert descriptor.level_index is None
    assert descriptor.long_name == "Surface temperature"
    assert descriptor.units == "K"
    assert descriptor.fid["FA"] == "SURFTEMPERATURE"
    assert descriptor.encoding.spectral is False


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_descriptor_for_model_level_field_includes_AB():
    resource = NativeFAResource(str(REAL_SAMPLE))
    descriptor = resource.field_descriptor("S001TEMPERATURE")
    assert descriptor.level_type == "model"
    assert descriptor.level_index == 1
    assert descriptor.base_name == "TEMPERATURE"
    assert descriptor.a_coefficient is not None
    assert descriptor.b_coefficient is not None
    assert descriptor.encoding.spectral is True


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_descriptor_for_pressure_level_field():
    resource = NativeFAResource(str(REAL_SAMPLE))
    descriptor = resource.field_descriptor("P50000TEMPERATURE")
    assert descriptor.level_type == "pressure"
    assert descriptor.pressure_pa == 50000
    assert descriptor.base_name == "TEMPERATURE"


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_descriptor_for_misc_field_is_serialisable():
    import json

    resource = NativeFAResource(str(REAL_SAMPLE))
    descriptor = resource.field_descriptor("FULLPOS")
    encoded = json.dumps(descriptor.as_dict(), default=str)
    assert "FULLPOS" in encoded
    decoded = json.loads(encoded)
    assert decoded["level_type"] in ("header", "other")


@pytest.mark.skipif(not GAUSS_C1.exists(), reason="gaussC1.fa not available")
def test_descriptor_for_global_gauss_field():
    resource = NativeFAResource(str(GAUSS_C1))
    descriptor = resource.field_descriptor("SURFGEOPOTENTIEL")
    assert descriptor.level_type == "surface"
    assert descriptor.encoding.kngrib == 0
    assert descriptor.encoding.spectral is False


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="real LAM FA file not available")
def test_descriptor_falls_back_to_base_name_metadata():
    """A model-level field with no exact catalog match falls back to TEMPERATURE."""

    resource = NativeFAResource(str(REAL_SAMPLE))
    # S087TEMPERATURE is unlikely to be in the catalog as-is, but TEMPERATURE is.
    descriptor = resource.field_descriptor("S087TEMPERATURE")
    assert descriptor.long_name == "Temperature"
    assert descriptor.units == "K"
