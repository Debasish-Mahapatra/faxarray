import math

import numpy as np

from faxarray.backends.grib_mf_codec import LegacyGribMFCodec


def test_kngrib1_simple_encode_decode_round_trip():
    values = np.array([-3.25, -1.5, 0.25, 2.75, 5.0], dtype=np.float64)
    words = _template_words(kngr=1, bits=12, count=values.size)

    encoded = LegacyGribMFCodec().encode(words, values)
    decoded = LegacyGribMFCodec().decode(encoded, values.size)

    np.testing.assert_allclose(decoded, values, atol=4.0e-3)


def test_kngrib2_simple_encode_decode_round_trip():
    values = np.linspace(-2.0, 3.0, 17, dtype=np.float64)
    words = _template_words(kngr=2, bits=16, count=values.size)

    encoded = LegacyGribMFCodec().encode(words, values)
    decoded = LegacyGribMFCodec().decode(encoded, values.size)

    np.testing.assert_allclose(decoded, values, atol=1.0e-4)


def _template_words(kngr: int, bits: int, count: int) -> np.ndarray:
    payload = _template_payload(bits, count)
    payload_words = _payload_to_words(payload)
    if kngr == 1:
        return np.array([1, 0, bits, *payload_words], dtype=np.int64)
    if kngr == 2:
        return np.array([2, 0, bits, 0, 0, *payload_words], dtype=np.int64)
    raise ValueError(kngr)


def _template_payload(bits: int, count: int) -> bytes:
    pds = bytes([0, 0, 24, 0, 98, 1, 254, 0]) + bytes(16)
    total_data_bits = bits * count
    total_bits = 11 * 8 + total_data_bits
    unused_bits = (16 - (total_bits % 16)) % 16
    bds_length = (total_bits + unused_bits) // 8
    bds = bytearray(bds_length)
    bds[0:3] = bds_length.to_bytes(3, "big")
    bds[3] = unused_bits
    bds[10] = bits
    payload = b"GRIB" + pds + bytes(bds)
    padding = math.ceil(len(payload) / 8) * 8 - len(payload)
    return payload + b"\x00" * padding


def _payload_to_words(payload: bytes) -> list[int]:
    return [
        int.from_bytes(payload[index : index + 8], "big", signed=True)
        for index in range(0, len(payload), 8)
    ]
