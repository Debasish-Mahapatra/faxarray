"""Pure Python codec for legacy FA GRIB_MF simple packing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class FAGribMFError(RuntimeError):
    """Raised when the legacy FA GRIB_MF codec cannot decode or encode."""


@dataclass(frozen=True)
class _LegacyMessage:
    payload: bytes
    bds_offset: int
    bds_length: int
    bds_flag: int
    unused_bits: int
    binary_scale: int
    reference_value: float
    bits_per_value: int
    data_offset: int
    data_count: int


class LegacyGribMFCodec:
    """Decode and encode legacy ``KNGRIB=1/2`` FA packed payloads."""

    def decode(self, words: np.ndarray, output_size: int, spectral: bool = False) -> np.ndarray:
        """Decode a legacy ``KNGRIB=1/2`` FA field article."""

        kngr = int(words[0])
        if kngr not in (1, 2):
            raise FAGribMFError(f"legacy GRIB_MF only handles KNGRIB=1/2, got {kngr}")

        idecal = _legacy_payload_offset(kngr, spectral)
        message = _parse_message(_words_to_payload_bytes(words[idecal:]))
        if message.data_count < output_size:
            raise FAGribMFError(
                f"legacy GRIB_MF decoded {message.data_count} values, expected {output_size}"
            )

        packed = _unpack_unsigned(
            message.payload[message.data_offset : message.bds_offset + message.bds_length],
            message.bits_per_value,
            int(output_size),
        )
        if kngr == 2:
            pmin = _float_from_word(int(words[idecal - 2]))
            pmax = _float_from_word(int(words[idecal - 1]))
            return _decode_arpege_values(packed, message.bits_per_value, pmin, pmax)
        return message.reference_value + packed.astype(np.float64) * (2.0 ** message.binary_scale)

    def encode(
        self,
        words: np.ndarray,
        values: np.ndarray,
        spectral: bool = False,
    ) -> np.ndarray:
        """Encode values back into a legacy packed FA article layout."""

        raise FAGribMFError("pure Python legacy GRIB_MF encoding is not implemented yet")


_CODEC: LegacyGribMFCodec | None = None


def get_legacy_codec() -> LegacyGribMFCodec:
    global _CODEC
    if _CODEC is None:
        _CODEC = LegacyGribMFCodec()
    return _CODEC


def _legacy_payload_offset(kngr: int, spectral: bool) -> int:
    return 1 + 2 * kngr + (2 if spectral else 0)


def _parse_message(payload: bytes) -> _LegacyMessage:
    if len(payload) < 40 or payload[:4] != b"GRIB":
        raise FAGribMFError("legacy GRIB_MF payload does not start with GRIB")

    pos = 4
    is_new = payload[pos : pos + 4] == b"\x00\x00\x18\x00"
    if is_new:
        pds_length = _read_u24(payload, pos)
        block_flag = payload[pos + 7]
        has_grid = block_flag in (128, 192)
        has_bitmap = block_flag in (64, 192)
    else:
        pds_length = 20
        block_flag = payload[pos + 3]
        has_grid = block_flag in (1, 3)
        has_bitmap = block_flag in (2, 3)

    pos = 4 + pds_length
    if has_grid:
        grid_length = _read_u24(payload, pos)
        pos += grid_length
    if has_bitmap:
        raise FAGribMFError("legacy GRIB_MF bitmap fields are not supported")

    bds_offset = pos
    bds_length = _read_u24(payload, bds_offset)
    if bds_length < 11 or bds_offset + bds_length > len(payload):
        raise FAGribMFError("legacy GRIB_MF binary data block has an invalid length")

    bds_flag = payload[bds_offset + 3]
    if is_new:
        representation = bds_flag // 128
        unused_bits = bds_flag % 16
    else:
        representation = bds_flag // 16
        unused_bits = bds_flag - representation * 16
    if representation != 0:
        raise FAGribMFError("legacy GRIB_MF complex binary data blocks are not supported")

    binary_scale = _decode_signed_scale(_read_u16(payload, bds_offset + 4))
    reference_value = _decode_grib_float(
        payload[bds_offset + 6],
        _read_u24(payload, bds_offset + 7),
    )
    bits_per_value = payload[bds_offset + 10]
    if bits_per_value <= 0 or bits_per_value > 64:
        raise FAGribMFError(f"invalid legacy GRIB_MF bits per value: {bits_per_value}")

    data_bits = (bds_length - 11) * 8 - unused_bits
    if data_bits < 0:
        raise FAGribMFError("legacy GRIB_MF binary data block has invalid unused bits")
    data_count = data_bits // bits_per_value
    return _LegacyMessage(
        payload=payload,
        bds_offset=bds_offset,
        bds_length=bds_length,
        bds_flag=bds_flag,
        unused_bits=unused_bits,
        binary_scale=binary_scale,
        reference_value=reference_value,
        bits_per_value=bits_per_value,
        data_offset=bds_offset + 11,
        data_count=data_count,
    )


def _words_to_payload_bytes(words: np.ndarray) -> bytes:
    chunks = bytearray()
    for word in np.ravel(words):
        chunks.extend((int(word) & ((1 << 64) - 1)).to_bytes(8, "big", signed=False))
    return bytes(chunks)


def _unpack_unsigned(data: bytes, width: int, count: int) -> np.ndarray:
    if count < 0:
        raise FAGribMFError("legacy GRIB_MF output size cannot be negative")
    if count == 0:
        return np.zeros(0, dtype=np.uint64)

    needed_bits = int(width) * int(count)
    needed_bytes = (needed_bits + 7) // 8
    if len(data) < needed_bytes:
        raise FAGribMFError("legacy GRIB_MF packed data is shorter than expected")

    byte_values = np.frombuffer(data[:needed_bytes], dtype=np.uint8)
    bits = np.unpackbits(byte_values, bitorder="big")[:needed_bits]
    bit_rows = bits.reshape((count, width)).astype(np.uint64, copy=False)
    weights = np.left_shift(
        np.uint64(1),
        np.arange(width - 1, -1, -1, dtype=np.uint64),
    )
    return bit_rows @ weights


def _decode_arpege_values(
    packed: np.ndarray,
    bits_per_value: int,
    pmin: float,
    pmax: float,
) -> np.ndarray:
    max_code = (1 << bits_per_value) - 1
    if max_code <= 0:
        raise FAGribMFError(f"invalid legacy GRIB_MF bits per value: {bits_per_value}")
    scale = (pmax - pmin) / float(max_code)
    if not math.isfinite(scale) or scale <= 0.0:
        return np.full(packed.size, pmin, dtype=np.float64)

    packed_float = packed.astype(np.float64)
    values = np.empty(packed.size, dtype=np.float64)
    lower = packed < (1 << (bits_per_value - 1))
    values[lower] = pmin + scale * packed_float[lower]
    values[~lower] = pmax - scale * (float(max_code) - packed_float[~lower])
    return values


def _decode_signed_scale(raw: int) -> int:
    if raw <= 2**15:
        return int(raw)
    return int(2**15 - raw)


def _decode_grib_float(exponent: int, mantissa: int) -> float:
    if exponent < 128:
        return float(mantissa) * (16.0 ** (int(exponent) - 70))
    return -float(mantissa) * (16.0 ** (int(exponent) - 198))


def _float_from_word(word: int) -> float:
    return np.array([np.int64(word)], dtype=np.int64).view(np.float64)[0].item()


def _read_u16(data: bytes, offset: int) -> int:
    return (data[offset] << 8) | data[offset + 1]


def _read_u24(data: bytes, offset: int) -> int:
    return (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]
