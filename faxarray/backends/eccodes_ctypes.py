"""Minimal ctypes access to system ecCodes for GRIB-backed FA fields."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class EccodesCError(RuntimeError):
    """Raised when system ecCodes cannot decode a GRIB message."""


_LIB: Optional[ctypes.CDLL] = None


def decode_grib_values(message: bytes) -> np.ndarray:
    """Decode the ``values`` array from a GRIB message using system ecCodes.

    Works for both gridpoint-packed messages and (for spectral-complex
    packings supported by ecCodes) the raw spectral coefficient array.
    The caller is responsible for any post-processing such as packing
    re-ordering or transform.
    """

    lib = _load_eccodes()
    buffer = ctypes.create_string_buffer(message)
    handle = lib.codes_handle_new_from_message(None, buffer, len(message))
    if not handle:
        raise EccodesCError("ecCodes could not create a handle from the GRIB message")

    try:
        size = ctypes.c_size_t()
        _check(lib, lib.codes_get_size(handle, b"values", ctypes.byref(size)))
        values = np.empty(size.value, dtype=np.float64)
        length = ctypes.c_size_t(size.value)
        _check(
            lib,
            lib.codes_get_double_array(
                handle,
                b"values",
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(length),
            ),
        )
        return values[: length.value].copy()
    finally:
        lib.codes_handle_delete(handle)


def decode_grib_spectral_message(message: bytes) -> Tuple[np.ndarray, dict]:
    """Decode a spectral GRIB message into (values, metadata).

    Returns the coefficient array exactly as ecCodes serialises it for the
    message, plus a small dict with ``J``, ``K``, ``M`` (truncation
    parameters), ``bitsPerValue`` and ``packingType``.
    """

    lib = _load_eccodes()
    buffer = ctypes.create_string_buffer(message)
    handle = lib.codes_handle_new_from_message(None, buffer, len(message))
    if not handle:
        raise EccodesCError("ecCodes could not create a handle from the GRIB message")

    try:
        meta = {
            "J": _get_long(lib, handle, b"J", default=0),
            "K": _get_long(lib, handle, b"K", default=0),
            "M": _get_long(lib, handle, b"M", default=0),
            "bitsPerValue": _get_long(lib, handle, b"bitsPerValue", default=0),
            "packingType": _get_string(lib, handle, b"packingType"),
        }
        size = ctypes.c_size_t()
        _check(lib, lib.codes_get_size(handle, b"values", ctypes.byref(size)))
        values = np.empty(size.value, dtype=np.float64)
        length = ctypes.c_size_t(size.value)
        _check(
            lib,
            lib.codes_get_double_array(
                handle,
                b"values",
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(length),
            ),
        )
        return values[: length.value].copy(), meta
    finally:
        lib.codes_handle_delete(handle)


def encode_grib_values(message: bytes, values: np.ndarray) -> bytes:
    """Encode a new GRIB message using ``message`` as a template.

    The packing parameters (``packingType``, ``bitsPerValue``, ...) are
    inherited from the template. Raises :class:`EccodesCError` if the
    new buffer would not fit in the template's allocated length.
    """

    lib = _load_eccodes()
    buffer = ctypes.create_string_buffer(message)
    handle = lib.codes_handle_new_from_message(None, buffer, len(message))
    if not handle:
        raise EccodesCError("ecCodes could not clone the template GRIB message")

    flat_values = np.ascontiguousarray(np.ravel(values), dtype=np.float64)
    try:
        _check(
            lib,
            lib.codes_set_double_array(
                handle,
                b"values",
                flat_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(flat_values.size),
            ),
        )
        out_buffer = ctypes.c_void_p()
        out_size = ctypes.c_size_t()
        _check(
            lib,
            lib.codes_get_message(
                handle,
                ctypes.byref(out_buffer),
                ctypes.byref(out_size),
            ),
        )
        if not out_buffer.value or out_size.value == 0:
            raise EccodesCError("ecCodes returned an empty message after encoding")
        return ctypes.string_at(out_buffer, out_size.value)
    finally:
        lib.codes_handle_delete(handle)


def _get_long(lib: ctypes.CDLL, handle: int, key: bytes, default: int) -> int:
    out = ctypes.c_long(default)
    rc = lib.codes_get_long(handle, key, ctypes.byref(out))
    if rc != 0:
        return default
    return int(out.value)


def _get_string(lib: ctypes.CDLL, handle: int, key: bytes) -> Optional[str]:
    buf = ctypes.create_string_buffer(64)
    size = ctypes.c_size_t(len(buf))
    rc = lib.codes_get_string(handle, key, buf, ctypes.byref(size))
    if rc != 0:
        return None
    return buf.value.decode("utf-8", errors="replace")


def _load_eccodes() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = _find_eccodes_library()
        if path is None:
            raise EccodesCError("system ecCodes library was not found")
        lib = ctypes.CDLL(path)
        lib.codes_handle_new_from_message.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.codes_handle_new_from_message.restype = ctypes.c_void_p
        lib.codes_handle_delete.argtypes = [ctypes.c_void_p]
        lib.codes_handle_delete.restype = None
        lib.codes_get_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.codes_get_size.restype = ctypes.c_int
        lib.codes_get_double_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.codes_get_double_array.restype = ctypes.c_int
        lib.codes_set_double_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        lib.codes_set_double_array.restype = ctypes.c_int
        lib.codes_get_long.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_long),
        ]
        lib.codes_get_long.restype = ctypes.c_int
        lib.codes_get_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.codes_get_string.restype = ctypes.c_int
        lib.codes_get_message.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.codes_get_message.restype = ctypes.c_int
        lib.codes_get_error_message.argtypes = [ctypes.c_int]
        lib.codes_get_error_message.restype = ctypes.c_char_p
        _LIB = lib
    return _LIB


def _find_eccodes_library() -> Optional[str]:
    override = os.environ.get("ECCODES_LIBRARY")
    if override:
        return override

    found = ctypes.util.find_library("eccodes")
    if found:
        return found

    candidates = [
        "/opt/homebrew/lib/libeccodes.dylib",
        "/usr/local/lib/libeccodes.dylib",
        "/usr/lib/libeccodes.so",
        "/usr/local/lib/libeccodes.so",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _check(lib: ctypes.CDLL, code: int) -> None:
    if code == 0:
        return
    message = lib.codes_get_error_message(code)
    text = message.decode("utf-8", errors="replace") if message else f"ecCodes error {code}"
    raise EccodesCError(text)
