"""Minimal ctypes access to system ecCodes for GRIB-backed FA fields."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

import numpy as np


class EccodesCError(RuntimeError):
    """Raised when system ecCodes cannot decode a GRIB message."""


_LIB: Optional[ctypes.CDLL] = None


def decode_grib_values(message: bytes) -> np.ndarray:
    """Decode the ``values`` array from a GRIB message using system ecCodes."""

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
