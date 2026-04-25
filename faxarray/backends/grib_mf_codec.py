"""ctypes bridge to the vendored legacy FA GRIB_MF codec."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
from typing import Iterable, List

import numpy as np


class FAGribMFError(RuntimeError):
    """Raised when the legacy FA GRIB_MF codec cannot decode or encode."""


class LegacyGribMFCodec:
    """Small native wrapper around rootpack ``DECOGA`` and ``CODEGA``."""

    def __init__(self) -> None:
        self._lib = ctypes.CDLL(str(_ensure_library()))
        self._decode = self._lib.faxarray_fa_decode_legacy
        self._decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._decode.restype = None

        self._encode = self._lib.faxarray_fa_encode_legacy
        self._encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._encode.restype = None

    def decode(self, words: np.ndarray, output_size: int, spectral: bool = False) -> np.ndarray:
        """Decode a legacy ``KNGRIB=1/2`` FA field article."""

        kngr = int(words[0])
        if kngr not in (1, 2):
            raise FAGribMFError(f"legacy GRIB_MF only handles KNGRIB=1/2, got {kngr}")

        idecal = _legacy_payload_offset(kngr, spectral)
        kgrib = np.ascontiguousarray(words[idecal:], dtype=np.int64)
        values = np.zeros(int(output_size), dtype=np.float64)
        kerr = ctypes.c_int()
        kjlenf = ctypes.c_int()
        kbits = ctypes.c_int()
        kword = ctypes.c_int()

        self._decode(
            kgrib.ctypes.data,
            int(kgrib.size),
            int(values.size),
            64,
            int(words[idecal - 2]),
            int(words[idecal - 1]),
            kngr == 2,
            values.ctypes.data,
            ctypes.byref(kerr),
            ctypes.byref(kjlenf),
            ctypes.byref(kbits),
            ctypes.byref(kword),
        )

        if kerr.value != 0:
            raise FAGribMFError(f"DECOGA failed with code {kerr.value}")
        if kjlenf.value < output_size:
            raise FAGribMFError(f"DECOGA decoded {kjlenf.value} values, expected {output_size}")
        return values[:output_size]

    def encode(
        self,
        words: np.ndarray,
        values: np.ndarray,
        spectral: bool = False,
    ) -> np.ndarray:
        """Encode values back into a legacy packed FA article layout."""

        kngr = int(words[0])
        if kngr not in (1, 2):
            raise FAGribMFError(f"legacy GRIB_MF only handles KNGRIB=1/2, got {kngr}")

        idecal = _legacy_payload_offset(kngr, spectral)
        kbits = int(words[2])
        template_kgrib = np.ascontiguousarray(words[idecal:], dtype=np.int64)
        encoded_kgrib = np.zeros_like(template_kgrib)
        native_values = np.ascontiguousarray(np.ravel(values), dtype=np.float64)
        kword = ctypes.c_int()
        kerr = ctypes.c_int()
        pmin_bits = ctypes.c_int64()
        pmax_bits = ctypes.c_int64()

        self._encode(
            template_kgrib.ctypes.data,
            int(template_kgrib.size),
            native_values.ctypes.data,
            int(native_values.size),
            64,
            kbits,
            kngr == 2,
            encoded_kgrib.ctypes.data,
            int(encoded_kgrib.size),
            ctypes.byref(kword),
            ctypes.byref(kerr),
            ctypes.byref(pmin_bits),
            ctypes.byref(pmax_bits),
        )

        if kerr.value != 0:
            raise FAGribMFError(f"CODEGA failed with code {kerr.value}")
        if kword.value > template_kgrib.size:
            raise FAGribMFError(
                f"CODEGA produced {kword.value} words, template has {template_kgrib.size}"
            )

        replacement = np.array(words, dtype=np.int64, copy=True)
        replacement[idecal - 2] = pmin_bits.value
        replacement[idecal - 1] = pmax_bits.value
        replacement[idecal:] = 0
        replacement[idecal : idecal + kword.value] = encoded_kgrib[: kword.value]
        return replacement


_CODEC: LegacyGribMFCodec | None = None


def get_legacy_codec() -> LegacyGribMFCodec:
    global _CODEC
    if _CODEC is None:
        _CODEC = LegacyGribMFCodec()
    return _CODEC


def _legacy_payload_offset(kngr: int, spectral: bool) -> int:
    return 1 + 2 * kngr + (2 if spectral else 0)


def _ensure_library() -> Path:
    lib_path = _library_cache_path()
    if lib_path.exists():
        return lib_path

    compiler = shlex.split(os.environ.get("FC", "gfortran"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise FAGribMFError(
            "legacy packed FA fields need gfortran to build the vendored GRIB_MF codec"
        )

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    _build_library(compiler, lib_path)
    return lib_path


def _build_library(compiler: List[str], lib_path: Path) -> None:
    sources = _fortran_sources()
    objects = [source.with_suffix(".o").name for source in sources]
    compile_cmd = [
        *compiler,
        "-cpp",
        "-fPIC",
        "-fallow-argument-mismatch",
        *_endian_flags(),
        "-c",
        *[str(source) for source in sources],
    ]
    link_cmd = [*compiler, _shared_flag(), "-o", str(lib_path), *objects]

    try:
        subprocess.run(
            compile_cmd,
            cwd=lib_path.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            link_cmd,
            cwd=lib_path.parent,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        raise FAGribMFError(f"failed to build FA GRIB_MF codec:\n{output}") from exc


def _fortran_sources() -> List[Path]:
    root = Path(__file__).resolve().parents[1] / "_vendor" / "ifsaux"
    return [
        root / "module" / "parkind1.F90",
        root / "module" / "lfi_precision.F90",
        root / "shim" / "yomhook_stub.F90",
        root / "shim" / "sdl_mod_stub.F90",
        root / "shim" / "oml_mod_stub.F90",
        *sorted((root / "grib_mf").glob("*.F")),
        root / "shim" / "fa_grib_mf_codec.F90",
    ]


def _library_cache_path() -> Path:
    digest = _source_digest(_fortran_sources())
    return _cache_root() / "fa_grib_mf" / digest / _library_name()


def _source_digest(sources: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    hasher.update(platform.platform().encode("utf-8"))
    hasher.update(sys.version.encode("utf-8"))
    hasher.update(sys.byteorder.encode("utf-8"))
    for source in sources:
        hasher.update(str(source.name).encode("utf-8"))
        hasher.update(source.read_bytes())
    return hasher.hexdigest()[:16]


def _cache_root() -> Path:
    override = os.environ.get("FAXARRAY_NATIVE_CACHE")
    if override:
        return Path(override)
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "faxarray"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "faxarray"


def _library_name() -> str:
    if platform.system() == "Darwin":
        return "libfaxarray_fa_grib_mf.dylib"
    if platform.system() == "Windows":
        return "faxarray_fa_grib_mf.dll"
    return "libfaxarray_fa_grib_mf.so"


def _shared_flag() -> str:
    if platform.system() == "Darwin":
        return "-dynamiclib"
    return "-shared"


def _endian_flags() -> List[str]:
    if sys.byteorder == "little":
        return ["-DLITTLE"]
    return []
