"""ctypes bridge to the optional legacy FA GRIB_MF codec."""

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
import tarfile
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
    override = os.environ.get("FAXARRAY_GRIB_MF_LIBRARY")
    if override:
        lib_path = Path(override).expanduser()
        if not lib_path.exists():
            raise FAGribMFError(f"FAXARRAY_GRIB_MF_LIBRARY does not exist: {lib_path}")
        return lib_path

    sources = _fortran_sources()
    lib_path = _library_cache_path(sources)
    if lib_path.exists():
        return lib_path

    compiler = shlex.split(os.environ.get("FC", "gfortran"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise FAGribMFError(
            "legacy packed FA fields need gfortran to build the GRIB_MF codec"
        )

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    _build_library(compiler, lib_path, sources)
    return lib_path


def _build_library(compiler: List[str], lib_path: Path, sources: List[Path]) -> None:
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
    root = _external_ifsaux_root()
    shim_root = _ensure_shim_sources()
    sources = [
        root / "module" / "parkind1.F90",
        root / "module" / "lfi_precision.F90",
        shim_root / "yomhook_stub.F90",
        shim_root / "sdl_mod_stub.F90",
        shim_root / "oml_mod_stub.F90",
        *[root / "grib_mf" / name for name in _GRIB_MF_SOURCES],
        shim_root / "fa_grib_mf_codec.F90",
    ]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise FAGribMFError(
            "ifsaux source path is missing files needed for legacy FA packing:\n"
            + "\n".join(missing)
        )
    return sources


def _library_cache_path(sources: Iterable[Path]) -> Path:
    digest = _source_digest(sources)
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


_GRIB_MF_SOURCES = [
    "codega.F",
    "confi.F",
    "confp_mf.F",
    "decfp_mf.F",
    "decoga.F",
    "gbyte_mf.F",
    "gbytes_mf.F",
    "gsbite_mf.F",
    "gsbyte_mf.F",
    "mxmn_mf.F",
    "offset_mf.F",
    "packgb.F",
    "prtbin_mf.F",
    "sbyte_mf.F",
    "sbytes_mf.F",
    "unpagb.F",
]

_MODULE_SOURCES = ["parkind1.F90", "lfi_precision.F90"]

_SHIM_SOURCES = {
    "yomhook_stub.F90": """MODULE YOMHOOK
USE PARKIND1, ONLY : JPRB
IMPLICIT NONE
LOGICAL :: LHOOK = .FALSE.
CONTAINS
SUBROUTINE DR_HOOK(name, flag, handle)
CHARACTER(LEN=*), INTENT(IN) :: name
INTEGER, INTENT(IN) :: flag
REAL(KIND=JPRB), INTENT(INOUT) :: handle
handle = 0.0_JPRB
END SUBROUTINE DR_HOOK
END MODULE YOMHOOK
""",
    "sdl_mod_stub.F90": """MODULE SDL_MOD
IMPLICIT NONE
CONTAINS
SUBROUTINE SDL_SRLABORT()
ERROR STOP 'SDL_SRLABORT called from faxarray FA legacy codec'
END SUBROUTINE SDL_SRLABORT
END MODULE SDL_MOD
""",
    "oml_mod_stub.F90": """MODULE OML_MOD
IMPLICIT NONE
CONTAINS
INTEGER FUNCTION OML_MY_THREAD()
OML_MY_THREAD = 1
END FUNCTION OML_MY_THREAD
INTEGER FUNCTION OML_GET_NUM_THREADS()
OML_GET_NUM_THREADS = 1
END FUNCTION OML_GET_NUM_THREADS
END MODULE OML_MOD
""",
    "fa_grib_mf_codec.F90": """SUBROUTINE faxarray_fa_decode_legacy(kgrib, kleng, klenf, knbit, pmin_bits, pmax_bits, ldarpe, values, kerr, kjlenf, kbits, kword) BIND(C)
  USE ISO_C_BINDING, ONLY : c_bool, c_double, c_int, c_int64_t
  USE LFI_PRECISION
  IMPLICIT NONE
  INTEGER(c_int), VALUE :: kleng, klenf, knbit
  INTEGER(c_int64_t), INTENT(IN) :: kgrib(kleng)
  INTEGER(c_int64_t), VALUE :: pmin_bits, pmax_bits
  LOGICAL(c_bool), VALUE :: ldarpe
  REAL(c_double), INTENT(OUT) :: values(klenf)
  INTEGER(c_int), INTENT(OUT) :: kerr, kjlenf, kbits, kword

  INTEGER(KIND=JPLIKM) :: ib1par(19), ib2par(17)
  INTEGER(KIND=JPLIKM) :: ibits, icpack, ierr, ijlenf, ijlenv
  INTEGER(KIND=JPLIKM) :: ilenf, ileng, inbit, iscalp, iword
  INTEGER(KIND=JPLIKB), ALLOCATABLE :: igrib(:)
  LOGICAL :: ldarpe_f
  REAL(KIND=JPDBLD) :: zmax, zmin
  REAL(KIND=JPDBLD) :: zvalues(klenf), zvert(64)

  ileng = kleng
  ilenf = klenf
  inbit = knbit
  ALLOCATE(igrib(ileng))
  igrib = INT(kgrib, JPLIKB)
  zmin = TRANSFER(pmin_bits, zmin)
  zmax = TRANSFER(pmax_bits, zmax)
  ldarpe_f = LOGICAL(ldarpe)
  zvalues = 0.0_JPDBLD
  zvert = 0.0_JPDBLD

  CALL DECOGA(zvalues, ilenf, ibits, inbit, ib1par, ib2par, zvert, SIZE(zvert), &
       igrib, ileng, iword, ijlenv, ijlenf, icpack, iscalp, ierr, zmin, zmax, ldarpe_f)

  values = REAL(zvalues, c_double)
  kerr = ierr
  kjlenf = ijlenf
  kbits = ibits
  kword = iword
  DEALLOCATE(igrib)
END SUBROUTINE faxarray_fa_decode_legacy

SUBROUTINE faxarray_fa_encode_legacy(template_kgrib, template_kleng, values, klenf, knbit, kbits, ldarpe, out_kgrib, out_kleng, kword, kerr, pmin_bits, pmax_bits) BIND(C)
  USE ISO_C_BINDING, ONLY : c_bool, c_double, c_int, c_int64_t
  USE LFI_PRECISION
  IMPLICIT NONE
  INTEGER(c_int), VALUE :: template_kleng, klenf, knbit, kbits, out_kleng
  INTEGER(c_int64_t), INTENT(IN) :: template_kgrib(template_kleng)
  REAL(c_double), INTENT(IN) :: values(klenf)
  LOGICAL(c_bool), VALUE :: ldarpe
  INTEGER(c_int64_t), INTENT(OUT) :: out_kgrib(out_kleng)
  INTEGER(c_int), INTENT(OUT) :: kword, kerr
  INTEGER(c_int64_t), INTENT(OUT) :: pmin_bits, pmax_bits

  INTEGER(KIND=JPLIKM) :: ib1par(19), ib2par(17)
  INTEGER(KIND=JPLIKM) :: ibits, icpack, ierr, ijlenf, ijlenv
  INTEGER(KIND=JPLIKM) :: ilenf, inbit, iout_leng, iscalp, itemplate_leng, iword
  INTEGER(KIND=JPLIKB), ALLOCATABLE :: igrib_out(:), igrib_template(:)
  LOGICAL :: ldarpe_f
  REAL(KIND=JPDBLD) :: zmax, zmin
  REAL(KIND=JPDBLD) :: ztmp(1), zvalues(klenf), zvert(64)

  itemplate_leng = template_kleng
  iout_leng = out_kleng
  ilenf = klenf
  inbit = knbit
  ibits = kbits
  ALLOCATE(igrib_template(itemplate_leng))
  ALLOCATE(igrib_out(iout_leng))
  igrib_template = INT(template_kgrib, JPLIKB)
  igrib_out = 0_JPLIKB
  ldarpe_f = LOGICAL(ldarpe)
  ztmp = 0.0_JPDBLD
  zvert = 0.0_JPDBLD
  zmin = 0.0_JPDBLD
  zmax = 0.0_JPDBLD

  CALL DECOGA(ztmp, 1, ibits, inbit, ib1par, ib2par, zvert, SIZE(zvert), &
       igrib_template, itemplate_leng, iword, ijlenv, ijlenf, icpack, iscalp, ierr, &
       zmin, zmax, ldarpe_f)

  IF (ierr /= 0) THEN
    kerr = ierr
    kword = 0
    out_kgrib = 0_c_int64_t
    pmin_bits = 0_c_int64_t
    pmax_bits = 0_c_int64_t
    DEALLOCATE(igrib_template, igrib_out)
    RETURN
  ENDIF

  zvalues = REAL(values, JPDBLD)
  zmin = 0.0_JPDBLD
  zmax = 0.0_JPDBLD

  CALL CODEGA(zvalues, ilenf, ibits, inbit, ib1par, ib2par, zvert, MAX(ijlenv, 2), &
       igrib_out, iout_leng, iword, 0, 0, 0, ierr, zmin, zmax, ldarpe_f)

  out_kgrib = INT(igrib_out, c_int64_t)
  kword = iword
  kerr = ierr
  pmin_bits = TRANSFER(zmin, pmin_bits)
  pmax_bits = TRANSFER(zmax, pmax_bits)
  DEALLOCATE(igrib_template, igrib_out)
END SUBROUTINE faxarray_fa_encode_legacy
""",
}


def _external_ifsaux_root() -> Path:
    direct = os.environ.get("FAXARRAY_IFSAUX_ROOT")
    if direct:
        path = Path(direct).expanduser()
        if path.is_file():
            return _extract_ifsaux_from_tarball(path)
        return _normalise_ifsaux_root(path)

    tarball = os.environ.get("FAXARRAY_ROOTPACK_TARBALL")
    if tarball:
        return _extract_ifsaux_from_tarball(Path(tarball).expanduser())

    raise FAGribMFError(
        "faxarray does not ship rootpack or ifsaux model sources. "
        "Legacy KNGRIB=1/2 packed FA fields need an external source copy. "
        "Set FAXARRAY_IFSAUX_ROOT to an ifsaux source directory or unpacked "
        "rootpack tree, set FAXARRAY_ROOTPACK_TARBALL to a rootpack tarball, "
        "or set FAXARRAY_GRIB_MF_LIBRARY to a prebuilt compatible codec library."
    )


def _normalise_ifsaux_root(path: Path) -> Path:
    path = path.resolve()
    if _looks_like_ifsaux(path):
        return path

    direct = path / "src" / "local" / "ifsaux"
    if _looks_like_ifsaux(direct):
        return direct

    matches = [candidate for candidate in path.glob("*/src/local/ifsaux") if _looks_like_ifsaux(candidate)]
    if matches:
        return matches[0].resolve()

    raise FAGribMFError(
        f"could not find ifsaux sources under {path}. Expected grib_mf/ and module/ "
        "directories, or a rootpack tree containing */src/local/ifsaux."
    )


def _looks_like_ifsaux(path: Path) -> bool:
    return (path / "grib_mf").is_dir() and (path / "module").is_dir()


def _extract_ifsaux_from_tarball(tarball: Path) -> Path:
    tarball = tarball.resolve()
    if not tarball.exists():
        raise FAGribMFError(f"rootpack tarball does not exist: {tarball}")

    stat = tarball.stat()
    digest = hashlib.sha256(
        f"{tarball}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    dest = _cache_root() / "external_ifsaux" / digest / "ifsaux"
    if _required_source_files_exist(dest):
        return dest

    tmp = dest.parent / "ifsaux.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    needed = {
        ("module", name): tmp / "module" / name for name in _MODULE_SOURCES
    }
    needed.update({
        ("grib_mf", name): tmp / "grib_mf" / name for name in _GRIB_MF_SOURCES
    })

    with tarfile.open(tarball) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            key = None
            for index, part in enumerate(parts[:-2]):
                if part == "ifsaux":
                    candidate = (parts[index + 1], parts[index + 2])
                    if candidate in needed:
                        key = candidate
                    break
            if key is None:
                continue
            target = needed[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())

    if not _required_source_files_exist(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
        raise FAGribMFError(
            f"rootpack tarball does not contain the required ifsaux GRIB_MF sources: {tarball}"
        )

    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def _required_source_files_exist(root: Path) -> bool:
    required = [root / "module" / name for name in _MODULE_SOURCES]
    required.extend(root / "grib_mf" / name for name in _GRIB_MF_SOURCES)
    return all(path.exists() for path in required)


def _ensure_shim_sources() -> Path:
    root = _cache_root() / "fa_grib_mf_shims" / "v1"
    root.mkdir(parents=True, exist_ok=True)
    for name, content in _SHIM_SOURCES.items():
        path = root / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    return root
