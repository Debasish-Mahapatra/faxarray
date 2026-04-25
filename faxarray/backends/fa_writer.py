"""Build a brand-new FA file from native Python data.

This wraps :mod:`lfi_writer` with the FA-specific header articles and a
minimal field encoder. Two geometry families are supported by the
top-level :func:`create_fa_file` helper:

* **regular lon/lat (LAM)**: a rectangular grid expressed in degrees
* **global reduced Gauss**: optionally rotated/stretched (KTYPTR=1 or 2)

For projected LAM geometries (Lambert / Mercator / polar stereographic)
``create_fa_file`` raises ``NotImplementedError``. The projection
parameter encoding inside ``CADRE-SINLATITUD`` for those cases is more
involved than the public API needs today, so users should keep using
``write_fa()`` with an existing template for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime, timedelta
import math
import struct
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lfi_writer import LFIWriter


_RAW_FIELD_HEADER_WORDS = 2  # KNGRIB, spectral flag


@dataclass
class FAFieldData:
    """One H2D field to write into a new FA file.

    ``values`` must be a 2D ``(ny, nx)`` array for LAM/regular lon-lat
    geometry, or a 1D unstructured array for global Gauss geometry. The
    writer always stores the field as raw float64 (``KNGRIB=0``); use
    :func:`NativeFAResource.write_template` afterwards if you need
    packing.
    """

    name: str
    values: np.ndarray
    spectral: bool = False


@dataclass
class FAValidityInput:
    """Validity to write into ``DATE-DES-DONNEES`` / ``DATX-DES-DONNEES``."""

    base_time: datetime
    lead_time: timedelta = timedelta(0)
    process_type: int = 10  # forecast
    cumulative_duration: Optional[timedelta] = None


@dataclass
class FAVerticalInput:
    """Hybrid vertical coordinate to write into ``CADRE-FOCOHYBRID``."""

    reference_pressure: float = 101325.0
    a_coefficients: Sequence[float] = (0.0, 0.0)
    b_coefficients: Sequence[float] = (0.0, 1.0)


@dataclass
class FARegularLonLatGeometry:
    """Regular lon/lat geometry definition (degrees)."""

    nx: int
    ny: int
    lon0: float        # degrees, geographic centre
    lat0: float        # degrees
    dx: float          # degrees per grid step in x
    dy: float          # degrees per grid step in y
    truncation: int = 0


@dataclass
class FAGlobalGaussGeometry:
    """Global reduced-Gauss geometry definition.

    ``sin_lat_nh`` are sine-of-latitude values for the northern
    hemisphere (north pole to equator inclusive). ``lon_number_by_lat_nh``
    is the number of longitude points on each of those latitudes.
    Truncation defaults to ``len(sin_lat_nh) - 1`` if not supplied.
    """

    sin_lat_nh: Sequence[float]
    lon_number_by_lat_nh: Sequence[int]
    truncation: Optional[int] = None
    pole_sin_lat: float = 1.0
    pole_cos_lon: float = 1.0
    pole_sin_lon: float = 0.0
    stretching_factor: float = 1.0
    max_zonal_wavenumber_by_lat_nh: Optional[Sequence[int]] = None

    @property
    def is_rotated(self) -> bool:
        return abs(1.0 - self.pole_sin_lat) > 1e-10 or abs(self.pole_sin_lon) > 1e-10 or abs(self.stretching_factor - 1.0) > 1e-10

    @property
    def knlati(self) -> int:
        return 2 * len(self.sin_lat_nh)

    @property
    def knxlon(self) -> int:
        return int(max(self.lon_number_by_lat_nh))


def _pack_int64(values: Iterable[int], endian: str = ">") -> bytes:
    arr = np.asarray(list(values), dtype=f"{endian}i8")
    return arr.tobytes()


def _pack_float64(values: Iterable[float], endian: str = ">") -> bytes:
    arr = np.asarray(list(values), dtype=f"{endian}f8")
    return arr.tobytes()


def _pack_field_raw_float64(values: np.ndarray, spectral: bool, endian: str = ">") -> bytes:
    flat = np.ascontiguousarray(np.ravel(values), dtype=f"{endian}f8")
    header = struct.pack(f"{endian}qq", 0, 1 if spectral else 0)
    return header + flat.tobytes()


def _build_regular_lonlat_articles(
    geometry: FARegularLonLatGeometry,
    vertical: FAVerticalInput,
    endian: str,
) -> List[Tuple[str, bytes]]:
    nlevels = max(0, len(vertical.a_coefficients) - 1)
    ktronc = max(0, geometry.truncation)
    ktyptr = -max(1, ktronc) if ktronc > 0 else -1  # sign-encoded LAM marker

    dimensions = _pack_int64(
        [ktronc, geometry.ny, geometry.nx, nlevels, ktyptr], endian=endian
    )
    franchschmi = _pack_float64([0.0, 0.0, 0.0, 0.0], endian=endian)

    sinlat = [
        -1.0,
        -9.0,  # regular lon/lat marker (read by NativeFAResource)
        0.0,
        0.0,
        math.radians(geometry.lon0),
        math.radians(geometry.lat0),
        math.radians(geometry.dx),
        math.radians(geometry.dy),
        0.0,
        0.0,
        float(geometry.nx) * geometry.dx,
        float(geometry.ny) * geometry.dy,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    sinlat_blob = _pack_float64(sinlat, endian=endian)

    # Minimal CADRE-REDPOINPOL: 8 projection params + 2 ISULEI + 4-per-meridian
    # spectral-by-row block (zero-filled when there are no spectral fields).
    isulei = max(ktronc, 1)
    redpoints: List[int] = [
        10,        # JNEXPL flag
        -1,        # geometry marker for LAM
        1, geometry.nx,  # X CI zone bounds
        1, geometry.ny,  # Y CI zone bounds
        8, 8,            # sponge widths
        isulei, isulei,  # ISULEI / duplicate
    ]
    # Zero-padded spectral by-meridian table so the reader's spectral path
    # (if a future caller adds spectral fields) doesn't crash on first use.
    redpoints.extend([0] * (4 * (ktronc + 1)))
    red_blob = _pack_int64(redpoints, endian=endian)

    a_coefficients = list(vertical.a_coefficients) or [0.0, 0.0]
    b_coefficients = list(vertical.b_coefficients) or [0.0, 1.0]
    if len(a_coefficients) != len(b_coefficients):
        raise ValueError("vertical A/B coefficient arrays must have matching length")
    focohybrid = _pack_float64(
        [vertical.reference_pressure, *a_coefficients, *b_coefficients], endian=endian
    )

    return [
        ("CADRE-DIMENSIONS", dimensions),
        ("CADRE-FRANKSCHMI", franchschmi),
        ("CADRE-REDPOINPOL", red_blob),
        ("CADRE-SINLATITUD", sinlat_blob),
        ("CADRE-FOCOHYBRID", focohybrid),
    ]


def _build_global_gauss_articles(
    geometry: FAGlobalGaussGeometry,
    vertical: FAVerticalInput,
    endian: str,
) -> List[Tuple[str, bytes]]:
    inpahe = len(geometry.sin_lat_nh)
    if len(geometry.lon_number_by_lat_nh) != inpahe:
        raise ValueError(
            "lon_number_by_lat_nh and sin_lat_nh must have the same length"
        )
    knlati = geometry.knlati
    knxlon = geometry.knxlon
    nlevels = max(0, len(vertical.a_coefficients) - 1)
    ktronc = geometry.truncation if geometry.truncation is not None else inpahe - 1
    ktyptr = 2 if geometry.is_rotated else 1

    dimensions = _pack_int64([ktronc, knlati, knxlon, nlevels, ktyptr], endian=endian)

    franchschmi = _pack_float64(
        [
            geometry.pole_sin_lat,
            geometry.pole_cos_lon,
            geometry.pole_sin_lon,
            geometry.stretching_factor,
        ],
        endian=endian,
    )

    sinlat_blob = _pack_float64(geometry.sin_lat_nh, endian=endian)

    if geometry.max_zonal_wavenumber_by_lat_nh is not None:
        knozpa = list(geometry.max_zonal_wavenumber_by_lat_nh)
    else:
        knozpa = [0] * inpahe
    if len(knozpa) != inpahe:
        raise ValueError(
            "max_zonal_wavenumber_by_lat_nh must have the same length as sin_lat_nh"
        )
    redpoints = list(geometry.lon_number_by_lat_nh) + knozpa
    red_blob = _pack_int64(redpoints, endian=endian)

    a_coefficients = list(vertical.a_coefficients) or [0.0, 0.0]
    b_coefficients = list(vertical.b_coefficients) or [0.0, 1.0]
    if len(a_coefficients) != len(b_coefficients):
        raise ValueError("vertical A/B coefficient arrays must have matching length")
    focohybrid = _pack_float64(
        [vertical.reference_pressure, *a_coefficients, *b_coefficients], endian=endian
    )

    return [
        ("CADRE-DIMENSIONS", dimensions),
        ("CADRE-FRANKSCHMI", franchschmi),
        ("CADRE-REDPOINPOL", red_blob),
        ("CADRE-SINLATITUD", sinlat_blob),
        ("CADRE-FOCOHYBRID", focohybrid),
    ]


def _build_validity_articles(validity: FAValidityInput, endian: str) -> List[Tuple[str, bytes]]:
    base = validity.base_time
    lead_seconds = int(validity.lead_time.total_seconds())
    process_type = int(validity.process_type)

    # DATE format: [year, month, day, hour, minute, unit, term, ?, process_type, ?, ?]
    # The native reader prefers DATX (seconds) when present; we set both.
    if lead_seconds % 3600 == 0:
        unit, term = 1, lead_seconds // 3600
    elif lead_seconds % 86400 == 0:
        unit, term = 2, lead_seconds // 86400
    else:
        unit, term = 0, 0

    date = [
        base.year,
        base.month,
        base.day,
        base.hour,
        base.minute,
        unit,
        term,
        0,
        process_type,
        0,
        0,
    ]
    cumulative = (
        int(validity.cumulative_duration.total_seconds())
        if validity.cumulative_duration is not None
        else 0
    )
    datx = [
        base.year,
        base.month,
        base.day,
        lead_seconds,    # the reader picks this up as datx[3]
        cumulative,      # extra slot for cumulative duration
        0,
        0,
        0,
        process_type,
        0,
        0,
    ]
    date_blob = _pack_int64(date, endian=endian)
    datx_blob = _pack_int64(datx, endian=endian)
    return [
        ("DATE-DES-DONNEES", date_blob),
        ("DATX-DES-DONNEES", datx_blob),
    ]


def create_fa_file(
    path: str,
    geometry,
    fields: Sequence[FAFieldData],
    validity: Optional[FAValidityInput] = None,
    vertical: Optional[FAVerticalInput] = None,
    page_size_bytes: int = 24576,
    endian: str = ">",
) -> None:
    """Create a new FA file from native Python data.

    Parameters
    ----------
    path : str
        Output file path.
    geometry : FARegularLonLatGeometry or FAGlobalGaussGeometry
        Grid geometry. Projected LAM geometries (Lambert/Mercator/polar
        stereo) are not supported by this writer yet.
    fields : sequence of FAFieldData
        H2D fields to write. Each field is encoded as raw float64
        (``KNGRIB=0``). Wrap with :func:`NativeFAResource.write_template`
        if you need to repack.
    validity : FAValidityInput, optional
        Defaults to a placeholder analysis at 0001-01-15 00:00 UTC.
    vertical : FAVerticalInput, optional
        Defaults to a single hybrid layer with ``A=[0,0]`` and ``B=[0,1]``.
    page_size_bytes : int, default 24576
        LFI page size. Larger values let more articles fit in a single
        index section.
    endian : str, default ">"
        ``">"`` for big-endian (matches existing FA samples).
    """

    if validity is None:
        validity = FAValidityInput(base_time=datetime(1, 1, 15))
    if vertical is None:
        vertical = FAVerticalInput()

    if isinstance(geometry, FARegularLonLatGeometry):
        header_articles = _build_regular_lonlat_articles(geometry, vertical, endian)
        expected_shape = (geometry.ny, geometry.nx)
        is_global_gauss = False
    elif isinstance(geometry, FAGlobalGaussGeometry):
        header_articles = _build_global_gauss_articles(geometry, vertical, endian)
        total_points = int(sum(geometry.lon_number_by_lat_nh)) * 2
        expected_shape = (1, total_points)
        is_global_gauss = True
    else:
        raise NotImplementedError(
            f"FA creation for geometry type {type(geometry).__name__} is not supported yet. "
            "Use NativeFAResource.write_template() with an existing template instead."
        )

    validity_articles = _build_validity_articles(validity, endian)

    writer = LFIWriter(path, page_size_bytes=page_size_bytes, endian=endian)
    for name, payload in header_articles:
        writer.add_article(name, payload)
    for name, payload in validity_articles:
        writer.add_article(name, payload)

    for fa_field in fields:
        values = np.asarray(fa_field.values)
        squeezed = np.squeeze(values)
        if is_global_gauss:
            # Accept either a 1D unstructured array or the 2D (1, N) shape.
            if squeezed.ndim == 2 and squeezed.shape[0] == 1:
                squeezed = squeezed[0]
            if squeezed.ndim != 1 or squeezed.size != expected_shape[1]:
                raise ValueError(
                    f"field {fa_field.name!r} has shape {values.shape}, expected "
                    f"a flat array of length {expected_shape[1]}"
                )
        else:
            if squeezed.shape != expected_shape:
                raise ValueError(
                    f"field {fa_field.name!r} has shape {values.shape}, expected "
                    f"{expected_shape}"
                )
        writer.add_article(
            fa_field.name,
            _pack_field_raw_float64(squeezed, fa_field.spectral, endian=endian),
        )

    writer.write()
