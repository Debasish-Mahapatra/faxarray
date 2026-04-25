"""Native FA access built on the vendored LFI/FA format knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .eccodes_ctypes import (
    EccodesCError,
    decode_grib_spectral_message,
    decode_grib_values,
    encode_grib_values,
)
from .grib_mf_codec import FAGribMFError, get_legacy_codec
from .native_lfi import LFIFile
from .spectral import (
    GaussSpectralLayout,
    LamSpectralLayout,
    gauss_gp2sp,
    gauss_sp2gp,
    gaussian_latitudes_and_weights,
    lam_gp2sp,
    lam_sp2gp,
    make_gauss_layout,
)


_HEADER_ARTICLES = {
    "CADRE-DIMENSIONS",
    "CADRE-FRANKSCHMI",
    "CADRE-REDPOINPOL",
    "CADRE-SINLATITUD",
    "CADRE-FOCOHYBRID",
    "DATE-DES-DONNEES",
    "DATX-DES-DONNEES",
}


class NativeFAError(RuntimeError):
    """Base error for native FA backend failures."""


class UnsupportedFAEncodingError(NativeFAError):
    """Raised for FA encodings outside the native backend scope."""


@dataclass(frozen=True)
class NativeFAGeometry:
    """Geometry information needed by the public faxarray reader.

    For LAM and regular lon/lat grids this is a 2D ``(ny, nx)`` rectangle.
    For global reduced Gauss the public surface keeps a 2D shape ``(1, N)``
    holding the flat unstructured points; ``projection`` carries the
    ``lat_number``, ``lon_number_by_lat`` and stretching/rotation params.
    """

    name: str
    shape: Tuple[int, int]
    lons: np.ndarray
    lats: np.ndarray
    projection: Optional[Dict[str, object]] = None

    @property
    def is_global_gauss(self) -> bool:
        return self.name in ("reduced_gauss", "rotated_reduced_gauss")


@dataclass(frozen=True)
class NativeFAHeader:
    """Decoded subset of the FA header articles."""

    ktronc: int
    knlati: int
    knxlon: int
    nlevels: int
    ktyptr: int
    red_points: np.ndarray
    sinlat: np.ndarray
    franchschmi: np.ndarray
    hybrid: np.ndarray

    # Backwards-compatible aliases.
    @property
    def ny(self) -> int:
        return self.knlati

    @property
    def nx(self) -> int:
        return self.knxlon

    @property
    def grid_size(self) -> int:
        return self.knlati * self.knxlon


@dataclass(frozen=True)
class NativeFAValidity:
    """Validity information decoded from DATE-DES-DONNEES / DATX-DES-DONNEES."""

    base_time: Optional[np.datetime64]
    valid_time: Optional[np.datetime64]
    lead_time: Optional[np.timedelta64]
    cumulative_duration: Optional[np.timedelta64] = None
    process_type: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "valid_time": self.valid_time,
            "base_time": self.base_time,
            "lead_time": self.lead_time,
            "cumulative_duration": self.cumulative_duration,
            "process_type": self.process_type,
        }


@dataclass(frozen=True)
class NativeFAVertical:
    """Hybrid vertical coordinate (Ai, Bi) and reference pressure."""

    reference_pressure: float
    a_coefficients: np.ndarray
    b_coefficients: np.ndarray

    @property
    def n_levels(self) -> int:
        return max(self.a_coefficients.size, 1) - 1

    def half_level_pressures(self, surface_pressure: np.ndarray) -> np.ndarray:
        """Compute the half-level pressures Ai + Bi * Psurf."""

        a = self.a_coefficients[:, None] * self.reference_pressure
        b = self.b_coefficients[:, None]
        return a + b * surface_pressure[None, :]


@dataclass(frozen=True)
class NativeFASpectralLayoutLAM:
    """Decoded ALADIN spectral coefficient layout from the FA frame."""

    no_zpar: np.ndarray
    by_meridian: Tuple[Tuple[int, int, int], ...]
    coeff_count: int

    def to_layout(self) -> LamSpectralLayout:
        return LamSpectralLayout(by_meridian=self.by_meridian, coeff_count=self.coeff_count)


@dataclass(frozen=True)
class NativeFAFieldEncoding:
    """Decoded field encoding metadata for one FA article."""

    name: str
    exists: bool
    ftype: str
    spectral: bool = False
    kngrib: int = 0
    kngrib_label: str = ""
    knbits: int = 0
    kstron: int = 0
    kpuila: int = 0

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "exists": self.exists,
            "ftype": self.ftype,
        }
        if self.exists and self.ftype == "H2D":
            out.update(
                {
                    "spectral": self.spectral,
                    "KNGRIB": self.kngrib,
                    "KNGRIB_label": self.kngrib_label,
                    "KNBITS": self.knbits,
                    "KSTRON": self.kstron,
                    "KPUILA": self.kpuila,
                }
            )
        return out


@dataclass(frozen=True)
class NativeFAFieldDescriptor:
    """Per-field metadata snapshot, EPYGRAM-style.

    Combines the file-level encoding info (``KNGRIB``, spectrality, packing
    bits) with name-derived information (level kind / index) and
    catalog metadata (``long_name``, ``units``, ``standard_name``,
    ``comment``) sourced from :mod:`faxarray.fa_metadata`.

    For model-level fields the matching hybrid ``A``/``B`` coefficients
    from ``CADRE-FOCOHYBRID`` are also surfaced via
    :attr:`a_coefficient` / :attr:`b_coefficient`. For pressure-level
    fields the value of the level in Pa is surfaced via
    :attr:`pressure_pa`.
    """

    name: str
    encoding: NativeFAFieldEncoding
    base_name: Optional[str] = None
    long_name: Optional[str] = None
    units: Optional[str] = None
    standard_name: Optional[str] = None
    comment: Optional[str] = None
    level_type: Optional[str] = None
    level_index: Optional[int] = None
    pressure_pa: Optional[int] = None
    a_coefficient: Optional[float] = None
    b_coefficient: Optional[float] = None
    fid: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "fid": dict(self.fid),
            "base_name": self.base_name,
            "long_name": self.long_name,
            "units": self.units,
            "standard_name": self.standard_name,
            "comment": self.comment,
            "level_type": self.level_type,
            "level_index": self.level_index,
            "pressure_pa": self.pressure_pa,
            "a_coefficient": self.a_coefficient,
            "b_coefficient": self.b_coefficient,
            "encoding": self.encoding.as_dict(),
        }


_MODEL_LEVEL_PATTERN = re.compile(r"^S(\d{3})(.+)$")
_PRESSURE_LEVEL_PATTERN = re.compile(r"^P(\d{5})(.+)$")
_SURFACE_PREFIXES = ("SURF", "CLS", "PROFTEMPERATURE", "MSL")


_KNGRIB_LABELS = {
    -2: "raw float32",
    -1: "raw float64",
    0: "raw float64",
    1: "legacy GRIB_MF",
    2: "legacy GRIB_MF (extended)",
    3: "legacy GRIB_MF (extra)",
    4: "legacy GRIB_MF (rich)",
    100: "GRIB_API simple packing",
    101: "GRIB_API spectral complex packing",
    102: "GRIB_API simple packing (CCSDS-ish)",
    103: "GRIB_API spectral complex packing (default)",
    104: "GRIB_API simple packing (alt)",
}


class NativeFAResource:
    """Read FA files without EPYGRAM."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.lfi = LFIFile(filepath)
        self._fields: Optional[List[str]] = None
        self._misc_fields: Optional[List[str]] = None
        self._header: Optional[NativeFAHeader] = None
        self._geometry: Optional[NativeFAGeometry] = None
        self._spectral_layout_lam: Optional[NativeFASpectralLayoutLAM] = None
        self._spectral_layout_global: Optional[GaussSpectralLayout] = None
        self._gauss_quadrature: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._validity: Optional[NativeFAValidity] = None
        self._vertical: Optional[NativeFAVertical] = None

    def close(self) -> None:
        """Mirror the EPYGRAM resource API."""

    @property
    def fields(self) -> List[str]:
        if self._fields is None:
            self._fields = self.lfi.list_fa_fields()
        return self._fields

    def listfields(self) -> List[str]:
        return list(self.fields)

    def list_h2d_fields(self) -> List[str]:
        """Return fields whose encoding header reports H2D."""

        return [name for name in self.fields if self.fieldencoding_object(name).ftype == "H2D"]

    def list_misc_fields(self) -> List[str]:
        """Return fields not classified as H2D (header-like, scalar, etc.)."""

        if self._misc_fields is None:
            self._misc_fields = [
                name for name in self.fields if self.fieldencoding_object(name).ftype != "H2D"
            ]
        return list(self._misc_fields)

    @property
    def header(self) -> NativeFAHeader:
        if self._header is None:
            self._header = self._read_header()
        return self._header

    @property
    def geometry(self) -> NativeFAGeometry:
        if self._geometry is None:
            self._geometry = self._build_geometry()
        return self._geometry

    @property
    def vertical(self) -> NativeFAVertical:
        if self._vertical is None:
            self._vertical = self._read_vertical()
        return self._vertical

    @property
    def validity(self) -> NativeFAValidity:
        if self._validity is None:
            self._validity = self._read_validity()
        return self._validity

    def metadata_summary(self) -> Dict[str, object]:
        """Return a JSON-friendly snapshot of the FA file metadata."""

        header = self.header
        geometry = self.geometry
        vertical = self.vertical
        validity = self.validity
        return {
            "filepath": self.filepath,
            "geometry": {
                "name": geometry.name,
                "shape": list(geometry.shape),
                "is_global_gauss": geometry.is_global_gauss,
                "projection": geometry.projection,
            },
            "header": {
                "ktronc": header.ktronc,
                "knlati": header.knlati,
                "knxlon": header.knxlon,
                "nlevels": header.nlevels,
                "ktyptr": header.ktyptr,
            },
            "vertical": {
                "reference_pressure": vertical.reference_pressure,
                "n_levels": vertical.n_levels,
                "a_coefficients": vertical.a_coefficients.tolist(),
                "b_coefficients": vertical.b_coefficients.tolist(),
            },
            "validity": {
                key: (str(value) if value is not None else None)
                for key, value in validity.as_dict().items()
            },
            "n_fields": len(self.fields),
            "n_misc_fields": len(self.list_misc_fields()),
        }

    def _article_as_int64(self, name: str) -> np.ndarray:
        return np.frombuffer(self.lfi.read_article_bytes(name), dtype=self.lfi.word_dtype).copy()

    def _article_as_float64(self, name: str) -> np.ndarray:
        return np.frombuffer(self.lfi.read_article_bytes(name), dtype=self.lfi.float64_dtype).copy()

    def _read_header(self) -> NativeFAHeader:
        dims = self._article_as_int64("CADRE-DIMENSIONS")
        if dims.size < 5:
            raise NativeFAError("FA header article CADRE-DIMENSIONS is too short")
        try:
            franchschmi = self._article_as_float64("CADRE-FRANKSCHMI")
        except KeyError:
            franchschmi = np.zeros(4, dtype=np.float64)
        return NativeFAHeader(
            ktronc=int(dims[0]),
            knlati=int(dims[1]),
            knxlon=int(dims[2]),
            nlevels=int(dims[3]),
            ktyptr=int(dims[4]),
            red_points=self._article_as_int64("CADRE-REDPOINPOL"),
            sinlat=self._article_as_float64("CADRE-SINLATITUD"),
            franchschmi=franchschmi,
            hybrid=self._article_as_float64("CADRE-FOCOHYBRID"),
        )

    def _build_geometry(self) -> NativeFAGeometry:
        header = self.header
        if header.ktyptr in (1, 2):
            return self._global_gauss_geometry(header)

        sinlat = header.sinlat
        if sinlat.size < 12:
            raise NativeFAError("FA geometry article CADRE-SINLATITUD is too short")

        if int(sinlat[1]) == -9:
            return self._regular_lonlat_geometry(header)
        return self._projected_geometry(header)

    def _regular_lonlat_geometry(self, header: NativeFAHeader) -> NativeFAGeometry:
        sinlat = header.sinlat
        nx = header.nx
        ny = header.ny
        lon0 = math.degrees(float(sinlat[4]))
        lat0 = math.degrees(float(sinlat[5]))
        dx = math.degrees(float(sinlat[6]))
        dy = math.degrees(float(sinlat[7]))
        x0 = (nx - 1) / 2.0
        y0 = (ny - 1) / 2.0
        x = lon0 + (np.arange(nx, dtype=np.float64) - x0) * dx
        y = lat0 + (np.arange(ny, dtype=np.float64) - y0) * dy
        lons, lats = np.meshgrid(x, y)
        return NativeFAGeometry("regular_lonlat", (ny, nx), lons, lats)

    def _projected_geometry(self, header: NativeFAHeader) -> NativeFAGeometry:
        sinlat = header.sinlat
        red = header.red_points
        nx = header.nx
        ny = header.ny
        ref_lon = math.degrees(float(sinlat[2]))
        ref_lat = math.degrees(float(sinlat[3]))
        input_lon = math.degrees(float(sinlat[4]))
        input_lat = math.degrees(float(sinlat[5]))
        dx = float(sinlat[6])
        dy = float(sinlat[7])
        x_ci = int(red[3] - red[2] + 1) if red.size > 5 else nx
        y_ci = int(red[5] - red[4] + 1) if red.size > 5 else ny
        x_input = (x_ci - 1) / 2.0
        y_input = (y_ci - 1) / 2.0

        proj_value = float(sinlat[1])
        if abs(proj_value) <= 1e-12:
            name = "mercator"
        elif 1.0 - abs(proj_value) <= 1e-12:
            name = "polar_stereographic"
        else:
            name = "lambert"

        projection = {
            "reference_lon": ref_lon,
            "reference_lat": ref_lat,
            "input_lon": input_lon,
            "input_lat": input_lat,
            "x_resolution": dx,
            "y_resolution": dy,
        }

        try:
            from pyproj import CRS, Transformer
        except Exception:
            x = input_lon + (np.arange(nx, dtype=np.float64) - x_input) * dx
            y = input_lat + (np.arange(ny, dtype=np.float64) - y_input) * dy
            lons, lats = np.meshgrid(x, y)
            return NativeFAGeometry(name, (ny, nx), lons, lats, projection)

        if name == "mercator":
            crs = CRS.from_proj4(f"+proj=merc +lat_ts={ref_lat} +lon_0={ref_lon} +R=6371229")
        elif name == "polar_stereographic":
            lat_0 = 90.0 if proj_value > 0 else -90.0
            crs = CRS.from_proj4(
                f"+proj=stere +lat_0={lat_0} +lat_ts={ref_lat} +lon_0={ref_lon} +R=6371229"
            )
        else:
            crs = CRS.from_proj4(
                f"+proj=lcc +lat_1={ref_lat} +lat_0={ref_lat} +lon_0={ref_lon} +R=6371229"
            )

        to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_x, center_y = to_xy.transform(input_lon, input_lat)
        xs = center_x + (np.arange(nx, dtype=np.float64) - x_input) * dx
        ys = center_y + (np.arange(ny, dtype=np.float64) - y_input) * dy
        grid_x, grid_y = np.meshgrid(xs, ys)
        lons, lats = to_ll.transform(grid_x, grid_y)
        return NativeFAGeometry(name, (ny, nx), np.asarray(lons), np.asarray(lats), projection)

    def _global_gauss_geometry(self, header: NativeFAHeader) -> NativeFAGeometry:
        knlati = header.knlati
        inpahe = (1 + knlati) // 2

        sinlat_nh = np.asarray(header.sinlat[:inpahe], dtype=np.float64)
        knlopa_nh = np.asarray(header.red_points[:inpahe], dtype=np.int64)

        if sinlat_nh.size != inpahe or knlopa_nh.size != inpahe:
            raise NativeFAError(
                "FA Gauss header articles are inconsistent with declared latitude count"
            )

        # Mirror NH (north pole → equator) into SH (equator → south pole).
        if knlati % 2 == 0:
            knlopa = np.concatenate([knlopa_nh, knlopa_nh[::-1]])
            sinlat = np.concatenate([sinlat_nh, -sinlat_nh[::-1]])
        else:
            knlopa = np.concatenate([knlopa_nh, knlopa_nh[-2::-1]])
            sinlat = np.concatenate([sinlat_nh, -sinlat_nh[-2::-1]])

        lats_per_row = np.degrees(np.arcsin(np.clip(sinlat, -1.0, 1.0)))

        total = int(knlopa.sum())
        lons_flat = np.empty(total, dtype=np.float64)
        lats_flat = np.empty(total, dtype=np.float64)
        offsets = np.zeros(knlopa.size + 1, dtype=np.int64)
        np.cumsum(knlopa, out=offsets[1:])

        for j in range(knlopa.size):
            n = int(knlopa[j])
            lons_flat[offsets[j] : offsets[j + 1]] = np.arange(n, dtype=np.float64) * (360.0 / n)
            lats_flat[offsets[j] : offsets[j + 1]] = lats_per_row[j]

        # Rotated pole transformation (cos_pole_lon, sin_pole_lon, sin_pole_lat).
        if header.ktyptr == 2:
            sslapo, sclopo, sslopo, _ = header.franchschmi[:4]
            lons_flat, lats_flat = _apply_pole_rotation(
                lons_flat, lats_flat, float(sslapo), float(sclopo), float(sslopo)
            )

        lons_2d = lons_flat.reshape(1, -1)
        lats_2d = lats_flat.reshape(1, -1)

        knozpa_nh = (
            np.asarray(header.red_points[inpahe : 2 * inpahe], dtype=np.int64)
            if header.red_points.size >= 2 * inpahe
            else np.zeros(0, dtype=np.int64)
        )

        projection = {
            "lat_number": int(knlati),
            "lon_number_by_lat": knlopa.tolist(),
            "max_lon_number": int(header.knxlon),
            "lat_values": lats_per_row.tolist(),
            "sin_lat_values": sinlat.tolist(),
            "max_zonal_wavenumber_by_lat_nh": knozpa_nh.tolist(),
            "stretching_factor": float(header.franchschmi[3])
            if header.franchschmi.size >= 4
            else 1.0,
            "pole_sin_lat": float(header.franchschmi[0])
            if header.franchschmi.size >= 4
            else 1.0,
            "pole_cos_lon": float(header.franchschmi[1])
            if header.franchschmi.size >= 4
            else 1.0,
            "pole_sin_lon": float(header.franchschmi[2])
            if header.franchschmi.size >= 4
            else 0.0,
            "total_points": total,
        }
        name = "rotated_reduced_gauss" if header.ktyptr == 2 else "reduced_gauss"
        return NativeFAGeometry(name, (1, total), lons_2d, lats_2d, projection)

    def _read_vertical(self) -> NativeFAVertical:
        header = self.header
        levels = max(header.nlevels, 0)
        hybrid = header.hybrid
        if hybrid.size < 1:
            return NativeFAVertical(
                reference_pressure=101325.0,
                a_coefficients=np.zeros(0, dtype=np.float64),
                b_coefficients=np.zeros(0, dtype=np.float64),
            )
        prefer = float(hybrid[0])
        n = levels + 1
        if hybrid.size < 1 + 2 * n:
            n = max(0, (hybrid.size - 1) // 2)
        a = np.asarray(hybrid[1 : 1 + n], dtype=np.float64)
        b = np.asarray(hybrid[1 + n : 1 + 2 * n], dtype=np.float64)
        return NativeFAVertical(
            reference_pressure=prefer,
            a_coefficients=a,
            b_coefficients=b,
        )

    def fieldencoding(self, name: str) -> Dict[str, object]:
        """Return a dict describing the encoding of ``name`` (legacy API)."""

        return self.fieldencoding_object(name).as_dict()

    def fieldencoding_object(self, name: str) -> NativeFAFieldEncoding:
        """Same as :meth:`fieldencoding` but typed."""

        if name in _HEADER_ARTICLES:
            return NativeFAFieldEncoding(name=name, exists=True, ftype="Misc")
        try:
            words = self.lfi.read_article_words(name, max_words=5)
        except KeyError:
            return NativeFAFieldEncoding(name=name, exists=False, ftype="?")
        if len(words) < 2:
            return NativeFAFieldEncoding(name=name, exists=True, ftype="Misc")

        kngr = int(words[0])
        spectral_flag = int(words[1])
        if spectral_flag not in (0, 1):
            return NativeFAFieldEncoding(name=name, exists=True, ftype="Misc")
        if not (-2 <= kngr <= 4 or kngr >= 100):
            return NativeFAFieldEncoding(name=name, exists=True, ftype="Misc")

        if kngr in (-2, -1, 0):
            knbits = 0
            kstron = 0
            kpuila = 0
        else:
            knbits = int(words[2]) if len(words) > 2 else 0
            kstron = int(words[3]) if len(words) > 3 else 0
            kpuila = int(words[4]) if len(words) > 4 else 0

        return NativeFAFieldEncoding(
            name=name,
            exists=True,
            ftype="H2D",
            spectral=spectral_flag == 1,
            kngrib=kngr,
            kngrib_label=_KNGRIB_LABELS.get(kngr, f"unknown KNGRIB={kngr}"),
            knbits=knbits,
            kstron=kstron,
            kpuila=kpuila,
        )

    def field_descriptor(self, name: str) -> NativeFAFieldDescriptor:
        """Return rich per-field metadata for ``name`` (EPYGRAM-style).

        Combines the FA encoding header, the name-derived level info
        (model level S###, pressure level P#####, or surface), the
        ``faxarray.fa_metadata`` catalog (``long_name``, ``units``), and
        the matching hybrid ``A`` / ``B`` coefficients for model-level
        fields.
        """

        encoding = self.fieldencoding_object(name)
        from ..fa_metadata import get_metadata

        catalog = get_metadata(name) or {}
        # Strip pure-data parts of the catalog dict (it sometimes has
        # ``description``-only entries with empty units).
        long_name = catalog.get("long_name") or None
        units = catalog.get("units") or None
        standard_name = catalog.get("standard_name") or None
        comment = catalog.get("description") or catalog.get("comment") or None

        base_name = None
        level_type: Optional[str] = None
        level_index: Optional[int] = None
        pressure_pa: Optional[int] = None
        a_coefficient: Optional[float] = None
        b_coefficient: Optional[float] = None

        match = _MODEL_LEVEL_PATTERN.match(name)
        if match:
            level_index = int(match.group(1))
            base_name = match.group(2)
            level_type = "model"
            vertical = self.vertical
            # Hybrid arrays are 0-indexed and have N+1 entries (half levels).
            n_half = vertical.a_coefficients.size
            if n_half > 0:
                # The native model-level numbering goes 1..N from the model top
                # down to the surface. EPYGRAM stores half-level coefficients,
                # so for level k we report the half-level immediately below it.
                half_idx = min(max(level_index, 1), n_half - 1)
                a_coefficient = float(vertical.a_coefficients[half_idx])
                b_coefficient = float(vertical.b_coefficients[half_idx])
        else:
            match = _PRESSURE_LEVEL_PATTERN.match(name)
            if match:
                encoded = int(match.group(1))
                base_name = match.group(2)
                level_type = "pressure"
                pressure_pa = 100000 if encoded == 0 else encoded
            elif any(name.startswith(prefix) for prefix in _SURFACE_PREFIXES):
                level_type = "surface"
            elif encoding.ftype == "Misc":
                level_type = "header"
            else:
                level_type = "other"

        # If we have a level prefix but the catalog returned nothing, try
        # the un-prefixed base name (e.g. TEMPERATURE for S001TEMPERATURE).
        if base_name and not (long_name or units):
            base_catalog = get_metadata(base_name) or {}
            long_name = long_name or base_catalog.get("long_name") or None
            units = units or base_catalog.get("units") or None
            standard_name = standard_name or base_catalog.get("standard_name") or None
            comment = comment or base_catalog.get("description") or base_catalog.get("comment") or None

        return NativeFAFieldDescriptor(
            name=name,
            encoding=encoding,
            base_name=base_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            comment=comment,
            level_type=level_type,
            level_index=level_index,
            pressure_pa=pressure_pa,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient,
            fid={"FA": name, "shortName": name},
        )

    def get_validity(self) -> Dict[str, object]:
        """Backwards-compatible accessor returning a plain dict."""

        return self.validity.as_dict()

    def _read_validity(self) -> NativeFAValidity:
        try:
            date = self._article_as_int64("DATE-DES-DONNEES")
        except KeyError:
            return NativeFAValidity(None, None, None)

        if date.size < 7 or int(date[0]) <= 0:
            return NativeFAValidity(None, None, None)

        try:
            base = datetime(int(date[0]), int(date[1]), int(date[2]), int(date[3]), int(date[4]))
        except ValueError:
            return NativeFAValidity(None, None, None)

        seconds = 0
        cumulative_seconds: Optional[int] = None
        try:
            datx = self._article_as_int64("DATX-DES-DONNEES")
            if datx.size > 3 and int(datx[3]) > 0:
                seconds = int(datx[3])
            if datx.size > 4 and int(datx[4]) > 0:
                cumulative_seconds = int(datx[4])
        except KeyError:
            pass

        if seconds == 0:
            unit = int(date[5])
            term = int(date[6])
            if unit == 1:
                seconds = term * 3600
            elif unit == 2:
                seconds = term * 86400

        process_type: Optional[int]
        if date.size > 8:
            process_type = int(date[8])
        else:
            process_type = None

        lead = timedelta(seconds=seconds)
        return NativeFAValidity(
            base_time=np.datetime64(base),
            valid_time=np.datetime64(base + lead),
            lead_time=np.timedelta64(int(seconds), "s"),
            cumulative_duration=np.timedelta64(int(cumulative_seconds), "s")
            if cumulative_seconds is not None
            else None,
            process_type=process_type,
        )

    # ---------- Field reading ----------

    def readfield(self, name: str, convert_spectral: bool = True) -> np.ndarray:
        encoding = self.fieldencoding_object(name)
        if not encoding.exists:
            raise KeyError(f"Field is unknown in file: {name}")
        if encoding.ftype != "H2D":
            raise UnsupportedFAEncodingError(f"Field is not a horizontal data field: {name}")

        if encoding.spectral:
            coefficients = self._read_spectral_coefficients(name, encoding)
            if convert_spectral:
                return self._spectral_to_gridpoint(coefficients)
            return coefficients

        kngr = encoding.kngrib
        if kngr in (-1, 0):
            return self._read_raw64(name, spectral=False)
        if kngr == -2:
            return self._read_raw32(name, spectral=False)
        return self._read_packed(name, kngr, spectral=False)

    def read_misc_field_bytes(self, name: str) -> bytes:
        """Return the raw payload bytes of a non-H2D / Misc article."""

        encoding = self.fieldencoding_object(name)
        if not encoding.exists:
            raise KeyError(f"Field is unknown in file: {name}")
        if encoding.ftype == "H2D":
            raise UnsupportedFAEncodingError(
                f"{name} is an H2D field; use readfield() instead"
            )
        return self.lfi.read_article_bytes(name)

    def read_misc_field_words(self, name: str) -> np.ndarray:
        """Return the int64 word array for a Misc / header-like article."""

        return np.frombuffer(self.read_misc_field_bytes(name), dtype=self.lfi.word_dtype).copy()

    def _expected_size(self, spectral: bool) -> int:
        if spectral:
            raise UnsupportedFAEncodingError("spectral coefficient sizing is not implemented yet")
        if self.geometry.is_global_gauss:
            return int(self.geometry.projection["total_points"])
        return self.header.grid_size

    def _reshape(self, values: np.ndarray, spectral: bool) -> np.ndarray:
        if spectral:
            return values
        shape = self.geometry.shape
        expected = shape[0] * shape[1]
        if values.size < expected:
            raise NativeFAError(f"field contains {values.size} values, expected {expected}")
        return values[:expected].reshape(shape)

    def _read_raw64(self, name: str, spectral: bool) -> np.ndarray:
        data = self.lfi.read_article_bytes(name)
        values = np.frombuffer(data[16:], dtype=self.lfi.float64_dtype).astype(np.float64, copy=False)
        return self._reshape(values.copy(), spectral)

    def _read_raw32(self, name: str, spectral: bool) -> np.ndarray:
        data = self.lfi.read_article_bytes(name)
        values = np.frombuffer(data[16:], dtype=self.lfi.float32_dtype).astype(np.float64, copy=False)
        return self._reshape(values.copy(), spectral)

    def _read_packed(self, name: str, kngr: int, spectral: bool) -> np.ndarray:
        if kngr in (1, 2):
            words = self._article_as_int64(name)
            try:
                values = get_legacy_codec().decode(words, self._expected_size(spectral=False))
            except FAGribMFError as exc:
                raise UnsupportedFAEncodingError(str(exc)) from exc
            return self._reshape(values, spectral=False)

        data = self.lfi.read_article_bytes(name)
        grib_at = data.find(b"GRIB")
        if grib_at >= 0:
            try:
                values = decode_grib_values(data[grib_at:])
            except EccodesCError as exc:
                raise UnsupportedFAEncodingError(
                    f"{name} is packed and needs system ecCodes: {exc}"
                ) from exc
            return self._reshape(values, spectral=False)

        raise UnsupportedFAEncodingError(f"{name} uses unsupported packed FA encoding KNGRIB={kngr}")

    # ---------- Spectral handling ----------

    def _get_spectral_layout_lam(self) -> NativeFASpectralLayoutLAM:
        if self._spectral_layout_lam is not None:
            return self._spectral_layout_lam

        header = self.header
        if header.ktyptr >= 0:
            raise UnsupportedFAEncodingError(
                "ALADIN bi-Fourier layout requested for a non-LAM file"
            )

        red = header.red_points
        if red.size < 12:
            raise NativeFAError("FA spectral layout article CADRE-REDPOINPOL is too short")

        no_zpar = np.asarray(red[8:], dtype=np.int64)
        pair_values = no_zpar[2:]
        if pair_values.size < 2 or pair_values.size % 2:
            raise NativeFAError("FA spectral layout has an invalid wave-number table")

        by_meridian: List[Tuple[int, int, int]] = []
        coeff_count = 0
        for index in range(0, pair_values.size, 2):
            start = int(pair_values[index])
            end = int(pair_values[index + 1])
            if start <= 0 or end < start:
                break
            length = end - start + 1
            if length % 4:
                raise NativeFAError("ALADIN spectral coefficient block is not a multiple of 4")
            by_meridian.append((start, end, length // 4 - 1))
            coeff_count = max(coeff_count, end)

        expected_rows = header.ktronc + 1
        if len(by_meridian) < expected_rows:
            raise NativeFAError(
                f"FA spectral layout has {len(by_meridian)} rows, expected at least {expected_rows}"
            )

        self._spectral_layout_lam = NativeFASpectralLayoutLAM(
            no_zpar=no_zpar,
            by_meridian=tuple(by_meridian[:expected_rows]),
            coeff_count=coeff_count,
        )
        return self._spectral_layout_lam

    def _get_spectral_layout_global(self) -> GaussSpectralLayout:
        if self._spectral_layout_global is None:
            self._spectral_layout_global = make_gauss_layout(self.header.ktronc)
        return self._spectral_layout_global

    def _read_spectral_coefficients(
        self, name: str, encoding: NativeFAFieldEncoding
    ) -> np.ndarray:
        kngr = encoding.kngrib
        if kngr in (-1, 0):
            return self._read_raw64(name, spectral=True)
        if kngr == -2:
            return self._read_raw32(name, spectral=True)
        if kngr in (1, 2):
            return self._read_legacy_packed_spectral(name, kngr, encoding)

        data = self.lfi.read_article_bytes(name)
        grib_at = data.find(b"GRIB")
        if grib_at >= 0:
            try:
                values, _meta = decode_grib_spectral_message(data[grib_at:])
            except EccodesCError as exc:
                raise UnsupportedFAEncodingError(
                    f"{name} is a GRIB_API-packed spectral field that ecCodes could not decode: {exc}"
                ) from exc
            return values
        raise UnsupportedFAEncodingError(f"{name} uses unsupported spectral FA encoding KNGRIB={kngr}")

    def _read_legacy_packed_spectral(
        self,
        name: str,
        kngr: int,
        encoding: NativeFAFieldEncoding,
    ) -> np.ndarray:
        layout = self._get_spectral_layout_lam()
        kstron = encoding.kstron
        kpuila = encoding.kpuila
        compacted_count, preserved_count = self._spectral_packed_counts(kstron, layout)
        words = self._article_as_int64(name)

        try:
            compacted = get_legacy_codec().decode(words, compacted_count, spectral=True)
        except FAGribMFError as exc:
            raise UnsupportedFAEncodingError(str(exc)) from exc

        full = np.zeros(layout.coeff_count, dtype=np.float64)
        compacted_index = 0
        for meridian in range(1, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            offset = 4 * (kstron + 1 - meridian)
            if offset <= 0:
                offset = 4
            compacted_start = start + offset
            if compacted_start <= end:
                count = end - compacted_start + 1
                full[compacted_start - 1 : end] = compacted[compacted_index : compacted_index + count]
                compacted_index += count

        if compacted_index != compacted_count:
            raise NativeFAError(
                f"{name} decoded {compacted_index} compacted spectral coefficients, expected {compacted_count}"
            )

        payload_offset = 1 + 2 * kngr + 2
        coded_words = len(words) - payload_offset - preserved_count
        if coded_words < 0:
            raise NativeFAError(f"{name} spectral article is too short for preserved coefficients")
        preserved_words = words[payload_offset + coded_words : payload_offset + coded_words + preserved_count]
        preserved = preserved_words.view(self.lfi.float64_dtype).astype(np.float64, copy=False)
        if preserved.size != preserved_count:
            raise NativeFAError(f"{name} has an invalid preserved spectral coefficient block")

        preserved_index = 0
        for meridian in range(0, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            if meridian == 0:
                preserved_end = end
            else:
                preserved_end = start + 4 * (kstron + 1 - meridian) - 1
                if preserved_end <= start:
                    preserved_end = start + 3
                preserved_end = min(preserved_end, end)
            count = preserved_end - start + 1
            if count > 0:
                full[start - 1 : preserved_end] = preserved[preserved_index : preserved_index + count]
                preserved_index += count

        if preserved_index != preserved_count:
            raise NativeFAError(
                f"{name} restored {preserved_index} preserved spectral coefficients, expected {preserved_count}"
            )

        return self._undo_laplacian_packing(full, kstron, kpuila, layout)

    def _spectral_packed_counts(
        self,
        kstron: int,
        layout: NativeFASpectralLayoutLAM,
    ) -> Tuple[int, int]:
        preserved_count = layout.by_meridian[0][1] - layout.by_meridian[0][0] + 1
        compacted_count = 0
        for meridian in range(1, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            offset = 4 * (kstron + 1 - meridian)
            if offset <= 0:
                offset = 4
            compacted_start = start + offset
            if compacted_start <= end:
                compacted_count += end - compacted_start + 1

            preserved_end = start + offset - 1
            if preserved_end <= start:
                preserved_end = start + 3
            preserved_end = min(preserved_end, end)
            preserved_count += preserved_end - start + 1

        return compacted_count, preserved_count

    def _undo_laplacian_packing(
        self,
        coefficients: np.ndarray,
        kstron: int,
        kpuila: int,
        layout: NativeFASpectralLayoutLAM,
    ) -> np.ndarray:
        return _laplacian_packing(coefficients, kstron, kpuila, layout, self.header.ktronc, inverse=True)

    def _redo_laplacian_packing(
        self,
        coefficients: np.ndarray,
        kstron: int,
        kpuila: int,
        layout: NativeFASpectralLayoutLAM,
    ) -> np.ndarray:
        return _laplacian_packing(coefficients, kstron, kpuila, layout, self.header.ktronc, inverse=False)

    def _spectral_to_gridpoint(self, coefficients: np.ndarray) -> np.ndarray:
        if self.geometry.is_global_gauss:
            return self._global_spectral_to_gridpoint(coefficients)
        layout = self._get_spectral_layout_lam()
        ny, nx = self.geometry.shape
        return lam_sp2gp(coefficients, layout.to_layout(), ny, nx)

    def _global_spectral_to_gridpoint(self, coefficients: np.ndarray) -> np.ndarray:
        layout = self._get_spectral_layout_global()
        proj = self.geometry.projection
        if proj is None:
            raise NativeFAError("Gauss geometry projection metadata is missing")
        sin_lats = np.asarray(proj["sin_lat_values"], dtype=np.float64)
        lon_number_by_lat = proj["lon_number_by_lat"]
        expected = layout.total_real_coeffs
        if coefficients.size < expected:
            raise NativeFAError(
                f"global spectral field has {coefficients.size} reals, expected {expected}"
            )
        return gauss_sp2gp(coefficients[:expected], layout, sin_lats, lon_number_by_lat).reshape(
            self.geometry.shape
        )

    def _gridpoint_to_spectral_lam(self, values: np.ndarray) -> np.ndarray:
        layout = self._get_spectral_layout_lam()
        return lam_gp2sp(values, layout.to_layout())

    # ---------- Field writing ----------

    def write_template(
        self,
        output: str,
        fields: Mapping[str, np.ndarray],
        overwrite: bool = False,
    ) -> None:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        if output_path.exists():
            output_path.unlink()
        shutil.copy2(self.filepath, output_path)

        writable = NativeFAResource(str(output_path))
        for field_name, data in fields.items():
            writable._replace_field(field_name, np.asarray(data))

    def _replace_field(self, name: str, values: np.ndarray) -> None:
        encoding = self.fieldencoding_object(name)
        if encoding.ftype != "H2D":
            raise UnsupportedFAEncodingError(f"Cannot write non-H2D field: {name}")

        if encoding.spectral:
            self._replace_spectral_field(name, values, encoding)
            return

        kngr = encoding.kngrib
        expected_shape = self.geometry.shape
        squeezed = np.squeeze(values)
        if squeezed.shape != expected_shape:
            if self.geometry.is_global_gauss and squeezed.shape == (expected_shape[1],):
                squeezed = squeezed.reshape(expected_shape)
            else:
                raise ValueError(f"{name} has shape {squeezed.shape}, expected {expected_shape}")

        original = self.lfi.read_article_bytes(name)
        if kngr in (-2, -1, 0):
            prefix = original[:16]
            if kngr == -2:
                payload = np.asarray(squeezed, dtype=self.lfi.float32_dtype).tobytes()
            else:
                payload = np.asarray(squeezed, dtype=self.lfi.float64_dtype).tobytes()
            replacement = prefix + payload
        elif kngr in (1, 2):
            words = np.frombuffer(original, dtype=self.lfi.word_dtype).copy()
            try:
                encoded = get_legacy_codec().encode(words, squeezed, spectral=False)
            except FAGribMFError as exc:
                raise UnsupportedFAEncodingError(str(exc)) from exc
            replacement = encoded.astype(self.lfi.word_dtype, copy=False).tobytes()
        elif kngr >= 100:
            replacement = self._encode_grib_api_gridpoint(name, original, squeezed)
        else:
            raise UnsupportedFAEncodingError(f"Cannot write packed field {name} with KNGRIB={kngr}")

        replacement = self._pad_replacement(replacement, len(original), name)
        self.lfi.write_article_bytes(name, replacement)

    def _replace_spectral_field(
        self,
        name: str,
        values: np.ndarray,
        encoding: NativeFAFieldEncoding,
    ) -> None:
        kngr = encoding.kngrib
        original = self.lfi.read_article_bytes(name)
        squeezed = np.squeeze(values)
        is_global = self.geometry.is_global_gauss

        if is_global:
            # User must pass spectral coefficients directly because we do not
            # have a bit-perfect global gp->sp transform on Mac.
            if squeezed.ndim == 2 and squeezed.shape[0] == 1:
                squeezed = squeezed[0]
            global_layout = self._get_spectral_layout_global()
            if squeezed.size != global_layout.total_real_coeffs:
                raise UnsupportedFAEncodingError(
                    f"global spectral write of {name} requires a coefficient array of "
                    f"length {global_layout.total_real_coeffs}, got shape {values.shape}. "
                    "Use faxarray.backends.spectral.gauss_gp2sp() to transform first."
                )
            coefficients = np.ascontiguousarray(squeezed, dtype=np.float64)
        else:
            ny, nx = self.geometry.shape
            layout = self._get_spectral_layout_lam()
            if squeezed.shape == (ny, nx):
                coefficients = lam_gp2sp(squeezed, layout.to_layout())
            elif squeezed.ndim == 1 and squeezed.size == layout.coeff_count:
                coefficients = np.ascontiguousarray(squeezed, dtype=np.float64)
            else:
                raise ValueError(
                    f"spectral write of {name} expects gridpoint shape {(ny, nx)} "
                    f"or coefficient length {layout.coeff_count}, got {squeezed.shape}"
                )

        if kngr in (-1, 0):
            prefix = original[:16]
            payload = np.asarray(coefficients, dtype=self.lfi.float64_dtype).tobytes()
            replacement = prefix + payload
        elif kngr == -2:
            prefix = original[:16]
            payload = np.asarray(coefficients, dtype=self.lfi.float32_dtype).tobytes()
            replacement = prefix + payload
        elif kngr in (1, 2):
            if is_global:
                raise UnsupportedFAEncodingError(
                    f"writing legacy KNGRIB={kngr} spectral fields for global Gauss "
                    f"is not supported (legacy spectral packing is LAM-only)"
                )
            replacement = self._encode_legacy_spectral(name, encoding, coefficients, original)
        elif kngr >= 100:
            replacement = self._encode_grib_api_spectral(name, original, coefficients)
        else:
            raise UnsupportedFAEncodingError(
                f"writing spectral field {name} with KNGRIB={kngr} is not supported"
            )

        replacement = self._pad_replacement(replacement, len(original), name)
        self.lfi.write_article_bytes(name, replacement)

    def _encode_grib_api_spectral(
        self, name: str, original: bytes, coefficients: np.ndarray
    ) -> bytes:
        """Re-encode a GRIB_API-packed spectral article in place.

        The coefficient layout (J/K/M truncation, packing type, bits per
        value) is inherited from the original GRIB message; we only swap
        the coefficient values via ``codes_set_double_array`` and ask
        ecCodes to re-emit the message.
        """

        grib_at = original.find(b"GRIB")
        if grib_at < 0:
            raise UnsupportedFAEncodingError(
                f"{name} has KNGRIB>=100 but no embedded GRIB message"
            )
        try:
            new_grib = encode_grib_values(original[grib_at:], coefficients)
        except EccodesCError as exc:
            raise UnsupportedFAEncodingError(
                f"{name} could not be re-encoded as GRIB_API spectral via ecCodes: {exc}"
            ) from exc
        return original[:grib_at] + new_grib

    def _encode_legacy_spectral(
        self,
        name: str,
        encoding: NativeFAFieldEncoding,
        coefficients: np.ndarray,
        original: bytes,
    ) -> bytes:
        kngr = encoding.kngrib
        kstron = encoding.kstron
        kpuila = encoding.kpuila
        layout = self._get_spectral_layout_lam()

        repacked = self._redo_laplacian_packing(coefficients, kstron, kpuila, layout)

        compacted_count, preserved_count = self._spectral_packed_counts(kstron, layout)
        compacted = np.zeros(compacted_count, dtype=np.float64)
        compacted_index = 0
        for meridian in range(1, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            offset = 4 * (kstron + 1 - meridian)
            if offset <= 0:
                offset = 4
            compacted_start = start + offset
            if compacted_start <= end:
                count = end - compacted_start + 1
                compacted[compacted_index : compacted_index + count] = repacked[
                    compacted_start - 1 : end
                ]
                compacted_index += count

        words = np.frombuffer(original, dtype=self.lfi.word_dtype).copy()
        try:
            encoded_words = get_legacy_codec().encode(words, compacted, spectral=True)
        except FAGribMFError as exc:
            raise UnsupportedFAEncodingError(str(exc)) from exc

        # Restore preserved (un-packed) coefficients at the tail of the article.
        payload_offset = 1 + 2 * kngr + 2
        preserved = np.zeros(preserved_count, dtype=np.float64)
        preserved_index = 0
        for meridian in range(0, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            if meridian == 0:
                preserved_end = end
            else:
                preserved_end = start + 4 * (kstron + 1 - meridian) - 1
                if preserved_end <= start:
                    preserved_end = start + 3
                preserved_end = min(preserved_end, end)
            count = preserved_end - start + 1
            if count > 0:
                preserved[preserved_index : preserved_index + count] = repacked[
                    start - 1 : preserved_end
                ]
                preserved_index += count

        result = encoded_words.astype(self.lfi.word_dtype, copy=False).tobytes()
        # Append the preserved coefficient block at the original tail position.
        preserved_bytes = np.asarray(preserved, dtype=self.lfi.float64_dtype).tobytes()
        head_byte_count = 8 * (
            payload_offset + len(encoded_words) - payload_offset - preserved_count
        )
        if head_byte_count < 8 * payload_offset:
            head_byte_count = 8 * payload_offset
        head = result[: head_byte_count]
        return head + preserved_bytes

    def _encode_grib_api_gridpoint(
        self, name: str, original: bytes, values: np.ndarray
    ) -> bytes:
        grib_at = original.find(b"GRIB")
        if grib_at < 0:
            raise UnsupportedFAEncodingError(
                f"{name} has KNGRIB>=100 but no embedded GRIB message; cannot encode"
            )
        try:
            new_grib = encode_grib_values(original[grib_at:], values)
        except EccodesCError as exc:
            raise UnsupportedFAEncodingError(
                f"{name} could not be re-encoded via ecCodes: {exc}"
            ) from exc
        return original[:grib_at] + new_grib

    def _pad_replacement(self, replacement: bytes, target_length: int, name: str) -> bytes:
        if len(replacement) > target_length:
            raise ValueError(
                f"{name} replacement is {len(replacement)} bytes, exceeds template article "
                f"length of {target_length} bytes"
            )
        if len(replacement) < target_length:
            replacement = replacement + b"\x00" * (target_length - len(replacement))
        return replacement


# ---------- Helpers ----------


def _apply_pole_rotation(
    lons_deg: np.ndarray,
    lats_deg: np.ndarray,
    sin_pole_lat: float,
    cos_pole_lon: float,
    sin_pole_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ARPEGE pole rotation to a flat lon/lat array (degrees)."""

    cos_pole_lat = math.sqrt(max(0.0, 1.0 - sin_pole_lat * sin_pole_lat))
    pole_lon = math.atan2(sin_pole_lon, cos_pole_lon)

    phi = np.radians(lats_deg)
    lam = np.radians(lons_deg)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    sin_phi_rot = sin_pole_lat * sin_phi + cos_pole_lat * cos_phi * cos_lam
    sin_phi_rot = np.clip(sin_phi_rot, -1.0, 1.0)
    phi_rot = np.arcsin(sin_phi_rot)

    cos_phi_rot = np.cos(phi_rot)
    safe = np.where(np.abs(cos_phi_rot) < 1e-12, 1e-12, cos_phi_rot)

    sin_dlam = (cos_phi * sin_lam) / safe
    cos_dlam = (
        (-sin_pole_lat * cos_phi * cos_lam + cos_pole_lat * sin_phi) / safe
    )
    dlam = np.arctan2(sin_dlam, cos_dlam)
    lam_rot = pole_lon + dlam

    return ((np.degrees(lam_rot) + 180.0) % 360.0 - 180.0), np.degrees(phi_rot)


def _laplacian_packing(
    coefficients: np.ndarray,
    kstron: int,
    kpuila: int,
    layout: NativeFASpectralLayoutLAM,
    ktronc: int,
    inverse: bool,
) -> np.ndarray:
    """Apply or undo the ALADIN Laplacian re-scaling on packed coefficients."""

    if kpuila == 0:
        return np.array(coefficients, dtype=np.float64, copy=True)

    result = np.array(coefficients, dtype=np.float64, copy=True)
    power = abs(kpuila)
    flip = kpuila > 0  # KPUILA>0 means coefficients were divided -> multiply on read

    for meridian in range(1, ktronc + 1):
        start, end, _ = layout.by_meridian[meridian]
        first = max(start + 4 * (1 + kstron - meridian), start + 4)
        if first > end:
            continue
        positions = np.arange(first, end + 1, dtype=np.int64)
        zonal = (positions - start) // 4
        laplacian = np.asarray(meridian * meridian + zonal * zonal, dtype=np.float64)
        factors = laplacian**power
        if flip == inverse:
            factors = 1.0 / factors
        result[first - 1 : end] *= factors
    return result


def _normalise_variables(ds, variables: Optional[Iterable[str]]) -> List[str]:
    if variables is None:
        return list(ds.data_vars)
    return list(variables)


def _collect_template_fields(ds, template: NativeFAResource, variables: Optional[Iterable[str]]) -> Dict[str, np.ndarray]:
    selected = _normalise_variables(ds, variables)
    available = set(template.fields)
    fields: Dict[str, np.ndarray] = {}

    for var_name in selected:
        if var_name not in ds:
            raise KeyError(f"Variable not found in dataset: {var_name}")
        array = ds[var_name]
        original_fields = array.attrs.get("original_fields")
        if original_fields and any(dim in array.dims for dim in ("level", "pressure")):
            level_dim = "level" if "level" in array.dims else "pressure"
            for index, field_name in enumerate(original_fields):
                if field_name not in available:
                    raise KeyError(f"Template field not found: {field_name}")
                fields[field_name] = array.isel({level_dim: index}).values
            continue

        candidates = [var_name, var_name.replace("_", ".")]
        field_name = next((candidate for candidate in candidates if candidate in available), None)
        if field_name is None:
            raise KeyError(f"Could not map dataset variable {var_name} to a template FA field")
        fields[field_name] = array.values

    return fields


def write_fa(
    ds,
    output: str,
    template: str,
    variables: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> None:
    """Write xarray data into a copy of an existing FA template."""

    template_resource = NativeFAResource(template)
    fields = _collect_template_fields(ds, template_resource, variables)
    template_resource.write_template(output, fields, overwrite=overwrite)


def create_fa_from_scratch(
    output: str,
    geometry,
    fields,
    validity=None,
    vertical=None,
    page_size_bytes: int = 24576,
    endian: str = ">",
) -> None:
    """Create a brand-new FA file from native Python data.

    This is a thin wrapper around :func:`fa_writer.create_fa_file`. Two
    geometry families are supported: regular lon/lat (LAM) and global
    reduced Gauss (optionally rotated/stretched). Use
    :class:`fa_writer.FARegularLonLatGeometry`,
    :class:`fa_writer.FAGlobalGaussGeometry`, and
    :class:`fa_writer.FAFieldData` to describe the inputs.

    Projected LAM geometries (Lambert/Mercator/polar stereographic) raise
    ``NotImplementedError``. For those, copy an existing template via
    :func:`write_fa` instead.
    """

    from .fa_writer import create_fa_file

    create_fa_file(
        output,
        geometry=geometry,
        fields=fields,
        validity=validity,
        vertical=vertical,
        page_size_bytes=page_size_bytes,
        endian=endian,
    )
