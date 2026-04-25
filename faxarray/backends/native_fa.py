"""Native FA access built on the vendored LFI/FA format knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .eccodes_ctypes import EccodesCError, decode_grib_values
from .grib_mf_codec import FAGribMFError, get_legacy_codec
from .native_lfi import LFIFile


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
    """Geometry information needed by the public faxarray reader."""

    name: str
    shape: Tuple[int, int]
    lons: np.ndarray
    lats: np.ndarray
    projection: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class NativeFAHeader:
    """Decoded subset of the FA header."""

    ktronc: int
    ny: int
    nx: int
    nlevels: int
    ktyptr: int
    red_points: np.ndarray
    sinlat: np.ndarray
    hybrid: np.ndarray

    @property
    def grid_size(self) -> int:
        return self.nx * self.ny


@dataclass(frozen=True)
class NativeFASpectralLayout:
    """Decoded ALADIN spectral coefficient layout from the FA frame."""

    no_zpar: np.ndarray
    by_meridian: Tuple[Tuple[int, int, int], ...]
    coeff_count: int


class NativeFAResource:
    """Read FA files without EPYGRAM."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.lfi = LFIFile(filepath)
        self._fields: Optional[List[str]] = None
        self._header: Optional[NativeFAHeader] = None
        self._geometry: Optional[NativeFAGeometry] = None
        self._spectral_layout: Optional[NativeFASpectralLayout] = None

    def close(self) -> None:
        """Mirror the EPYGRAM resource API."""

    @property
    def fields(self) -> List[str]:
        if self._fields is None:
            self._fields = self.lfi.list_fa_fields()
        return self._fields

    def listfields(self) -> List[str]:
        return list(self.fields)

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

    def _article_as_int64(self, name: str) -> np.ndarray:
        return np.frombuffer(self.lfi.read_article_bytes(name), dtype=self.lfi.word_dtype).copy()

    def _article_as_float64(self, name: str) -> np.ndarray:
        return np.frombuffer(self.lfi.read_article_bytes(name), dtype=self.lfi.float64_dtype).copy()

    def _read_header(self) -> NativeFAHeader:
        dims = self._article_as_int64("CADRE-DIMENSIONS")
        if dims.size < 5:
            raise NativeFAError("FA header article CADRE-DIMENSIONS is too short")
        return NativeFAHeader(
            ktronc=int(dims[0]),
            ny=int(dims[1]),
            nx=int(dims[2]),
            nlevels=int(dims[3]),
            ktyptr=int(dims[4]),
            red_points=self._article_as_int64("CADRE-REDPOINPOL"),
            sinlat=self._article_as_float64("CADRE-SINLATITUD"),
            hybrid=self._article_as_float64("CADRE-FOCOHYBRID"),
        )

    def _build_geometry(self) -> NativeFAGeometry:
        header = self.header
        if header.ktyptr > 0:
            raise UnsupportedFAEncodingError("global reduced Gauss FA geometry is not supported yet")

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

    def fieldencoding(self, name: str) -> Dict[str, object]:
        if name in _HEADER_ARTICLES:
            return {"exists": True, "ftype": "Misc"}
        try:
            words = self.lfi.read_article_words(name, max_words=5)
        except KeyError:
            return {"exists": False, "ftype": "?"}
        if len(words) < 2:
            return {"exists": True, "ftype": "Misc"}

        kngr = int(words[0])
        spectral_flag = int(words[1])
        if spectral_flag not in (0, 1):
            return {"exists": True, "ftype": "Misc"}
        if not (-2 <= kngr <= 4 or kngr >= 100):
            return {"exists": True, "ftype": "Misc"}

        if kngr in (-2, -1, 0):
            knbits = 0
            kstron = 0
            kpuila = 0
        else:
            knbits = int(words[2]) if len(words) > 2 else 0
            kstron = int(words[3]) if len(words) > 3 else 0
            kpuila = int(words[4]) if len(words) > 4 else 0

        return {
            "exists": True,
            "ftype": "H2D",
            "spectral": spectral_flag == 1,
            "KNGRIB": kngr,
            "KNBITS": knbits,
            "KSTRON": kstron,
            "KPUILA": kpuila,
        }

    def get_validity(self) -> Dict[str, object]:
        try:
            date = self._article_as_int64("DATE-DES-DONNEES")
        except KeyError:
            return {"valid_time": None, "base_time": None, "lead_time": None}

        if date.size < 7 or int(date[0]) <= 0:
            return {"valid_time": None, "base_time": None, "lead_time": None}

        try:
            base = datetime(int(date[0]), int(date[1]), int(date[2]), int(date[3]), int(date[4]))
        except ValueError:
            return {"valid_time": None, "base_time": None, "lead_time": None}

        seconds = 0
        try:
            datx = self._article_as_int64("DATX-DES-DONNEES")
            if datx.size > 3 and int(datx[3]) > 0:
                seconds = int(datx[3])
        except KeyError:
            pass

        if seconds == 0:
            unit = int(date[5])
            term = int(date[6])
            if unit == 1:
                seconds = term * 3600
            elif unit == 2:
                seconds = term * 86400

        lead = timedelta(seconds=seconds)
        return {
            "valid_time": np.datetime64(base + lead),
            "base_time": np.datetime64(base),
            "lead_time": np.timedelta64(int(seconds), "s"),
        }

    def readfield(self, name: str, convert_spectral: bool = True) -> np.ndarray:
        encoding = self.fieldencoding(name)
        if not encoding.get("exists"):
            raise KeyError(f"Field is unknown in file: {name}")
        if encoding.get("ftype") != "H2D":
            raise UnsupportedFAEncodingError(f"Field is not a horizontal data field: {name}")

        spectral = bool(encoding.get("spectral"))
        if spectral:
            coefficients = self._read_spectral_coefficients(name, encoding)
            if convert_spectral:
                return self._spectral_to_gridpoint(coefficients)
            return coefficients

        kngr = int(encoding["KNGRIB"])
        if kngr in (-1, 0):
            return self._read_raw64(name, spectral=False)
        if kngr == -2:
            return self._read_raw32(name, spectral=False)
        return self._read_packed(name, kngr, spectral=False)

    def _expected_size(self, spectral: bool) -> int:
        if spectral:
            raise UnsupportedFAEncodingError("spectral coefficient sizing is not implemented yet")
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

    def _get_spectral_layout(self) -> NativeFASpectralLayout:
        if self._spectral_layout is not None:
            return self._spectral_layout

        header = self.header
        if header.ktyptr >= 0:
            raise UnsupportedFAEncodingError(
                "native spectral-to-gridpoint conversion currently supports ALADIN LAM fields only"
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

        self._spectral_layout = NativeFASpectralLayout(
            no_zpar=no_zpar,
            by_meridian=tuple(by_meridian[:expected_rows]),
            coeff_count=coeff_count,
        )
        return self._spectral_layout

    def _read_spectral_coefficients(self, name: str, encoding: Mapping[str, object]) -> np.ndarray:
        kngr = int(encoding["KNGRIB"])
        if kngr in (-1, 0):
            return self._read_raw64(name, spectral=True)
        if kngr == -2:
            return self._read_raw32(name, spectral=True)
        if kngr in (1, 2):
            return self._read_legacy_packed_spectral(name, kngr, encoding)

        data = self.lfi.read_article_bytes(name)
        if data.find(b"GRIB") >= 0:
            raise UnsupportedFAEncodingError(
                f"{name} is a GRIB_API-packed spectral field. Native spectral GRIB_API decoding is not supported yet."
            )
        raise UnsupportedFAEncodingError(f"{name} uses unsupported spectral FA encoding KNGRIB={kngr}")

    def _read_legacy_packed_spectral(
        self,
        name: str,
        kngr: int,
        encoding: Mapping[str, object],
    ) -> np.ndarray:
        layout = self._get_spectral_layout()
        kstron = int(encoding["KSTRON"])
        kpuila = int(encoding["KPUILA"])
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
        layout: NativeFASpectralLayout,
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
        layout: NativeFASpectralLayout,
    ) -> np.ndarray:
        if kpuila == 0:
            return coefficients

        result = np.array(coefficients, dtype=np.float64, copy=True)
        power = abs(kpuila)
        inverse = kpuila > 0
        for meridian in range(1, self.header.ktronc + 1):
            start, end, _ = layout.by_meridian[meridian]
            first = max(start + 4 * (1 + kstron - meridian), start + 4)
            if first > end:
                continue
            positions = np.arange(first, end + 1, dtype=np.int64)
            zonal = (positions - start) // 4
            laplacian = np.asarray(meridian * meridian + zonal * zonal, dtype=np.float64)
            factors = laplacian**power
            if inverse:
                factors = 1.0 / factors
            result[first - 1 : end] *= factors
        return result

    def _spectral_to_gridpoint(self, coefficients: np.ndarray) -> np.ndarray:
        layout = self._get_spectral_layout()
        ny, nx = self.geometry.shape
        scale = float(nx * ny)
        spectrum = np.zeros((ny, nx), dtype=np.complex128)

        for meridian, (start, _, max_zonal) in enumerate(layout.by_meridian):
            for zonal in range(max_zonal + 1):
                block = coefficients[start - 1 + 4 * zonal : start - 1 + 4 * zonal + 4]
                a = float(block[0])
                c = float(block[1])
                b = float(block[2])
                d = float(block[3])

                if zonal == 0 and meridian == 0:
                    spectrum[0, 0] += scale * a
                elif meridian == 0:
                    spectrum[0, zonal] += 0.5 * scale * (a - 1j * b)
                    spectrum[0, (-zonal) % nx] += 0.5 * scale * (a + 1j * b)
                elif zonal == 0:
                    spectrum[meridian, 0] += 0.5 * scale * (a - 1j * c)
                    spectrum[(-meridian) % ny, 0] += 0.5 * scale * (a + 1j * c)
                else:
                    spectrum[meridian, zonal] += 0.25 * scale * (a - d - 1j * (b + c))
                    spectrum[(-meridian) % ny, zonal] += 0.25 * scale * (a + d - 1j * (b - c))
                    spectrum[meridian, (-zonal) % nx] += 0.25 * scale * (a + d + 1j * (b - c))
                    spectrum[(-meridian) % ny, (-zonal) % nx] += 0.25 * scale * (a - d + 1j * (b + c))

        return np.fft.ifft2(spectrum).real.astype(np.float64, copy=False)

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
            writable._replace_raw_field(field_name, np.asarray(data))

    def _replace_raw_field(self, name: str, values: np.ndarray) -> None:
        encoding = self.fieldencoding(name)
        if encoding.get("ftype") != "H2D":
            raise UnsupportedFAEncodingError(f"Cannot write non-H2D field: {name}")
        if encoding.get("spectral"):
            raise UnsupportedFAEncodingError(f"Cannot write spectral field yet: {name}")
        kngr = int(encoding["KNGRIB"])
        expected_shape = self.geometry.shape
        squeezed = np.squeeze(values)
        if squeezed.shape != expected_shape:
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
        else:
            raise UnsupportedFAEncodingError(f"Cannot write packed field {name} with KNGRIB={kngr}")

        if len(replacement) < len(original):
            replacement += b"\x00" * (len(original) - len(replacement))
        if len(replacement) != len(original):
            raise ValueError(f"{name} replacement does not fit the template article")
        self.lfi.write_article_bytes(name, replacement)


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
