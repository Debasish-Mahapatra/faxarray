# Changelog

## [0.4.0] - 2026-04-25

### Added

#### Native FA file creation (no template required)

- New :func:`faxarray.create_fa_file` builds a brand-new FA file from
  Python data, with no existing template needed. Supported geometries:
  - **Regular lon/lat (LAM)** via
    :class:`faxarray.FARegularLonLatGeometry`.
  - **Global reduced Gauss**, including the rotated/stretched (C2.4)
    variant, via :class:`faxarray.FAGlobalGaussGeometry`.
- New :func:`faxarray.create_fa_from_dataset` is the dataset-friendly
  wrapper: it pulls geometry from ``ds.lon`` / ``ds.lat`` when possible
  and writes every selected variable as a raw float64 H2D field.
- New :class:`faxarray.backends.lfi_writer.LFIWriter` for the
  underlying LFI container, with automatic multi-index pagination.
- New header / validity / vertical input dataclasses:
  :class:`faxarray.FAFieldData`, :class:`faxarray.FAValidityInput`,
  :class:`faxarray.FAVerticalInput`.

#### GRIB_API spectral writes

- ``write_template`` now re-encodes ``KNGRIB>=100`` (GRIB_API)
  spectral articles in place via system ecCodes, the same way it
  already handles ``KNGRIB>=100`` gridpoint writes. Coefficient size,
  packing type and ``J``/``K``/``M`` truncation are inherited from the
  original GRIB message.
- LAM spectral writes also accept a flat coefficient array directly,
  in addition to the existing gridpoint-shape input. This avoids a
  redundant gp→sp transform when the caller already has coefficients.

#### Per-field rich metadata

- New :class:`faxarray.backends.NativeFAFieldDescriptor` and
  :meth:`NativeFAResource.field_descriptor` return the EPYGRAM-style
  per-field bundle: name, ``fid`` dict, level kind/index, hybrid
  ``A``/``B`` for model levels, pressure value for P-level fields, plus
  the catalog ``long_name`` / ``units`` from
  :mod:`faxarray.fa_metadata`. The catalog falls back to the
  level-stripped base name (so e.g. ``S087TEMPERATURE`` inherits the
  ``TEMPERATURE`` description).
- :meth:`FADataset.to_xarray` now also runs
  :func:`apply_metadata_to_dataset`, so xarray Datasets built via the
  native API get the same ``long_name`` / ``units`` attributes that
  ``xr.open_dataset`` already produced.

### Removed

- The deliberate ``NotImplementedError`` stub in
  :func:`create_fa_from_scratch`. The function now performs creation
  for the supported geometry families and rejects only unrecognised
  geometry input types.
- Bundled rootpack/ifsaux model sources. Legacy ``KNGRIB=1/2`` packing now
  builds only from a user-supplied local source path or prebuilt codec library.

---

## [0.3.0] - 2026-04-25

### Added

#### Native backend coverage

- **Global reduced-Gauss FA geometry**: `KTYPTR=1` (standard) and
  `KTYPTR=2` (rotated/stretched, e.g. ARPEGE C2.4) are now decoded.
  The native geometry exposes `lat_number`, `lon_number_by_lat`, the
  Gaussian latitudes, and the stretching/pole-rotation parameters.
- **Global ARPEGE spectral-to-gridpoint** reference path using
  associated Legendre polynomials + FFT. Implemented in pure
  NumPy in `faxarray.backends.spectral`. The path runs end-to-end and
  returns finite gridpoint fields; for bit-identical agreement with
  ECMWF/ECTRANS the production pipeline should still call
  `ectrans4py` on Linux. The reference layout matches the FA "model"
  convention of `(T+1)^2` real coefficients.
- **GRIB_API spectral decode**: spectral fields packed with
  `KNGRIB >= 100` now flow through ecCodes (`packingType`,
  `bitsPerValue`, `J`/`K`/`M` truncation, real/imag arrays) when ecCodes
  is installed. Returns the raw coefficient array; downstream callers
  can pass it through `gauss_sp2gp`/`lam_sp2gp` for the inverse transform.
- **GRIB_API gridpoint write**: `write_template()` can now re-encode
  fields with `KNGRIB >= 100` in place via ecCodes, using the original
  message as the packing template.
- **Spectral writing for legacy `KNGRIB=1/2` LAM fields**: the new
  `lam_gp2sp` inverse + Laplacian repack path lets a gridpoint update
  be written back into the same packed-spectral article slot.
- **Misc / non-H2D field exposure**: header-like and scalar articles
  (e.g. `FULLPOS`, `Const.Clim.Surfa`) are now accessible as
  `read_misc_field_bytes(name)` / `read_misc_field_words(name)` and
  appear in `list_misc_fields()`.

#### Richer metadata

- New typed accessors on `NativeFAResource`:
  - `validity` returns a `NativeFAValidity` (base/valid/lead time +
    process type and cumulative duration).
  - `vertical` returns a `NativeFAVertical` exposing the hybrid `A`/`B`
    coefficients and the reference pressure, with a helper for
    half-level pressures from a surface pressure field.
  - `fieldencoding_object(name)` returns a typed `NativeFAFieldEncoding`
    that includes the human-readable `KNGRIB` packing label.
  - `metadata_summary()` returns a JSON-serialisable snapshot of the
    file's geometry, header, vertical, validity and field counts.
- `LFIFile.list_fa_fields()` now skips header articles by name rather
  than by index, so files with extra header markers (`FULLPOS`,
  `Const.Clim.Surfa`) report the right field count.

### Documented

- `create_fa_from_scratch()` now delegates to `create_fa_file()` for regular
  lon/lat and global Gauss geometries. Projected LAM creation from arbitrary
  projection parameters remains out of scope; use `write_fa()` with an
  existing template for that case.

### Changed

- `NativeFAHeader` now carries the `CADRE-FRANKSCHMI` block as
  `franchschmi`; `ny`/`nx` remain available as backwards-compatible
  aliases for the LAM `knlati`/`knxlon` fields.
- Existing `fieldencoding(name)` keeps returning a plain dict for
  backwards compatibility; new code should prefer
  `fieldencoding_object()`.

---

## [0.2.4] - 2026-02-19

### Fixed

- xarray backend `variables=[...]` now accepts stacked/normalized variable names shown in dataset output (e.g., `WIND_U_PHYS`), not only raw FA field names.

---

## [0.2.3] - 2026-02-12

### Added

#### New xarray Accessor Methods (DataArray)

- **`ds['var'].fa.extract_profile(lon, lat)`**: Extract vertical profile at a geographic point
  - Supports `method='nearest'` or `method='linear'` interpolation
  - Returns 1D DataArray with level coordinate
  - Adds profile location to attributes

- **`ds['var'].fa.extract_domain(region=...)`**: Extract subdomain by bounding box
  - Predefined regions: `france`, `alps`, `pyrenees`, `britain`, `iberia`, `italy`, `germany`, `benelux`, `scandinavia`, `mediterranean`
  - Or custom bounds: `lon_range=(-5, 10), lat_range=(41, 52)`


#### New xarray Accessor Methods (Dataset)

- **`ds.fa.extract_domain(region=...)`**: Extract subdomain for entire dataset


#### CF-Compliant Metadata

- **`fa_metadata.py`**: New module with field metadata mappings
  - `FA_METADATA`: Surface field mappings (long_name, standard_name, units)
  - `FA_3D_METADATA`: 3D field mappings
  - `PREDEFINED_REGIONS`: Geographic region definitions
  - `get_metadata(field_name)`: Lookup function
  - `apply_metadata_to_dataset(ds)`: Apply metadata to all variables

- Metadata automatically applied when loading FA files via xarray backend
- Variables now include CF-standard attributes for better interoperability

### Changed

- Version bumped to 0.2.3
- `__init__.py` exports new metadata functions and constants

---

## [0.2.2] - 2025-12-08

### Changed
- Improved README documentation
- Removed emojis from README

---

## [Unreleased] - dev branch


### Added
- **`convert-multi` CLI command**: Convert multiple FA files to single NetCDF with de-accumulation
  ```bash
  faxarray convert-multi pf*+* output.nc -d SURFPREC.EAU.CON
  ```
- **De-accumulation support** in `open_mfdataset()`:
  - `deaccumulate` parameter: List of fields to convert from cumulative to hourly
  - `chunk_hours` parameter: Memory control (default 1 hour at a time)
  - `output_file` parameter: Stream directly to NetCDF for large datasets
- **`--dlist` flag**: Read de-accumulation variables from a file
- **`_append_to_netcdf()` helper**: Incremental NetCDF writing for streaming

### Changed
- `open_mfdataset()` now sorts files by forecast hour extracted from filename
- `open_mfdataset()` produces N-1 timesteps from N files (first file is baseline)

### Benchmark Results
| Input Files | Output Timesteps | File Size |
|------------|-----------------|-----------|
| 2 files | 1 | 5.0 GB |
| 6 files | 5 | 25 GB |
| 11 files | 10 | 50 GB |

~30s per file, ~5GB per timestep output.

### Technical Notes
- De-accumulation verified against pure epygram (exact match)
- Time dimension uses proper datetime values from FA metadata
- Variable names with dots (e.g., `SURFPREC.EAU.CON`) automatically mapped to underscore version (`SURFPREC_EAU_CON`)
