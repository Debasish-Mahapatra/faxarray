# Changelog

## [0.3.0] - Unreleased

### Added

#### New xarray Accessor Methods (DataArray)

- **`ds['var'].fa.extract_profile(lon, lat)`**: Extract vertical profile at a geographic point
  - Supports `method='nearest'` or `method='linear'` interpolation
  - Returns 1D DataArray with level coordinate
  - Adds profile location to attributes

- **`ds['var'].fa.extract_domain(region=...)`**: Extract subdomain by bounding box
  - Predefined regions: `france`, `alps`, `pyrenees`, `britain`, `iberia`, `italy`, `germany`, `benelux`, `scandinavia`, `mediterranean`
  - Or custom bounds: `lon_range=(-5, 10), lat_range=(41, 52)`

- **`ds['var'].fa.animate(dim='time')`**: Create animations over any dimension
  - Returns matplotlib FuncAnimation object
  - Save with `.save('output.gif', writer='pillow')`

#### New xarray Accessor Methods (Dataset)

- **`ds.fa.wind_speed(u_var, v_var)`**: Compute wind speed from U/V components
- **`ds.fa.wind_direction(u_var, v_var)`**: Compute wind direction (meteorological convention)
- **`ds.fa.plot_wind(u_var, v_var)`**: Plot wind vectors as quiver or barbs
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

- Version bumped to 0.3.0
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
