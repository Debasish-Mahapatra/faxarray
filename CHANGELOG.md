# Changelog

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

### Technical Notes
- De-accumulation verified against pure epygram (exact match)
- Time dimension uses proper datetime values from FA metadata
- Variable names with dots (e.g., `SURFPREC.EAU.CON`) automatically mapped to underscore version (`SURFPREC_EAU_CON`)
