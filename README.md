# faxarray

**User-friendly interface for Météo-France FA files with xarray integration**

A Python package that wraps EPyGrAM with a clean, xarray-like API. Provides easy plotting, tar archive support, and NetCDF conversion.

## Features

- Easy plotting with `.plot()` methods (like xarray)
- Native xarray backend - use `xr.open_dataset()` directly on FA files
- Tar archive support - open FA files directly from `.tar.gz` archives
- Multi-file conversion - combine multiple FA files into one NetCDF
- Simple API - no complex initialization required
- CLI tool for quick operations

## Installation

**Important:** This package requires **Python 3.11**.

```bash
# Create a new environment with Python 3.11
conda create -n faxarray_env python=3.11
conda activate faxarray_env

# Install faxarray
pip install git+https://github.com/Debasish-Mahapatra/faxarray
```

## Quick Start

### Open FA files with xarray

```python
import xarray as xr
import faxarray  # Just import to register the backend

# Open FA file - auto-detects format
ds = xr.open_dataset('pfABOFABOF+0001')

# All xarray operations work
print(ds['SURFTEMPERATURE'].mean())
ds.to_netcdf('output.nc')
```

### Open FA files from tar archives

```python
import faxarray as fx

# Open directly from tar archive
ds = fx.open_tar('pf20130101.tar.gz', '/tmp/extract')

# Use xarray filtering (lazy loading - fast!)
temp = ds['SURFTEMPERATURE'].isel(time=0)
temp.plot()

ds.close()  # Cleanup temp files
```

See [docs/exploring_tar_data.md](docs/exploring_tar_data.md) for full documentation.

### Convert multiple FA files to NetCDF

```python
import faxarray as fx

# Combine multiple files with de-accumulation
ds = fx.open_mfdataset(
    'pf*+*',
    variables=['SURFPREC.EAU.CON'],  # Limit memory
    deaccumulate=['SURFPREC.EAU.CON'],
    output_file='output.nc'
)
```

## Command Line Interface

```bash
# File information
faxarray info file.fa
faxarray info file.fa --list-vars

# Convert single file
faxarray convert input.fa output.nc

# Convert multiple files with de-accumulation
faxarray convert-multi 'pf*+*' output.nc \
    -v SURFPREC.EAU.CON \
    -d SURFPREC.EAU.CON
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `fx.open_fa(path)` | Open single FA file |
| `fx.open_tar(tarpath, temp_dir)` | Open FA files from tar archive |
| `fx.open_mfdataset(pattern, ...)` | Combine multiple FA files |
| `xr.open_dataset(path)` | xarray backend (auto-registered) |

### Key Parameters

**open_tar:**
- `tarpath` - Path to tar archive
- `temp_dir` - Extraction directory (required)
- `variables` - List of variables to load
- `pattern` - Glob pattern to filter files

**open_mfdataset:**
- `paths` - Glob pattern or list of files
- `variables` - Variables to load (saves memory)
- `deaccumulate` - Variables to de-accumulate
- `output_file` - Stream directly to NetCDF

## Documentation

- [Exploring Tar Data](docs/exploring_tar_data.md) - Full guide for tar archives
- [Memory Usage](docs/MEMORY_USAGE.md) - Tips for large datasets

## Requirements

- Python 3.11
- numpy, xarray, netCDF4, matplotlib
- [epygram](https://github.com/UMR-CNRM/EPyGrAM) (v2.0.7+, auto-installed)
- h5py, cartopy

## License

MIT
