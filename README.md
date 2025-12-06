# faxarray

**User-friendly interface for Météo-France FA files with xarray integration**

A Python package that wraps EPyGrAM with a clean, xarray-like API. Provides easy plotting and NetCDF conversion.

## Features

- 📊 **Easy plotting** with `.plot()` methods (like xarray)
- 🔄 **Native xarray backend** - use `xr.open_dataset()` directly on FA files
- 📁 **Simple API** - no complex initialization required
- 🛠️ **CLI tool** for quick operations

## Installation

**Important:** This package requires **Python 3.11**.
This is because `epygram` depends on binary packages (`ectrans4py`, `falfilfa4py`) which are currently only available for specific python versions (3.12 is not yet fully supported).

We recommend using `conda` to manage the environment:

```bash
# Create a new environment with Python 3.11
conda create -n faxarray_env python=3.11
conda activate faxarray_env

# Install faxarray
pip install git+https://github.com/Debasish-Mahapatra/faxarray
```

**Note:** The installation will automatically pull `epygram` (v2.0.7), `h5py`, and `cartopy`.

## Quick Start

### Option 1: Native xarray (recommended)

```python
import xarray as xr
import faxarray  # Just import to register the backend

# Open FA file - auto-detects format
ds = xr.open_dataset('pfABOFABOF+0001')
ds = xr.open_dataset('file.sfx')  # SURFEX files work too

# All xarray operations work directly
print(ds['SURFTEMPERATURE'].mean())
ds.to_netcdf('output.nc')
```

### Option 2: faxarray native API

```python
import faxarray as fx

# Open an FA file
fa = fx.open_fa('/path/to/pfABOFABOF+0001')

# See what's inside
print(fa)  # Shows file info
print(fa.variables[:10])  # First 10 variables

# Select multiple variables
temps = fa.select('S*TEMPERATURE')  # All temperature levels
surf = fa.select('SURF*')  # All surface fields

# Stack levels into 3D array
temp_3d = fa.stack_levels('TEMPERATURE')  # Shape: (levels, y, x)

# Convert to xarray
ds = fa.to_xarray()

# Export to NetCDF
fa.to_netcdf('output.nc')
fa.to_netcdf('output.nc', compress='zlib')  # With compression
```

## Command Line Interface

```bash
# File information
faxarray info file.fa
faxarray info file.fa --list-vars

# Convert to NetCDF
faxarray convert input.fa output.nc
faxarray convert input.fa output.nc --compress zlib

# Quick plot
faxarray plot file.fa -f S001TEMPERATURE
faxarray plot file.fa -f SURFTEMPERATURE -o temp.png
```

## API Reference

### `open_fa(filepath)`

Open an FA file and return an `FADataset`.

### `FADataset`

Main class for working with FA files.

**Properties:**
- `variables` - List of variable names
- `nvars` - Number of variables
- `shape` - Grid shape (y, x)
- `geometry` - Grid geometry info
- `lon`, `lat` - Coordinate arrays

**Methods:**
- `fa['name']` - Get a variable
- `fa[['name1', 'name2']]` - Get multiple variables
- `select(pattern)` - Select by glob/regex pattern
- `select_levels(var)` - Get all levels of a 3D variable
- `stack_levels(var)` - Stack levels into 3D array
- `to_xarray()` - Convert to xarray.Dataset
- `to_netcdf(path)` - Export to NetCDF

### `FAVariable`

A single variable from an FA file.

**Properties:**
- `data` - Numpy array of values
- `shape` - Data shape
- `lon`, `lat` - Coordinates
- `plot` - Plot accessor

**Methods:**
- `min()`, `max()`, `mean()`, `std()` - Statistics
- `to_xarray()` - Convert to xarray.DataArray
- `plot()` - Quick plot
- `plot.contourf()`, `plot.contour()`, `plot.pcolormesh()` - Plot types

## Requirements

- Python 3.11 (strictly required for `epygram` binary dependencies)
- numpy
- xarray
- netCDF4
- matplotlib
- [epygram](https://github.com/UMR-CNRM/EPyGrAM) (v2.0.7+, automatically installed)
- h5py
- cartopy

## License

MIT
