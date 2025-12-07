# Exploring FA Data from Tar Archives

This guide explains how to use `faxarray` to work with FA files stored in `.tar` or `.tar.gz` archives directly, without manually extracting them.

## Overview

Many meteorological datasets are distributed as tar archives containing multiple FA files, one per timestep. `faxarray` provides the `open_tar()` function to:

1. Extract files from the archive to a temporary directory
2. Read all FA files and combine them into a single xarray Dataset
3. Provide lazy loading so data is only read into memory when accessed
4. Clean up temporary files when you're done

## Quick Start

```python
import faxarray as fx

# Open a tar archive
ds = fx.open_tar('pf20130101.tar.gz', temp_dir='/scratch/myuser/temp')

# Explore the data
print(ds)
print(ds.data_vars)

# Access a variable (data loaded on demand)
temp = ds['SURFTEMPERATURE']
print(temp.mean().values)

# Plot a single timestep
temp.isel(time=0).plot()

# IMPORTANT: Always close to cleanup temp files
ds.close()
```

## Function Signature

```python
fx.open_tar(
    tarpath,           # Path to archive (required)
    temp_dir,          # Directory for extraction (required)
    pattern='*',       # Glob pattern to filter files
    concat_dim='time', # Dimension to concatenate along
    variables=None,    # Specific variables to load
    progress=False     # Show progress messages
)
```

## Parameters Explained

### `tarpath` (required)

Path to the tar archive. Supports:
- `.tar` - uncompressed tar
- `.tar.gz` or `.tgz` - gzip compressed
- `.tar.bz2` - bzip2 compressed

```python
ds = fx.open_tar('/data/archives/pf20130101.tar.gz', temp_dir='/tmp/extract')
```

### `temp_dir` (required)

Directory where files will be extracted. This is a **required parameter** because:

1. **HPC systems** often don't have `/tmp` or it's too small
2. **Explicit is better than implicit** - you know where files go
3. **Control over cleanup** - files persist until you call `close()`

```python
# Good: Use a scratch directory on HPC
ds = fx.open_tar('archive.tar.gz', temp_dir='/scratch/user/mydata')

# Good: Use a local temp directory
ds = fx.open_tar('archive.tar.gz', temp_dir='./temp_extract')

# Good: Use an absolute path
ds = fx.open_tar('archive.tar.gz', temp_dir='/home/user/temp/extract')
```

**Warning**: The entire `temp_dir` is deleted when you call `ds.close()`. Don't use a directory with important files!

### `pattern` (optional, default='*')

Glob pattern to filter which files are extracted from the archive. Useful for:
- Extracting only certain timesteps
- Skipping analysis files
- Reducing extraction time

```python
# Only extract hour 0-3
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data', pattern='*+000[0-3]')

# Only extract pf files (skip other types)
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data', pattern='pf*')

# Extract hours 0, 6, 12, 18 (every 6 hours)
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data', pattern='*+00[01][0268]')
```

### `variables` (optional, default=None)

List of variable names to load. Reduces memory usage by only reading specified fields.

```python
# Load all variables (default) - uses more memory
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data')

# Load only 2 variables - uses less memory
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data',
                 variables=['SURFTEMPERATURE', 'SURFPREC.EAU.CON'])
```

**Memory comparison** (approximate for 25 FA files):
- All variables: ~10-15 GB
- 2-3 variables: ~0.5-1 GB

### `progress` (optional, default=False)

Show progress messages during extraction and loading.

```python
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data', progress=True)

# Output:
# Opening tar archive: archive.tar.gz
#   Extracting 25 files to /tmp/data...
#   Reading 25 FA files...
#   Loaded 5/25 files...
#   Loaded 10/25 files...
#   ...
#   Combined shape: {'time': 25, 'level': 87, 'y': 480, 'x': 480}
#   Done!
```

## Understanding Lazy Loading

When you call `open_tar()`, the data is **not** immediately loaded into memory. Instead:

1. Files are extracted to `temp_dir`
2. File headers are read to understand the structure
3. Data stays on disk as Dask arrays
4. Data is loaded into memory **only when you access it**

```python
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data')
# At this point: Files extracted, but data NOT in RAM

temp = ds['SURFTEMPERATURE']
# At this point: Still lazy, no data loaded

mean = temp.mean().values
# NOW: Data is loaded and computed

# Access a single timestep (only loads ~1/25 of the data)
slice_data = temp.isel(time=0).values
```

**Benefits of lazy loading:**
- Lower memory usage
- Faster initial open time
- Only load what you actually use

## Cleanup and Resource Management

**Always call `ds.close()` when done!** This deletes the `temp_dir` and all extracted files.

### Using try/finally (recommended)

```python
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/mydata')
try:
    # Your analysis here
    result = ds['SURFTEMPERATURE'].mean()
    print(result.values)
finally:
    ds.close()  # Always runs, even if error occurs
```

### What happens if you don't close?

- Extracted files remain in `temp_dir`
- Disk space is consumed
- On HPC, your quota may fill up
- You'll need to manually delete the directory

## Real-World Examples

### Example 1: Quick Data Exploration

```python
import faxarray as fx

# Open archive
ds = fx.open_tar('/data/AROME/pf20130101.tar.gz', 
                 temp_dir='/scratch/user/temp',
                 progress=True)
try:
    # Check what's in the dataset
    print(f"Variables: {list(ds.data_vars)[:10]}...")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Quick plot
    ds['SURFTEMPERATURE'].isel(time=0).plot()
    
finally:
    ds.close()
```

### Example 2: Extract Specific Data

```python
import faxarray as fx
import numpy as np

# Only load precipitation, only first 6 hours
ds = fx.open_tar('/data/AROME/pf20130101.tar.gz',
                 temp_dir='/scratch/user/temp',
                 pattern='*+000[0-5]',
                 variables=['SURFPREC.EAU.CON', 'SURFPREC.EAU.GEC'])
try:
    # Calculate total precipitation
    total_precip = ds['SURFPREC_EAU_CON'] + ds['SURFPREC_EAU_GEC']
    
    # Get maximum
    max_precip = total_precip.max(dim='time')
    print(f"Max precip: {max_precip.max().values:.2f}")
    
finally:
    ds.close()
```

### Example 3: Time Series at a Point

```python
import faxarray as fx
import matplotlib.pyplot as plt

ds = fx.open_tar('/data/AROME/pf20130101.tar.gz',
                 temp_dir='/scratch/user/temp',
                 variables=['SURFTEMPERATURE'])
try:
    # Extract time series at a specific grid point
    lat_idx, lon_idx = 240, 240  # Center of domain
    
    ts = ds['SURFTEMPERATURE'].isel(y=lat_idx, x=lon_idx)
    
    plt.figure(figsize=(10, 4))
    plt.plot(ts.time.values, ts.values - 273.15)  # Convert to Celsius
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature at Grid Point')
    plt.show()
    
finally:
    ds.close()
```

## Troubleshooting

### "temp_dir is required" error

```python
# Wrong - temp_dir not provided
ds = fx.open_tar('archive.tar.gz')  # Error!

# Right - provide temp_dir
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data')
```

### Out of disk space

The tar extraction needs space for all FA files. Each FA file is ~200 MB, so 25 files need ~5 GB.

**Solution**: Use a directory with enough space, or use `pattern` to limit extraction:

```python
# Only extract 6 hours instead of 25
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data', pattern='*+000[0-5]')
```

### Out of memory

If you run out of RAM when accessing data:

1. **Use `variables` parameter** to limit what's loaded
2. **Process timesteps individually** instead of all at once
3. **Increase swap space** if possible

```python
# Instead of loading all variables:
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data')  # High memory

# Load only what you need:
ds = fx.open_tar('archive.tar.gz', temp_dir='/tmp/data',
                 variables=['SURFTEMPERATURE'])  # Low memory
```

### Files not deleted after close()

If temp files remain after `ds.close()`:

1. Check for errors - if your code crashed, `close()` wasn't called
2. Use try/finally to ensure cleanup
3. Manually delete with `shutil.rmtree('/path/to/temp_dir')`

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Open FA files from tar archives |
| **temp_dir** | Required, files extracted here |
| **Loading** | Lazy (data loaded on demand) |
| **Cleanup** | Call `ds.close()` to delete temp files |
| **Memory tip** | Use `variables` parameter to reduce RAM usage |
| **Speed tip** | Use `pattern` to extract fewer files |
