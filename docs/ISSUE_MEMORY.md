# GitHub Issue: High memory usage during convert-multi with streaming

## Problem
When running `convert-multi` with streaming to file and `--chunk-hours 6`, memory usage exceeds 98% RAM on large datasets.

## Expected Behavior
With chunked streaming, memory should stay bounded regardless of total output size.

## Current Behavior
The `_append_to_netcdf` function loads the entire existing NetCDF file into memory before appending, causing memory to grow unbounded.

```python
# Current problematic code in _append_to_netcdf:
existing = xr.open_dataset(filepath)  # Loads entire file!
merged = xr.concat([existing, combined], dim=dim)
```

## Suggested Fix
Use true incremental NetCDF writing instead of load-concat-overwrite pattern:
- Use `netCDF4` library directly for append mode
- Or use Dask for out-of-core operations

## Test Case
```bash
faxarray convert-multi '/path/to/25_files/pf*' output.nc -d SURFPREC.EAU.CON --chunk-hours 6
# Result: >98% memory usage
```

## Labels
enhancement, bug
