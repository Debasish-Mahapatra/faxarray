# Memory Usage Guidelines

When converting FA files to NetCDF, memory usage depends on how many variables you're processing.

## Memory Requirements

| Variables | Approx. Memory |
|-----------|----------------|
| 1-5 variables | ~1 GB |
| 10-20 variables | ~2-3 GB |
| All variables (~100+) | ~10-15 GB |

## Low Memory Usage

Specify only the variables you need:

```python
import faxarray as fx

# Specify variables to limit memory
ds = fx.open_mfdataset(
    'pf*+*',
    variables=['SURFPREC.EAU.CON', 'SURFTEMPERATURE'],
    deaccumulate=['SURFPREC.EAU.CON'],
    output_file='output.nc'
)
```

## High Memory Usage (All Variables)

If you need all variables, ensure you have sufficient RAM (16 GB+ recommended):

```python
# This loads all ~100+ variables per file (~10 GB memory)
ds = fx.open_mfdataset('pf*+*', output_file='output.nc')
```

## Why This Happens

Each FA file contains ~100 variables. When `variables` is not specified, all are loaded into memory for efficient processing. The data is written to disk incrementally, but at least one file's worth of data must be in memory at any time.
