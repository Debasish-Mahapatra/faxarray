# Exploring FA Data from Tar Archives

`faxarray` provides a workflow for exploring FA files directly from `.tar.gz` archives without extracting them manually.

> **Note**: Dask lazy loading is available but may cause segfaults due to epygram's underlying C libraries not being thread-safe. For production use, prefer eager loading.

## Simple Usage

```python
import faxarray as fx

# Open archive (eager loading - loads all data)
ds = fx.open_tar('pf20130101.tar.gz')

print(ds)
# Dimensions: (time: 25, level: 87, y: 480, x: 480)
```

## Low Memory: Specify Variables

To save memory, only load the variables you need:

```python
ds = fx.open_tar('pf20130101.tar.gz', 
                 variables=['SURFTEMPERATURE', 'SURFPREC.EAU.CON'])
```

## Plotting Data

```python
# Select a variable and time slice
temp = ds['SURFTEMPERATURE'].isel(time=0)

# Plot
temp.plot()
```

## Cleanup

When using the default eager loading mode, temporary files are automatically cleaned up after data is loaded into memory. No manual cleanup is required.

```python
# No cleanup needed - temp files deleted automatically after load
ds = fx.open_tar('pf20130101.tar.gz')
```

## Performance Tips

1. **Specify variables**: Use `variables=['VAR1', 'VAR2']` to only load what you need
2. **Filter files**: Use `pattern='*+000*'` to only extract certain timesteps
3. **Use temp_dir**: For repeated access, set `temp_dir='/path/to/cache'` to avoid re-extraction
