# Exploring FA Data from Tar Archives

`faxarray` provides a workflow for exploring FA files directly from `.tar.gz` archives without extracting them manually.

## Loading Modes

### Eager Loading (Default, Recommended)

Data is loaded immediately into memory. This is the default and recommended mode:

```python
import faxarray as fx

# Eager loading - data loaded into memory immediately
ds = fx.open_tar('pf20130101.tar.gz')
```

**Cleanup**: Temporary files are deleted automatically after loading.

### Lazy Loading (Experimental, Not Recommended)

Data stays on disk until accessed. Uses Dask for chunked operations:

```python
# Lazy loading - data stays on disk (EXPERIMENTAL)
ds = fx.open_tar('pf20130101.tar.gz', chunks={'time': 1})
```

> **Warning**: Lazy loading may cause **segfaults** due to epygram's C libraries not being thread-safe. Use eager loading for production.

**Cleanup**: Call `ds.close()` to delete temporary files.

## Low Memory: Specify Variables

To reduce memory usage, only load the variables you need:

```python
ds = fx.open_tar('pf20130101.tar.gz', 
                 variables=['SURFTEMPERATURE', 'SURFPREC.EAU.CON'])
```

## Plotting

```python
temp = ds['SURFTEMPERATURE'].isel(time=0)
temp.plot()
```

## Performance Tips

1. **Specify variables**: Use `variables=['VAR1', 'VAR2']` to limit memory
2. **Filter timesteps**: Use `pattern='*+000*'` to extract only certain files
3. **Cache extraction**: Set `temp_dir='/path/to/cache'` to avoid re-extraction
