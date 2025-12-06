# Exploring FA Data from Tar Archives

`faxarray` provides a memory-efficient workflow for exploring FA files directly from `.tar.gz` archives without extracting them manually. This uses **Lazy Loading** (Dask) to ensure only the requested data is loaded into memory.

## Simple Usage

Using `open_tar` with the `chunks` parameter enables exploration mode:

```python
import faxarray as fx

# Open the archive with chunking (lazy loading)
ds = fx.open_tar('pf20130101.tar.gz', chunks={'time': 1})

print(ds)
# Dimensions: (time: 25, level: 87, y: 480, x: 480)
# Data is NOT loaded into memory yet!
```

## Plotting Data (Lazy)

You can select and plot slices without loading the entire dataset:

```python
# Select a variable and time slice
temp = ds['SURFTEMPERATURE'].isel(time=0)

# Plotting triggers loading ONLY for this slice (~1MB)
temp.plot()
```

## Automatic Cleanup

When you open a tar file, `faxarray` extracts files to a temporary directory. To ensure these are cleaned up, use the `close()` method or a context manager:

### Using Context Manager (Recommended)

```python
with fx.open_tar('pf20130101.tar.gz', chunks={'time': 1}) as ds:
    # Do your analysis here
    ds['SURFTEMPERATURE'].mean(dim=['x', 'y']).plot()

# Temporary files are automatically deleted here
```

### Manual Cleanup

```python
ds = fx.open_tar('pf20130101.tar.gz', chunks={'time': 1})
try:
    # Analysis...
    pass
finally:
    ds.close()  # Cleanup happens here
```

## Performance Tips

1.  **Always use `chunks={'time': 1}`**: This tells `xarray` to handle each file (timestep) as a separate chunk, which is optimal for the file structure.
2.  **Filter Variables**: If you only need specific variables, pass `variables=['VAR1', 'VAR2']` to `open_tar`. This speeds up the metadata reading phase.
    ```python
    ds = fx.open_tar('data.tar.gz', variables=['SURFTEMPERATURE'], chunks={'time': 1})
    ```
3.  **Use `join='outer'` is Automatic**: If your files have slightly different levels (e.g., some variables missing in one file), `faxarray` automatically handles this by padding with NaN.
