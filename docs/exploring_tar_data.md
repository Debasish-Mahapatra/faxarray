# Exploring FA Data from Tar Archives

`faxarray` provides a workflow for exploring FA files directly from `.tar.gz` archives.

## Basic Usage

```python
import faxarray as fx

# Open archive - temp_dir is required
ds = fx.open_tar('pf20130101.tar.gz', temp_dir='/tmp/mydata')

# Access data (lazy loading - data loaded on demand)
print(ds['SURFTEMPERATURE'])

# Cleanup temp files when done
ds.close()
```

## How It Works

1. Files are extracted to `temp_dir`
2. Data is read lazily (Dask arrays)
3. Data is loaded from disk only when accessed
4. `ds.close()` deletes the temp directory

## Memory Control

Specify variables to limit what's loaded:

```python
ds = fx.open_tar('archive.tar.gz',
                 temp_dir='/tmp/mydata',
                 variables=['SURFTEMPERATURE'])
```

## Filter Files

Only extract certain timesteps:

```python
ds = fx.open_tar('archive.tar.gz',
                 temp_dir='/tmp/mydata',
                 pattern='*+000*')  # Only hour 0
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `tarpath` | Yes | Path to tar archive |
| `temp_dir` | Yes | Directory to extract files to |
| `variables` | No | List of variables to load |
| `pattern` | No | Glob pattern to filter files |

## Important

- **Always call `ds.close()`** to delete temp files
- Data uses lazy loading (loaded when accessed)
- Use `variables` parameter to reduce memory
