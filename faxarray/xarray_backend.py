"""
xarray backend engine for FA files.

This module allows opening FA files directly with xarray:
    
    import faxarray  # Registers the backend
    import xarray as xr
    
    ds = xr.open_dataset('pfABOFABOF+0001')  # Auto-detects FA format
    ds = xr.open_dataset('file.fa')
    ds = xr.open_dataset('file.sfx')
"""

import os
import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint
from typing import Any, Dict, Iterable, Optional, Tuple

from .reader import FAReader


def is_fa_file(filename: str) -> bool:
    """
    Check if a file is an FA file by examining its content.
    
    FA files (LFI format) have a specific structure that we can detect
    by reading the first few bytes.
    
    Parameters
    ----------
    filename : str
        Path to the file
        
    Returns
    -------
    bool
        True if the file appears to be an FA file
    """
    # Check by extension first (fast path)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.fa', '.sfx'):
        return True
    
    # Check by filename pattern (common FA naming conventions)
    basename = os.path.basename(filename)
    fa_patterns = [
        basename.startswith('pf'),      # pfABOFABOF+0001
        basename.startswith('ICMSH'),   # ICMSHABOF+0001
        basename.startswith('PF'),      # PFABOFABOF+0001
        '+' in basename and not ext,    # Files with + in name and no extension
    ]
    if any(fa_patterns):
        return True
    
    # Check by file content (magic bytes) - LFI files have specific structure
    # This is a fallback for unusual naming
    try:
        with open(filename, 'rb') as f:
            # Read first 8 bytes
            header = f.read(8)
            if len(header) < 8:
                return False
            
            # LFI files start with record length markers (Fortran unformatted)
            # The first 4 bytes are typically a small integer (record length)
            # This is not 100% reliable but helps for detection
            first_int = int.from_bytes(header[:4], byteorder='little')
            
            # FA files typically have a small initial record (< 1MB)
            if 0 < first_int < 1_000_000:
                # Could be FA - try to validate further
                # Check if file size is reasonable for FA (> 1KB)
                f.seek(0, 2)  # Seek to end
                size = f.tell()
                if size > 1024:
                    return True
    except (IOError, OSError):
        pass
    
    return False


class FABackendEntrypoint(BackendEntrypoint):
    """
    xarray backend for FA files.
    
    This allows opening FA files directly with xr.open_dataset().
    The backend is automatically registered when faxarray is imported.
    """
    
    description = "Open Météo-France FA files using faxarray"
    open_dataset_parameters = ["filename_or_obj", "drop_variables", "variables", "stack_levels"]
    
    def open_dataset(
        self,
        filename_or_obj: str,
        *,
        drop_variables: Optional[Iterable[str]] = None,
        variables: Optional[Iterable[str]] = None,
        stack_levels: bool = True,
    ) -> xr.Dataset:
        """
        Open an FA file as an xarray Dataset.
        
        Parameters
        ----------
        filename_or_obj : str
            Path to the FA file
        drop_variables : iterable of str, optional
            Variables to exclude
        variables : iterable of str, optional
            Variables to include (if None, includes all)
        stack_levels : bool, default True
            If True, stack 3D fields (S001TEMP, S002TEMP → TEMP(level, y, x))
            
        Returns
        -------
        xarray.Dataset
        """
        # Use FADataset for consistent behavior with native API
        from .core import FADataset
        
        fa = FADataset(filename_or_obj)
        
        # Convert variables list if provided
        var_list = list(variables) if variables else None
        
        # Get the dataset using the same logic as native API
        ds = fa.to_xarray(variables=var_list, stack_levels=stack_levels, progress=False)
        
        # Drop variables if requested
        if drop_variables:
            ds = ds.drop_vars([v for v in drop_variables if v in ds.data_vars], errors='ignore')
        
        fa.close()
        
        # Add CF-compliant attributes
        ds.attrs.update({
            'source': str(filename_or_obj),
            'Conventions': 'CF-1.8',
            'institution': 'Météo-France',
            'source_format': 'FA',
        })
        
        # Add coordinate attributes if not present
        if 'lat' in ds.coords and 'units' not in ds['lat'].attrs:
            ds['lat'].attrs = {
                'units': 'degrees_north',
                'long_name': 'latitude',
                'standard_name': 'latitude',
            }
        if 'lon' in ds.coords and 'units' not in ds['lon'].attrs:
            ds['lon'].attrs = {
                'units': 'degrees_east',
                'long_name': 'longitude', 
                'standard_name': 'longitude',
            }
        
        # Set coordinates attribute on each variable for CF compliance
        for var_name in ds.data_vars:
            ds[var_name].attrs['coordinates'] = 'lat lon'
        
        return ds
    
    def guess_can_open(self, filename_or_obj: str) -> bool:
        """
        Check if this backend can open the given file.
        
        This is called by xarray to auto-detect the correct backend.
        """
        try:
            if isinstance(filename_or_obj, str):
                return is_fa_file(filename_or_obj)
        except Exception:
            pass
        return False


# Register with xarray
def _register_backend():
    """Register the FA backend with xarray."""
    try:
        # xarray 2022.03+ uses entry points, but we can also register manually
        xr.backends.list_engines()  # Just to verify xarray is available
    except Exception:
        pass


# Function to easily open FA files with xarray
def open_dataset(filename: str, **kwargs) -> xr.Dataset:
    """
    Open an FA file as xarray Dataset.
    
    This is a convenience function that uses our backend.
    
    Parameters
    ----------
    filename : str
        Path to the FA file
    **kwargs
        Additional arguments passed to the backend
        
    Returns
    -------
    xarray.Dataset
    """
    backend = FABackendEntrypoint()
    return backend.open_dataset(filename, **kwargs)


def open_mfdataset(
    paths,
    concat_dim: str = 'time',
    progress: bool = False,
    **kwargs
) -> xr.Dataset:
    """
    Open multiple FA files and concatenate along a dimension.
    
    This is a convenience function that opens multiple FA files and 
    concatenates them, typically along the time dimension.
    
    Parameters
    ----------
    paths : str or list of str
        Glob pattern (e.g., 'pf*+*') or list of file paths
    concat_dim : str, default 'time'
        Dimension to concatenate along
    progress : bool, default False
        Print progress while loading files
    **kwargs
        Additional arguments passed to open_dataset
        
    Returns
    -------
    xarray.Dataset
        Combined dataset with all files concatenated
        
    Example
    -------
    >>> ds = fx.open_mfdataset('pf*+*')  # All 25 forecast hours
    >>> ds['TEMPERATURE'].shape  # (25, 87, 480, 480)
    >>> ds.time.values  # ['00:00', '01:00', ..., '24:00']
    """
    from glob import glob
    
    # Handle glob pattern or list of files
    if isinstance(paths, str):
        file_list = sorted(glob(paths))
        if not file_list:
            raise FileNotFoundError(f"No files found matching pattern: {paths}")
    else:
        file_list = list(paths)
    
    if progress:
        print(f"Opening {len(file_list)} FA files...")
    
    # Open each file
    datasets = []
    for i, filepath in enumerate(file_list):
        ds = open_dataset(filepath, **kwargs)
        datasets.append(ds)
        if progress and (i + 1) % 5 == 0:
            print(f"  Loaded {i + 1}/{len(file_list)} files...")
    
    if progress:
        print(f"Concatenating along '{concat_dim}' dimension...")
    
    # Concatenate along the specified dimension
    combined = xr.concat(datasets, dim=concat_dim)
    
    if progress:
        print(f"Done! Combined shape: {dict(combined.sizes)}")
    
    return combined


class TarDataset(xr.Dataset):
    """
    Wrapper for xarray.Dataset that handles temporary directory cleanup.
    
    When this dataset is closed (via .close() or context manager),
    it automatically deletes the temporary directory if cleanup is enabled.
    """
    __slots__ = ('_temp_dir', '_cleanup')
    
    def __init__(self, ds: xr.Dataset, temp_dir: str, cleanup: bool = True):
        super().__init__(ds)
        self._temp_dir = temp_dir
        self._cleanup = cleanup
        self.set_close(self._close_callback)
    
    def _close_callback(self):
        """Cleanup temporary directory on close."""
        if self._cleanup and self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def _read_single_file(filepath: str, variables=None, stack_levels=True, lazy=False) -> xr.Dataset:
    """Helper to read a single FA file."""
    from .core import FADataset
    
    fa = FADataset(filepath)
    try:
        if lazy:
            # Lazy loading - dask arrays
            ds = fa.to_xarray_lazy(variables=variables, stack_levels=stack_levels)
        else:
            # Eager loading
            var_list = list(variables) if variables else None
            ds = fa.to_xarray(variables=var_list, stack_levels=stack_levels)
        
        # Add CF-compliant attributes
        ds.attrs.update({
            'source': str(filepath),
            'Conventions': 'CF-1.8',
            'institution': 'Météo-France',
            'source_format': 'FA',
        })
        
        return ds
    finally:
        if not lazy:
            fa.close()


def open_tar(
    tarpath: str,
    pattern: str = '*',
    temp_dir: str = None,
    chunks: dict = None,
    concat_dim: str = 'time',
    variables: list = None,
    progress: bool = False,
    **kwargs
) -> TarDataset:
    """
    Open FA files from a tar archive.
    
    Provides a memory-efficient way to explore data from FA archives using
    lazy loading (Dask).
    
    Parameters
    ----------
    tarpath : str
        Path to the tar archive (.tar, .tar.gz, .tgz)
    pattern : str, default '*'
        Glob pattern to filter files within the archive (e.g., '*+000*')
    temp_dir : str, optional
        Directory to extract files to. If None, uses system temp.
    chunks : dict, optional
        Chunk sizes for lazy loading (e.g., {'time': 1}).
        Providing this is highly recommended for memory efficiency.
        If None, data will be loaded eagerly (careful with large archives!).
    concat_dim : str, default 'time'
        Dimension to concatenate along
    variables : list of str, optional
        Specific variables to load (reduces memory usage)
    progress : bool, default False
        Print progress messages
    **kwargs
        Additional arguments passed to the backend
        
    Returns
    -------
    TarDataset
        Combined dataset wrapped in TarDataset for cleanup management.
        Call `ds.close()` when done to cleanup temp files.
        
    Examples
    --------
    Exploration mode (lazy, memory-efficient):
    
    >>> ds = fx.open_tar('pf20130101.tar.gz', chunks={'time': 1})
    >>> ds['SURFTEMPERATURE'].isel(time=0).plot()  # Only loads one timestep
    >>> ds.close()  # Cleanup temp files
    
    Using context manager for automatic cleanup:
    
    >>> with fx.open_tar('pf20130101.tar.gz', chunks={'time': 1}) as ds:
    ...     ds['SURFTEMPERATURE'].isel(time=0).plot()
    ... # Temp files automatically cleaned up
    """
    import tarfile
    import tempfile
    import shutil
    import fnmatch
    
    if progress:
        print(f"Opening tar archive: {tarpath}")
    
    # Determine extraction directory
    if temp_dir is None:
        extract_dir = tempfile.mkdtemp(prefix='faxarray_')
    else:
        extract_dir = temp_dir
        os.makedirs(extract_dir, exist_ok=True)
    
    try:
        # Open tar and filter members
        with tarfile.open(tarpath, 'r:*') as tar:
            all_members = tar.getmembers()
            # Filter by pattern (only files, not directories)
            members = [m for m in all_members 
                      if m.isfile() and fnmatch.fnmatch(m.name, pattern)]
            
            if not members:
                raise FileNotFoundError(
                    f"No files matching pattern '{pattern}' in {tarpath}"
                )
            
            if progress:
                print(f"  Extracting {len(members)} files to {extract_dir}...")
            
            # Extract matching files
            tar.extractall(extract_dir, members=members)
        
        # Get list of extracted files
        extracted_files = sorted([
            os.path.join(extract_dir, m.name) for m in members
        ])
        
        if progress:
            print(f"  Reading {len(extracted_files)} FA files...")
        
        # Read files sequentially
        # Use lazy loading when chunks is requested
        use_lazy = chunks is not None
        datasets = []
        for i, filepath in enumerate(extracted_files):
            ds = _read_single_file(filepath, variables=variables,
                                   stack_levels=kwargs.get('stack_levels', True),
                                   lazy=use_lazy)
            datasets.append(ds)
            if progress and (i + 1) % 5 == 0:
                print(f"  Loaded {i + 1}/{len(extracted_files)} files...")
        
        if progress:
            print(f"  Concatenating along '{concat_dim}' dimension...")
        
        # Concatenate along the specified dimension
        # join='outer' handles variables with different level counts (e.g. 86 vs 87)
        combined = xr.concat(datasets, dim=concat_dim, join='outer')
        
        if progress:
            print(f"  Combined shape: {dict(combined.sizes)}")
        
        # Apply chunking for lazy mode
        if chunks:
            combined = combined.chunk(chunks)
            if progress:
                print(f"  Applied chunking: {chunks}")
            # Return wrapped dataset for cleanup on close
            return TarDataset(combined, extract_dir, cleanup=True)
        else:
            # Eager mode: load data and allow cleanup
            if progress:
                print(f"  Loading data into memory...")
            combined = combined.load()
            
            if temp_dir is None:  # Only cleanup auto-created temp dirs
                shutil.rmtree(extract_dir, ignore_errors=True)
                extract_dir = None
            
            if progress:
                print(f"  Done!")
            
            return combined
            
    except Exception:
        # Cleanup on error if we created the temp dir
        if temp_dir is None and 'extract_dir' in locals() and extract_dir:
            shutil.rmtree(extract_dir, ignore_errors=True)
        raise
