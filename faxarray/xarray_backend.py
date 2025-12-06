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
