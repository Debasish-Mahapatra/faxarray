"""
faxarray - Fast, user-friendly interface for Météo-France FA files
===================================================================

A modern Python package for working with FA (Fichier Arpège) files,
providing an xarray-like interface with easy plotting and fast NetCDF export.

Example usage:
    >>> import faxarray as fx
    >>> 
    >>> # Open a FA file (native API)
    >>> fa = fx.open_fa('/path/to/file.fa')
    >>> print(fa.variables)
    >>> temp = fa['S001TEMPERATURE']
    >>> temp.plot()
    >>> fa.to_netcdf('output.nc')
    >>> 
    >>> # Or use xarray directly (after importing faxarray)
    >>> import xarray as xr
    >>> ds = xr.open_dataset('pfABOFABOF+0001', engine='faxarray')
    >>> # Or auto-detect:
    >>> ds = fx.open_dataset('pfABOFABOF+0001')
"""

from .core import FADataset, FAVariable, open_fa
from .reader import FAReader
from .xarray_backend import (
    FABackendEntrypoint, open_dataset, open_mfdataset, open_tar, is_fa_file,
    TarDataset
)
from . import xarray_accessor  # Register .fa accessor on xarray DataArrays

__version__ = "0.2.0"
__author__ = "Your Name"

__all__ = [
    # Main API
    "open_fa",
    "FADataset", 
    "FAVariable",
    "FAReader",
    # xarray integration
    "open_dataset",
    "open_mfdataset",
    "open_tar",
    "TarDataset",
    "is_fa_file",
    "FABackendEntrypoint",
    # Version
    "__version__",
]
