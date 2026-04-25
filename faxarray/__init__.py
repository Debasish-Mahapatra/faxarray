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

from .core import FADataset, FAVariable, create_fa_from_dataset, open_fa, write_fa
from .reader import FAReader
from .xarray_backend import (
    FABackendEntrypoint, open_dataset, open_mfdataset, open_tar, is_fa_file,
    TarDataset
)
from .fa_metadata import (
    FA_METADATA, PREDEFINED_REGIONS,
    get_metadata, apply_metadata_to_dataset
)
from .backends import (
    FAFieldData,
    FAGlobalGaussGeometry,
    FARegularLonLatGeometry,
    FAValidityInput,
    FAVerticalInput,
    create_fa_file,
)
from . import xarray_accessor  # Register .fa accessor on xarray DataArrays

__version__ = "0.4.0"
__author__ = "Debasish Mahapatra"

__all__ = [
    # Main API
    "open_fa",
    "write_fa",
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
    # FA creation from scratch
    "create_fa_file",
    "create_fa_from_dataset",
    "FAFieldData",
    "FARegularLonLatGeometry",
    "FAGlobalGaussGeometry",
    "FAValidityInput",
    "FAVerticalInput",
    # Metadata
    "FA_METADATA",
    "PREDEFINED_REGIONS",
    "get_metadata",
    "apply_metadata_to_dataset",
    # Version
    "__version__",
]
