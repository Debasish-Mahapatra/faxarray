"""
Core classes for faxarray: FADataset and FAVariable.

These provide the main user-facing API for working with FA files,
inspired by xarray's Dataset and DataArray interface.
"""

import re
import fnmatch
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
from collections import defaultdict
import threading

try:
    import dask.array as da
    from dask import delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Global lock for FA access. Some native and optional compiled FA routines keep
# process-level state, so lazy reads are serialized.
FA_BACKEND_LOCK = threading.Lock()

def read_field_delayed(filepath: str, field_name: str):
    """
    Read a field lazily using dask.delayed.
    Crucially, uses a lock to prevent concurrent FA backend access.
    """
    if not HAS_DASK:
        raise ImportError("Dask is required for lazy loading")
        
    def _read_with_lock(path, name):
        with FA_BACKEND_LOCK:
            # Create a FRESH reader for each access to avoid state issues
            reader = FAReader(path)
            try:
                return reader.read_field(name)
            finally:
                reader.close()
    
    return delayed(_read_with_lock)(filepath, field_name)

from .reader import FAReader, FAGeometry
from .plotting import PlotAccessor


# Patterns to detect level-based field names
# S-prefix: Model (η) levels - S001 to S0XX (3 digits), level 1 = top of atmosphere
MODEL_LEVEL_PATTERN = re.compile(r'^S(\d{3})(.+)$')
# P-prefix: Pressure levels - P followed by 5-digit pressure in Pa (e.g., P50000 = 500 hPa)
PRESSURE_LEVEL_PATTERN = re.compile(r'^P(\d{5})(.+)$')


def detect_3d_fields(field_names: List[str]) -> Dict[str, Dict]:
    """
    Detect 3D fields from a list of field names.
    
    Handles two types of vertical coordinates:
    - Model levels (S-prefix): S001TEMPERATURE, S002TEMPERATURE, etc.
      Level 1 is at top of atmosphere, highest level is at surface.
    - Pressure levels (P-prefix): P50000TEMPERATURE (500 hPa), P85000TEMPERATURE (850 hPa)
      Value is pressure in Pa. Lower pressure = higher altitude.
    
    Parameters
    ----------
    field_names : list of str
        List of field names from FA file
        
    Returns
    -------
    dict
        Mapping of base variable name to dict with:
        - 'levels': list of (level_value, full_name) tuples
        - 'type': 'model' or 'pressure'
        - 'units': 'level' or 'Pa'
    """
    model_groups = defaultdict(list)
    pressure_groups = defaultdict(list)
    
    for name in field_names:
        # Check for model levels (S-prefix)
        match = MODEL_LEVEL_PATTERN.match(name)
        if match:
            level = int(match.group(1))
            base_name = match.group(2)
            model_groups[base_name].append((level, name))
            continue
        
        # Check for pressure levels (P-prefix)
        match = PRESSURE_LEVEL_PATTERN.match(name)
        if match:
            pressure_pa = int(match.group(1))
            base_name = match.group(2)
            pressure_groups[base_name].append((pressure_pa, name))
    
    result = {}
    
    # Process model level groups
    for base_name, levels in model_groups.items():
        if len(levels) > 1:  # Only consider as 3D if more than 1 level
            # IMPORTANT: Reverse the native FA file orientation for more intuitive indexing
            # 
            # Native FA file convention:
            #   - S001 = model top (highest altitude)
            #   - S087 = surface (lowest altitude, for 87-level model)
            #
            # Faxarray stacking convention (REVERSED from native):
            #   - Index 0 = surface (S087, highest level number)
            #   - Index 86 = model top (S001, lowest level number)
            #
            # This makes the array indexing more intuitive: increasing index = increasing altitude
            # Sort DESCENDING by level number (87, 86, 85, ..., 3, 2, 1)
            sorted_levels = sorted(levels, key=lambda x: x[0], reverse=True)
            result[base_name] = {
                'levels': sorted_levels,
                'type': 'model',
                'units': '1',
                'positive': 'up',  # Array index increases upward from surface to model top
            }
    
    # Process pressure level groups
    for base_name, levels in pressure_groups.items():
        if len(levels) > 1:
            # Handle P00000 ambiguity: P00000 = 1000 hPa = 100000 Pa (surface)
            # Convert encoded value to actual pressure
            converted_levels = []
            for encoded_pa, name in levels:
                if encoded_pa == 0:
                    # P00000 means 1000 hPa = 100000 Pa (surface)
                    actual_pa = 100000
                else:
                    actual_pa = encoded_pa
                converted_levels.append((actual_pa, name))
            
            # Sort by pressure DESCENDING: high pressure (surface) first, low pressure (top) last
            # This way index 0 = surface, increasing index = higher altitude
            sorted_levels = sorted(converted_levels, key=lambda x: x[0], reverse=True)
            result[f'P_{base_name}'] = {  # Add P_ prefix to distinguish from model levels
                'levels': sorted_levels,
                'type': 'pressure',
                'units': 'Pa',
                'positive': 'up',  # Index increases toward lower pressure (higher altitude)
            }
    
    return result


def get_surface_fields(field_names: List[str]) -> List[str]:
    """
    Get field names that are surface (2D) fields, not part of 3D level data.
    
    Parameters
    ----------
    field_names : list of str
        List of field names
        
    Returns
    -------
    list of str
        Names of 2D surface fields
    """
    surface = []
    for name in field_names:
        if not MODEL_LEVEL_PATTERN.match(name) and not PRESSURE_LEVEL_PATTERN.match(name):
            surface.append(name)
    return surface


def _enable_lonlat_nearest_sel(ds: xr.Dataset) -> xr.Dataset:
    """
    Enable nearest-neighbor .sel(lon=..., lat=...) on datasets with 2D lon/lat.

    This uses xarray's NDPointIndex when available. On older xarray versions
    (or if index creation fails), the dataset is returned unchanged.
    """
    if 'lon' not in ds.coords or 'lat' not in ds.coords:
        return ds

    index_mod = getattr(xr, 'indexes', None)
    index_cls = getattr(index_mod, 'NDPointIndex', None) if index_mod is not None else None
    if index_cls is None or not hasattr(ds, 'set_xindex'):
        return ds

    try:
        return ds.set_xindex(('lon', 'lat'), index_cls)
    except Exception:
        return ds


class FAVariable:
    """
    A single variable from an FA file.
    
    Similar to xarray.DataArray, provides easy access to data
    and coordinates with built-in plotting.
    
    Attributes
    ----------
    name : str
        Variable name (e.g., 'S001TEMPERATURE')
    data : np.ndarray
        The data values (lazy loaded)
    shape : tuple
        Shape of the data
    lon : np.ndarray
        2D longitude coordinates
    lat : np.ndarray
        2D latitude coordinates
    plot : PlotAccessor
        Plotting methods (.plot(), .plot.contourf(), etc.)
        
    Example
    -------
    >>> temp = fa['S001TEMPERATURE']
    >>> print(temp.shape)  # (480, 480)
    >>> print(temp.min(), temp.max())
    >>> temp.plot()
    """
    
    def __init__(self, 
                 name: str,
                 data: np.ndarray,
                 lon: np.ndarray,
                 lat: np.ndarray,
                 attrs: Optional[Dict] = None):
        self.name = name
        self._data = data
        self._lon = lon
        self._lat = lat
        self.attrs = attrs or {}
        self.plot = PlotAccessor(self)
    
    @property
    def data(self) -> np.ndarray:
        """The data values as numpy array."""
        return self._data
    
    @property
    def values(self) -> np.ndarray:
        """Alias for data (xarray compatibility)."""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data."""
        return self._data.shape
    
    @property
    def dtype(self):
        """Data type."""
        return self._data.dtype
    
    @property
    def lon(self) -> np.ndarray:
        """2D longitude coordinates."""
        return self._lon
    
    @property
    def lat(self) -> np.ndarray:
        """2D latitude coordinates."""
        return self._lat
    
    def min(self) -> float:
        """Minimum value."""
        return float(np.nanmin(self._data))
    
    def max(self) -> float:
        """Maximum value."""
        return float(np.nanmax(self._data))
    
    def mean(self) -> float:
        """Mean value."""
        return float(np.nanmean(self._data))
    
    def std(self) -> float:
        """Standard deviation."""
        return float(np.nanstd(self._data))
    
    def to_xarray(self) -> xr.DataArray:
        """
        Convert to xarray.DataArray.
        
        Returns
        -------
        xarray.DataArray
            DataArray with lat/lon coordinates
        """
        return xr.DataArray(
            self._data,
            dims=['y', 'x'],
            coords={'lat': (['y', 'x'], self._lat),
                    'lon': (['y', 'x'], self._lon)},
            name=self.name,
            attrs=self.attrs
        )
    
    def __repr__(self) -> str:
        return (f"FAVariable: {self.name}\n"
                f"  Shape: {self.shape}\n"
                f"  Range: [{self.min():.4g}, {self.max():.4g}]\n"
                f"  Mean: {self.mean():.4g}")
    
    def __array__(self) -> np.ndarray:
        """Support numpy array conversion."""
        return self._data


class FADataset:
    """
    An FA file as a dataset of variables.
    
    Provides an xarray-like interface for accessing variables,
    with easy conversion to xarray.Dataset and NetCDF export.
    
    Parameters
    ----------
    filepath : str
        Path to the FA file
        
    Attributes
    ----------
    filepath : str
        Path to the source file
    variables : list
        List of variable names
    geometry : FAGeometry
        Grid geometry information
        
    Example
    -------
    >>> fa = FADataset('/path/to/file.fa')
    >>> print(fa.variables[:10])  # First 10 variables
    >>> 
    >>> # Access a variable
    >>> temp = fa['S001TEMPERATURE']
    >>> temp.plot()
    >>> 
    >>> # Select multiple variables
    >>> temps = fa.select('S*TEMPERATURE')
    >>> 
    >>> # Convert to xarray
    >>> ds = fa.to_xarray()
    >>> 
    >>> # Export to NetCDF
    >>> fa.to_netcdf('output.nc')
    """
    
    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self._reader = FAReader(self.filepath)
        self._cache: Dict[str, np.ndarray] = {}
        self._loaded_all = False
    
    def close(self):
        """Close the file."""
        self._reader.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    @property
    def variables(self) -> List[str]:
        """List of all variable names."""
        return self._reader.fields
    
    @property
    def nvars(self) -> int:
        """Number of variables."""
        return len(self.variables)
    
    @property
    def geometry(self) -> FAGeometry:
        """Grid geometry."""
        return self._reader.geometry
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (y, x)."""
        return self.geometry.shape
    
    @property
    def lon(self) -> np.ndarray:
        """2D longitude grid."""
        return self.geometry.lons
    
    @property
    def lat(self) -> np.ndarray:
        """2D latitude grid."""
        return self.geometry.lats
    
    def __len__(self) -> int:
        return len(self.variables)
    
    def __contains__(self, name: str) -> bool:
        return name in self.variables
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union[FAVariable, 'FADataset']:
        """
        Access variable(s) by name.
        
        Parameters
        ----------
        key : str or list of str
            Variable name or list of names
            
        Returns
        -------
        FAVariable or FADataset
            Single variable or subset dataset
        """
        if isinstance(key, str):
            return self._get_variable(key)
        elif isinstance(key, (list, tuple)):
            return self._subset(list(key))
        else:
            raise TypeError(f"Key must be str or list, got {type(key)}")
    
    def _get_variable(self, name: str) -> FAVariable:
        """Get a single variable."""
        if name not in self._cache:
            self._cache[name] = self._reader.read_field(name)
        
        return FAVariable(
            name=name,
            data=self._cache[name],
            lon=self.geometry.lons,
            lat=self.geometry.lats
        )
    
    def _subset(self, names: List[str]) -> 'FADataset':
        """Create a subset with only the specified variables."""
        subset = FADatasetSubset(self, names)
        return subset
    
    def select(self, pattern: str) -> List[FAVariable]:
        """
        Select variables matching a pattern.
        
        Parameters
        ----------
        pattern : str
            Glob pattern (e.g., 'S*TEMPERATURE', 'SURF*')
            or regex pattern (if starts with '^')
            
        Returns
        -------
        list of FAVariable
            Matching variables
            
        Example
        -------
        >>> temps = fa.select('S*TEMPERATURE')  # All temperature levels
        >>> surf = fa.select('SURF*')  # All surface fields
        """
        if pattern.startswith('^'):
            # Regex pattern
            regex = re.compile(pattern)
            matches = [v for v in self.variables if regex.match(v)]
        else:
            # Glob pattern
            matches = fnmatch.filter(self.variables, pattern)
        
        return [self._get_variable(name) for name in matches]
    
    def select_levels(self, variable: str, levels: Optional[List[int]] = None) -> List[FAVariable]:
        """
        Select all levels of a 3D variable.
        
        Parameters
        ----------
        variable : str
            Base variable name (e.g., 'TEMPERATURE', 'WIND.U.PHYS')
        levels : list of int, optional
            Specific levels to select. If None, selects all.
            
        Returns
        -------
        list of FAVariable
            Variables for each level
            
        Example
        -------
        >>> temps = fa.select_levels('TEMPERATURE')
        >>> temps_10 = fa.select_levels('TEMPERATURE', levels=[1, 2, 3, 4, 5])
        """
        if levels is None:
            pattern = f'S*{variable}'
            return self.select(pattern)
        else:
            names = [f'S{level:03d}{variable}' for level in levels]
            return [self._get_variable(n) for n in names if n in self.variables]
    
    def stack_levels(self, variable: str, levels: Optional[List[int]] = None) -> np.ndarray:
        """
        Stack all levels of a variable into a 3D array.
        
        Parameters
        ----------
        variable : str
            Base variable name
        levels : list of int, optional
            Specific levels. If None, auto-detects.
            
        Returns
        -------
        np.ndarray
            3D array with shape (levels, y, x)
        """
        vars_list = self.select_levels(variable, levels)
        return np.stack([v.data for v in vars_list], axis=0)
    
    def load(self, progress: bool = False):
        """
        Load all variables into memory.
        
        Parameters
        ----------
        progress : bool
            Print progress
        """
        if not self._loaded_all:
            self._cache = self._reader.read_all_fields(
                filter_shape=self.shape,
                progress=progress
            )
            self._loaded_all = True

    def _resolve_requested_fields(self, variables: List[str]) -> List[str]:
        """
        Resolve user-requested variable names to FA field names.

        Accepts:
        - Raw FA names (e.g., S001WIND.U.PHYS, SURFPREC.EAU.CON)
        - Normalized names with dots replaced by underscores
          (e.g., SURFPREC_EAU_CON)
        - Stacked 3D base names shown in xarray output
          (e.g., WIND_U_PHYS, WIND.U.PHYS)

        Returns
        -------
        list of str
            Resolved FA field names to read from file.
        """
        available_fields = list(self.variables)
        if not variables:
            return available_fields

        alias_to_fields: Dict[str, List[str]] = {}

        def add_alias(alias: str, fields: List[str]):
            if alias not in alias_to_fields:
                alias_to_fields[alias] = []
            for field in fields:
                if field not in alias_to_fields[alias]:
                    alias_to_fields[alias].append(field)

        # Aliases for raw fields (with and without dot normalization)
        for field in available_fields:
            add_alias(field, [field])
            add_alias(field.replace('.', '_'), [field])

        # Aliases for stacked 3D base names (e.g., WIND_U_PHYS)
        level_groups = detect_3d_fields(available_fields)
        for base_name, group_info in level_groups.items():
            group_fields = [name for _, name in group_info['levels']]
            add_alias(base_name, group_fields)
            add_alias(base_name.replace('.', '_'), group_fields)

        resolved: List[str] = []
        missing: List[str] = []

        for requested in variables:
            matches = alias_to_fields.get(requested)

            # Fallbacks for mixed input forms
            if not matches and '_' in requested:
                matches = alias_to_fields.get(requested.replace('_', '.'))
            if not matches and '.' in requested:
                matches = alias_to_fields.get(requested.replace('.', '_'))

            if not matches:
                missing.append(requested)
                continue

            for field in matches:
                if field not in resolved:
                    resolved.append(field)

        if missing:
            missing_str = ', '.join(missing)
            raise KeyError(f"Variable(s) not found in FA file: {missing_str}")

        return resolved
    
    def to_xarray(self, 
                  variables: Optional[List[str]] = None,
                  stack_levels: bool = True,
                  levels: Optional[List[int]] = None,
                  progress: bool = False) -> xr.Dataset:
        """
        Convert to xarray.Dataset.
        
        Parameters
        ----------
        variables : list of str, optional
            Variables to include. If None, includes all.
        stack_levels : bool, default True
            If True, automatically stack 3D fields (e.g., S001TEMPERATURE, S002TEMPERATURE)
            into single variables with a 'level' dimension.
        levels : list of int, optional
            Specific levels to include when stack_levels=True.
            If None, includes all available levels.
        progress : bool
            Print progress
            
        Returns
        -------
        xarray.Dataset
        """
        # Load all data first
        if variables is None:
            self.load(progress=progress)
            all_fields = list(self._cache.keys())
        else:
            all_fields = self._resolve_requested_fields(variables)
            for name in all_fields:
                if name not in self._cache:
                    self._cache[name] = self._reader.read_field(name)
        
        data_vars = {}
        level_coords = {}  # Store level coordinates for each type
        
        if stack_levels:
            # Detect 3D fields and stack them
            level_groups = detect_3d_fields(all_fields)
            processed_fields = set()
            
            if progress:
                n_model = sum(1 for v in level_groups.values() if v['type'] == 'model')
                n_pressure = sum(1 for v in level_groups.values() if v['type'] == 'pressure')
                print(f"  Detected {n_model} model-level + {n_pressure} pressure-level 3D variables")
            
            for base_name, group_info in level_groups.items():
                level_list = group_info['levels']
                level_type = group_info['type']
                level_units = group_info['units']
                level_positive = group_info['positive']
                
                # Filter levels if specified
                if levels is not None:
                    level_list = [(lvl, name) for lvl, name in level_list if lvl in levels]
                
                if not level_list:
                    continue
                
                # Stack the levels
                level_nums = [lvl for lvl, _ in level_list]
                field_names = [name for _, name in level_list]
                
                # Make sure all fields are in cache
                for name in field_names:
                    if name not in self._cache:
                        self._cache[name] = self._reader.read_field(name)
                
                # Stack into 3D array
                stacked = np.stack([self._cache[name] for name in field_names], axis=0)
                safe_name = base_name.replace('.', '_')
                
                # Determine dimension name based on level type
                if level_type == 'model':
                    dim_name = 'level'
                else:
                    dim_name = 'pressure'
                
                data_vars[safe_name] = (
                    [dim_name, 'y', 'x'], 
                    stacked, 
                    {
                        'level_values': level_nums, 
                        'level_type': level_type,
                        'original_fields': field_names,
                    }
                )
                
                # Store coordinate info for this level type
                if dim_name not in level_coords:
                    # Use sequential indices (0, 1, 2, ..., n-1) for the coordinate
                    # This makes sel(level=0) select the surface, sel(level=n-1) select model top
                    # The original FA level numbers are preserved in variable attributes
                    level_coords[dim_name] = {
                        'values': np.arange(len(level_nums), dtype=np.int32),
                        'attrs': {
                            'long_name': 'model level index' if level_type == 'model' else 'pressure level index',
                            'units': '1',
                            'positive': level_positive,
                            'description': 'Sequential level index: 0=surface, {}=model top'.format(len(level_nums)-1) if level_type == 'model' else 'Pressure level index',
                        }
                    }
                
                # Mark these fields as processed
                processed_fields.update(field_names)
            
            # Add remaining 2D fields (surface fields)
            for name in all_fields:
                if name not in processed_fields and name in self._cache:
                    safe_name = name.replace('.', '_')
                    data_vars[safe_name] = (['y', 'x'], self._cache[name])
        else:
            # Original behavior: all fields as 2D
            for name in all_fields:
                if name not in self._cache:
                    self._cache[name] = self._reader.read_field(name)
                safe_name = name.replace('.', '_')
                data_vars[safe_name] = (['y', 'x'], self._cache[name])
        
        # Build coordinates
        coords = {
            'lat': (['y', 'x'], self.lat),
            'lon': (['y', 'x'], self.lon)
        }

        # Add level coordinates from what we detected
        for dim_name, coord_info in level_coords.items():
            coords[dim_name] = coord_info['values']

        # Get time validity info
        validity = self._reader.get_validity()
        valid_time = validity['valid_time']
        base_time = validity['base_time']
        lead_time = validity['lead_time']

        # Create dataset (without time dim yet)
        ds = xr.Dataset(
            data_vars,
            coords=coords,
            attrs={
                'source': self.filepath,
                'Conventions': 'CF-1.8',
            }
        )

        # Attach per-field catalog metadata (long_name, units, ...).
        from .fa_metadata import apply_metadata_to_dataset
        ds = apply_metadata_to_dataset(ds)

        # Add CF-compliant attributes to level coordinates
        for dim_name, coord_info in level_coords.items():
            if dim_name in ds.coords:
                ds[dim_name].attrs = coord_info['attrs']
        
        # Add time dimension to all variables
        if valid_time is not None:
            # Expand all data variables to include time dimension at axis 0
            ds = ds.expand_dims(dim={'time': 1}, axis=0)
            
            # Assign the actual time coordinate value  
            # Use pandas Timestamp for proper CF encoding
            import pandas as pd
            time_value = pd.Timestamp(str(valid_time))
            ds = ds.assign_coords(time=[time_value])
            
            # Add CF-compliant time coordinate attributes for ncview compatibility
            ds['time'].attrs = {
                'long_name': 'valid time',
                'standard_name': 'time',
            }
            
            # Encode time for NetCDF (ncview needs this)
            ds['time'].encoding = {
                'units': 'hours since 1970-01-01',
                'calendar': 'proleptic_gregorian',
                'dtype': 'float64',
            }
            
            # Store base_time and lead_time as attributes
            if base_time is not None:
                ds.attrs['base_time'] = str(base_time)
            if lead_time is not None:
                ds.attrs['lead_time'] = str(lead_time)
        
        return _enable_lonlat_nearest_sel(ds)
    
    def to_xarray_lazy(self, 
                       variables: Optional[List[str]] = None,
                       stack_levels: bool = True) -> xr.Dataset:
        """
        Convert to xarray.Dataset using lazy loading (Dask).
        
        This avoids loading any data into memory until accessed.
        Essential for working with large datasets or archives.
        
        Parameters
        ----------
        variables : list of str, optional
            Variables to include. If None, includes all.
        stack_levels : bool, default True
            If True, automatically stack 3D fields.
            
        Returns
        -------
        xarray.Dataset
            Dataset with lazy dask arrays
        """
        if not HAS_DASK:
            raise ImportError("Dask is required for lazy loading")
            
        # Get list of variables without loading data
        all_fields = self._resolve_requested_fields(variables) if variables else list(self.variables)
        
        # Get shape from geometry
        shape = self.shape
        
        data_vars = {}
        level_coords = {}
        processed_fields = set()
        
        if stack_levels:
            level_groups = detect_3d_fields(all_fields)
            
            for base_name, group_info in level_groups.items():
                level_list = group_info['levels']
                level_type = group_info['type']
                level_units = group_info['units']
                level_positive = group_info['positive']
                
                if not level_list:
                    continue
                
                level_nums = [lvl for lvl, _ in level_list]
                field_names = [name for _, name in level_list]
                
                # Create lazy array for each level
                lazy_levels = []
                for field_name in field_names:
                    delayed_data = read_field_delayed(self.filepath, field_name)
                    lazy_arr = da.from_delayed(
                        delayed_data, 
                        shape=shape, 
                        dtype=np.float64
                    )
                    lazy_levels.append(lazy_arr)
                
                # Stack into 3D lazy array
                stacked = da.stack(lazy_levels, axis=0)
                safe_name = base_name.replace('.', '_')
                
                # Determine dimension name
                dim_name = 'level' if level_type == 'model' else 'pressure'
                
                data_vars[safe_name] = (
                    [dim_name, 'y', 'x'], 
                    stacked, 
                    {
                        'level_values': level_nums, 
                        'level_type': level_type,
                        'original_fields': field_names,
                    }
                )
                
                # Store coordinate info
                if dim_name not in level_coords:
                    level_coords[dim_name] = {
                        'values': np.array(level_nums, dtype=np.int32),
                        'attrs': {
                            'long_name': 'model level' if level_type == 'model' else 'pressure',
                            'units': level_units,
                            'positive': level_positive,
                        }
                    }
                
                processed_fields.update(field_names)
            
            # Add remaining 2D fields as lazy arrays
            for name in all_fields:
                if name not in processed_fields:
                    delayed_data = read_field_delayed(self.filepath, name)
                    lazy_arr = da.from_delayed(
                        delayed_data, 
                        shape=shape, 
                        dtype=np.float64
                    )
                    safe_name = name.replace('.', '_')
                    data_vars[safe_name] = (['y', 'x'], lazy_arr)
        else:
            # No stacking - all 2D lazy arrays
            for name in all_fields:
                delayed_data = read_field_delayed(self.filepath, name)
                lazy_arr = da.from_delayed(
                    delayed_data, 
                    shape=shape, 
                    dtype=np.float64
                )
                safe_name = name.replace('.', '_')
                data_vars[safe_name] = (['y', 'x'], lazy_arr)

        # Build coordinates
        coords = {
            'lat': (['y', 'x'], self.lat),
            'lon': (['y', 'x'], self.lon)
        }
        
        # Add level coordinates
        for dim_name, coord_info in level_coords.items():
            coords[dim_name] = coord_info['values']
            
        # Get time validity info
        validity = self._reader.get_validity()
        valid_time = validity['valid_time']
        
        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords=coords,
            attrs={
                'source': self.filepath,
                'Conventions': 'CF-1.8',
            }
        )
        
        # Add CF-compliant attributes to level coordinates
        for dim_name, coord_info in level_coords.items():
            if dim_name in ds.coords:
                ds[dim_name].attrs = coord_info['attrs']
        
        # Add time dimension
        if valid_time is not None:
            ds = ds.expand_dims(dim={'time': 1}, axis=0)
            import pandas as pd
            time_value = pd.Timestamp(str(valid_time))
            ds = ds.assign_coords(time=[time_value])
            ds['time'].attrs = {
                'long_name': 'valid time',
                'standard_name': 'time',
            }
            
        return _enable_lonlat_nearest_sel(ds)
    
    def to_netcdf(self,
                  output: str,
                  variables: Optional[List[str]] = None,
                  stack_levels: bool = True,
                  levels: Optional[List[int]] = None,
                  compress: Optional[str] = None,
                  compress_level: int = 4,
                  progress: bool = True):
        """
        Export to NetCDF file.
        
        Parameters
        ----------
        output : str
            Output file path
        variables : list of str, optional
            Variables to include. If None, includes all.
        stack_levels : bool, default True
            If True, automatically stack 3D fields into variables with 'level' dimension.
            If False, each level is saved as separate 2D variable.
        levels : list of int, optional
            Specific levels to include (e.g., [1, 2, 3] for first 3 levels).
            Only used when stack_levels=True.
        compress : str, optional
            Compression type: 'zlib' or None
        compress_level : int
            Compression level (1-9, only for zlib)
        progress : bool
            Print progress
            
        Example
        -------
        >>> fa.to_netcdf('output.nc')  # Auto-stacks 3D fields (default)
        >>> fa.to_netcdf('output.nc', stack_levels=False)  # Keep all 2D
        >>> fa.to_netcdf('output.nc', levels=[1, 10, 20])  # Only specific levels
        """
        import time
        start = time.time()
        
        if progress:
            print(f"Converting {self.filepath} to NetCDF...")
            if stack_levels:
                print(f"  Mode: 3D stacking enabled (levels will be combined)")
        
        ds = self.to_xarray(variables=variables, stack_levels=stack_levels, 
                           levels=levels, progress=progress)
        
        encoding = None
        if compress == 'zlib':
            encoding = {
                var: {'zlib': True, 'complevel': compress_level}
                for var in ds.data_vars
            }
        
        # Count 2D and 3D variables (both 'level' and 'pressure' dims are 3D)
        n_3d = sum(1 for v in ds.data_vars.values() if 'level' in v.dims or 'pressure' in v.dims)
        n_2d = len(ds.data_vars) - n_3d
        
        if progress:
            print(f"  Writing {n_3d} 3D + {n_2d} 2D variables to {output}...")
        
        ds.to_netcdf(output, encoding=encoding)
        
        if progress:
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")

    def to_fa(
        self,
        output: str,
        variables: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        """
        Write selected data to a new FA file using this file as template.

        The native writer replaces fields in a copy of the template. It supports
        raw, legacy KNGRIB=1/2, and GRIB_API gridpoint fields. LAM spectral
        fields can be written from gridpoint data or coefficient arrays. Global
        spectral template writes require coefficient arrays.
        """
        ds = self.to_xarray(variables=variables, stack_levels=True, progress=False)
        write_fa(ds, output, template=self.filepath, variables=None, overwrite=overwrite)
    
    def info(self) -> str:
        """Return summary information about the dataset."""
        return (f"FADataset: {self.filepath}\n"
                f"  Variables: {self.nvars}\n"
                f"  Grid: {self.shape[1]} x {self.shape[0]} ({self.geometry.name})\n"
                f"  Bounds: lon=[{self.lon.min():.2f}, {self.lon.max():.2f}], "
                f"lat=[{self.lat.min():.2f}, {self.lat.max():.2f}]")
    
    def __repr__(self) -> str:
        return self.info()


class FADatasetSubset:
    """A subset of an FADataset with limited variables."""
    
    def __init__(self, parent: FADataset, variables: List[str]):
        self._parent = parent
        self._variables = [v for v in variables if v in parent.variables]
    
    @property
    def variables(self) -> List[str]:
        return self._variables
    
    def __getitem__(self, key: str) -> FAVariable:
        if key in self._variables:
            return self._parent._get_variable(key)
        raise KeyError(f"Variable {key} not in subset")
    
    def to_xarray(self) -> xr.Dataset:
        return self._parent.to_xarray(variables=self._variables)
    
    def to_netcdf(self, output: str, **kwargs):
        self._parent.to_netcdf(output, variables=self._variables, **kwargs)

    def to_fa(self, output: str, overwrite: bool = False):
        self._parent.to_fa(output, variables=self._variables, overwrite=overwrite)


def open_fa(filepath: str) -> FADataset:
    """
    Open an FA file.
    
    This is the main entry point for working with FA files.
    
    Parameters
    ----------
    filepath : str
        Path to the FA file
        
    Returns
    -------
    FADataset
        Dataset object for accessing the file
        
    Example
    -------
    >>> import faxarray as fx
    >>> fa = fx.open_fa('/path/to/pfABOFABOF+0001')
    >>> print(fa)
    >>> temp = fa['S001TEMPERATURE']
    >>> temp.plot()
    """
    return FADataset(filepath)


def write_fa(
    ds: xr.Dataset,
    output: str,
    template: str,
    variables: Optional[List[str]] = None,
    overwrite: bool = False,
):
    """
    Write an xarray Dataset to FA using an existing FA file as template.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables to write.
    output : str
        Output FA path.
    template : str
        Existing FA file that supplies geometry, validity, field names, and
        article layout.
    variables : list of str, optional
        Dataset variables to write. If omitted, writes all data variables.
    overwrite : bool
        If True, replace an existing output file.
    """
    from .backends.native_fa import write_fa as native_write_fa

    native_write_fa(ds, output, template=template, variables=variables, overwrite=overwrite)


def create_fa_from_dataset(
    ds: xr.Dataset,
    output: str,
    geometry: Optional[object] = None,
    validity: Optional[object] = None,
    vertical: Optional[object] = None,
    variables: Optional[List[str]] = None,
    overwrite: bool = False,
) -> None:
    """Create a brand-new FA file from an xarray Dataset (no template).

    This is the no-template counterpart to :func:`write_fa`. The
    ``geometry`` argument selects which header articles get written.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset. Each selected variable must be 2D (regular
        lon/lat) or 1D (global Gauss flat array). Singleton ``time``
        dimensions are squeezed.
    output : str
        Path of the FA file to create.
    geometry : FARegularLonLatGeometry or FAGlobalGaussGeometry
        Required. Use :class:`faxarray.FARegularLonLatGeometry` or
        :class:`faxarray.FAGlobalGaussGeometry`. If omitted, a regular
        lon/lat geometry is inferred from ``ds.lon`` / ``ds.lat`` and
        the dataset's ``y, x`` dimensions.
    validity : FAValidityInput, optional
        Validity to encode. Defaults to a placeholder analysis date.
    vertical : FAVerticalInput, optional
        Hybrid vertical coordinate (defaults to a single layer).
    variables : list of str, optional
        Subset of ``ds`` variables to write. If omitted, writes all data
        variables (excluding coordinate-like ones).
    overwrite : bool
        If True, overwrite an existing output file.
    """

    from pathlib import Path
    from .backends import (
        FAFieldData,
        FARegularLonLatGeometry,
        FAGlobalGaussGeometry,
        FAValidityInput,
        FAVerticalInput,
        create_fa_file,
    )

    output_path = Path(output)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        output_path.unlink()

    selected = list(variables) if variables else list(ds.data_vars)

    if geometry is None:
        geometry = _infer_regular_lonlat_geometry(ds)

    fields: List[FAFieldData] = []
    for name in selected:
        if name not in ds:
            raise KeyError(f"variable not found in dataset: {name}")
        array = ds[name]
        # Drop the singleton time dim that to_xarray() typically adds.
        if "time" in array.dims and array.sizes["time"] == 1:
            array = array.isel(time=0)
        values = np.asarray(array.values)
        # If user kept dots replaced by underscores in xarray names, restore them.
        fa_name = name if "." in name else name.replace("_", ".")
        if "." in name:
            fa_name = name
        fields.append(FAFieldData(name=fa_name, values=values))

    create_fa_file(
        str(output_path),
        geometry=geometry,
        fields=fields,
        validity=validity,
        vertical=vertical,
    )


def _infer_regular_lonlat_geometry(ds: xr.Dataset):
    """Best-effort inference of FARegularLonLatGeometry from a Dataset."""

    from .backends import FARegularLonLatGeometry

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError(
            "create_fa_from_dataset() needs an explicit `geometry=` when "
            "the dataset has no `lon`/`lat` coordinates"
        )
    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)
    if lon.ndim != 2 or lat.ndim != 2 or lon.shape != lat.shape:
        raise ValueError(
            "regular lon/lat inference expects 2D `lon` and `lat` of equal shape"
        )
    ny, nx = lon.shape
    dx_row = np.diff(lon[0])
    dy_col = np.diff(lat[:, 0])
    if dx_row.size == 0 or dy_col.size == 0:
        raise ValueError("cannot infer dx/dy from a degenerate lon/lat grid")
    if not np.allclose(dx_row, dx_row[0], atol=1e-6) or not np.allclose(dy_col, dy_col[0], atol=1e-6):
        raise ValueError(
            "lon/lat grid is not regular; pass an explicit geometry= argument"
        )
    return FARegularLonLatGeometry(
        nx=nx,
        ny=ny,
        lon0=float(lon[ny // 2, nx // 2]),
        lat0=float(lat[ny // 2, nx // 2]),
        dx=float(dx_row[0]),
        dy=float(dy_col[0]),
    )
