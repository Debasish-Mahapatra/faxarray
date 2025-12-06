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
            # Sort by level number (ascending: 1, 2, 3... where 1 = top)
            sorted_levels = sorted(levels, key=lambda x: x[0])
            result[base_name] = {
                'levels': sorted_levels,
                'type': 'model',
                'units': '1',
                'positive': 'down',  # Level 1 at top, increases downward
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
            all_fields = variables
            for name in variables:
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
                    level_coords[dim_name] = {
                        'values': np.array(level_nums, dtype=np.int32),
                        'attrs': {
                            'long_name': 'model level' if level_type == 'model' else 'pressure',
                            'units': level_units,
                            'positive': level_positive,
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
        
        # Add CF-compliant attributes to level coordinates
        for dim_name, coord_info in level_coords.items():
            if dim_name in ds.coords:
                ds[dim_name].attrs = coord_info['attrs']
        
        # Add time dimension to all variables
        if valid_time is not None:
            # Expand all data variables to include time dimension at axis 0
            ds = ds.expand_dims(dim={'time': 1}, axis=0)
            
            # Assign the actual time coordinate value
            ds = ds.assign_coords(time=[valid_time])
            
            # Add time coordinate attributes
            ds['time'].attrs = {
                'long_name': 'valid time',
                'standard_name': 'time',
            }
            
            # Store base_time and lead_time as attributes
            if base_time is not None:
                ds.attrs['base_time'] = str(base_time)
            if lead_time is not None:
                ds.attrs['lead_time'] = str(lead_time)
        
        return ds
    
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
