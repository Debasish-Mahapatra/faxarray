"""FA file reader using the native faxarray backend."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .backends.native_fa import NativeFAResource


@dataclass
class FAGeometry:
    """Represents the geometry/grid of an FA file."""
    name: str  # e.g., 'mercator', 'lambert', 'regular_lonlat'
    shape: Tuple[int, int]  # (y, x)
    lons: np.ndarray  # 2D array of longitudes
    lats: np.ndarray  # 2D array of latitudes
    projection: Optional[Dict[str, Any]] = None
    
    @property
    def nx(self) -> int:
        return self.shape[1]
    
    @property
    def ny(self) -> int:
        return self.shape[0]


@dataclass 
class FAFieldInfo:
    """Metadata about a field in an FA file."""
    name: str
    spectral: bool = False
    shape: Optional[Tuple[int, ...]] = None
    dtype: str = 'float64'


class FAReader:
    """
    Low-level FA file reader using faxarray's native backend.
    
    This class provides a clean interface for extracting field data and
    metadata. It no longer requires EPyGRAM at import or install time.
    
    Parameters
    ----------
    filepath : str
        Path to the FA file
        
    Example
    -------
    >>> reader = FAReader('/path/to/file.fa')
    >>> print(reader.fields)  # List of field names
    >>> data = reader.read_field('S001TEMPERATURE')
    >>> reader.close()
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._resource = None
        self._geometry: Optional[FAGeometry] = None
        self._fields: Optional[List[str]] = None
        self._field_info: Dict[str, FAFieldInfo] = {}
        self._open()
    
    def _open(self):
        """Open the FA file."""
        self._resource = NativeFAResource(self.filepath)
    
    def close(self):
        """Close the FA file."""
        if self._resource is not None:
            self._resource.close()
            self._resource = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    @property
    def fields(self) -> List[str]:
        """List of all field names in the file."""
        if self._fields is None:
            self._fields = self._resource.listfields()
        return self._fields
    
    @property
    def geometry(self) -> FAGeometry:
        """Get the geometry of the file (lazy loaded)."""
        if self._geometry is None:
            self._geometry = self._load_geometry()
        return self._geometry
    
    def _load_geometry(self) -> FAGeometry:
        """Load geometry from the FA header."""
        geometry = self._resource.geometry
        return FAGeometry(
            name=geometry.name,
            shape=geometry.shape,
            lons=geometry.lons,
            lats=geometry.lats,
            projection=geometry.projection,
        )
    
    def get_field_info(self, name: str) -> FAFieldInfo:
        """Get metadata about a field without loading data."""
        if name not in self._field_info:
            try:
                encoding = self._resource.fieldencoding(name)
                self._field_info[name] = FAFieldInfo(
                    name=name,
                    spectral=bool(encoding.get('spectral', False))
                )
            except Exception:
                self._field_info[name] = FAFieldInfo(name=name)
        return self._field_info[name]
    
    def get_validity(self) -> dict:
        """
        Extract time validity info from the FA file.
        
        Returns
        -------
        dict with keys:
            - valid_time: datetime, the valid/forecast time
            - base_time: datetime, the initialization/reference time
            - lead_time: timedelta, the forecast lead time
        """
        return self._resource.get_validity()
    
    def read_field(self, name: str, convert_spectral: bool = True) -> np.ndarray:
        """
        Read a single field from the file.
        
        Parameters
        ----------
        name : str
            Field name (e.g., 'S001TEMPERATURE', 'SURFTEMPERATURE')
        convert_spectral : bool
            If True, convert spectral fields to gridpoint
            
        Returns
        -------
        np.ndarray
            Field data as numpy array
        """
        return self._resource.readfield(name, convert_spectral=convert_spectral)
    
    def read_fields(self, names: List[str], 
                    convert_spectral: bool = True,
                    progress: bool = False) -> Dict[str, np.ndarray]:
        """
        Read multiple fields from the file.
        
        Parameters
        ----------
        names : list of str
            Field names to read
        convert_spectral : bool
            If True, convert spectral fields to gridpoint
        progress : bool
            If True, print progress
            
        Returns
        -------
        dict
            Dictionary mapping field names to numpy arrays
        """
        result = {}
        total = len(names)
        
        for i, name in enumerate(names):
            try:
                result[name] = self.read_field(name, convert_spectral)
                if progress and (i + 1) % 500 == 0:
                    print(f"  Read {i+1}/{total} fields...")
            except Exception as e:
                if progress:
                    print(f"  Warning: Could not read {name}: {e}")
        
        return result
    
    def read_all_fields(self, convert_spectral: bool = True,
                        filter_shape: Optional[Tuple[int, int]] = None,
                        progress: bool = False) -> Dict[str, np.ndarray]:
        """
        Read all fields from the file.
        
        Parameters
        ----------
        convert_spectral : bool
            If True, convert spectral fields to gridpoint
        filter_shape : tuple, optional
            Only return fields matching this shape
        progress : bool
            If True, print progress
            
        Returns
        -------
        dict
            Dictionary mapping field names to numpy arrays
        """
        if filter_shape is None:
            filter_shape = self.geometry.shape
        
        result = {}
        total = len(self.fields)
        
        for i, name in enumerate(self.fields):
            try:
                data = self.read_field(name, convert_spectral)
                if data.shape == filter_shape:
                    result[name] = data
            except:
                pass
            
            if progress and (i + 1) % 500 == 0:
                print(f"  Read {i+1}/{total} fields...")
        
        return result
