"""
FA file reader using EPyGrAM backend.

This module wraps EPyGrAM's falfilfa4py for reading FA files,
providing a cleaner interface for the rest of the package.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


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
    Low-level FA file reader using EPyGrAM backend.
    
    This class handles the actual reading of FA files using EPyGrAM's
    Fortran-based reader (falfilfa4py). It provides a clean interface
    for extracting field data and metadata.
    
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
        self._epygram_initialized = False
        
        # Initialize EPyGrAM and open file
        self._open()
    
    def _init_epygram(self):
        """Initialize EPyGrAM environment (only once)."""
        if not self._epygram_initialized:
            import epygram
            epygram.init_env()
            self._epygram_initialized = True
    
    def _open(self):
        """Open the FA file."""
        self._init_epygram()
        import epygram
        self._resource = epygram.formats.resource(self.filepath, 'r')
    
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
        """Load geometry from the first valid field."""
        # Find a surface field to get geometry
        for fname in self.fields:
            if fname.startswith('SURF') or not fname.startswith('S0'):
                try:
                    f = self._resource.readfield(fname)
                    data = f.getdata()
                    if data.ndim == 2:
                        lons, lats = f.geometry.get_lonlat_grid()
                        proj_info = None
                        if hasattr(f.geometry, 'projection'):
                            proj_info = dict(f.geometry.projection)
                        return FAGeometry(
                            name=f.geometry.name,
                            shape=data.shape,
                            lons=lons,
                            lats=lats,
                            projection=proj_info
                        )
                except:
                    continue
        
        # Fallback: try any field
        for fname in self.fields[:10]:
            try:
                f = self._resource.readfield(fname)
                if hasattr(f, 'spectral') and f.spectral:
                    f.sp2gp()
                data = f.getdata()
                if data.ndim == 2:
                    lons, lats = f.geometry.get_lonlat_grid()
                    return FAGeometry(
                        name=f.geometry.name,
                        shape=data.shape,
                        lons=lons,
                        lats=lats
                    )
            except:
                continue
        
        raise RuntimeError("Could not determine geometry from file")
    
    def get_field_info(self, name: str) -> FAFieldInfo:
        """Get metadata about a field without loading data."""
        if name not in self._field_info:
            try:
                encoding = self._resource.fieldencoding(name)
                self._field_info[name] = FAFieldInfo(
                    name=name,
                    spectral=encoding.get('spectral', False)
                )
            except:
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
        import numpy as np
        
        # Read any field to get validity info
        for fname in self.fields[:10]:
            try:
                field = self._resource.readfield(fname)
                if hasattr(field, 'validity'):
                    valid_time = field.validity.get()
                    base_time = field.validity.getbasis()
                    lead_time = field.validity.term()
                    
                    return {
                        'valid_time': np.datetime64(valid_time),
                        'base_time': np.datetime64(base_time),
                        'lead_time': np.timedelta64(lead_time),
                    }
            except:
                continue
        
        # Fallback: no validity info available
        return {
            'valid_time': None,
            'base_time': None,
            'lead_time': None,
        }
    
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
        f = self._resource.readfield(name)
        
        if convert_spectral and hasattr(f, 'spectral') and f.spectral:
            f.sp2gp()
        
        return f.getdata()
    
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
