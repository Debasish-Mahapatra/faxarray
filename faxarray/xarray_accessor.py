"""
Custom xarray accessor for FA data plotting and analysis.

Registers a .fa accessor on xarray DataArrays and Datasets that provides:
- Automatic lat/lon plotting when data has those coordinates
- Subdomain extraction
- Vertical profile extraction
- Wind field helpers
- Animation creation
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union

from .fa_metadata import PREDEFINED_REGIONS


@xr.register_dataarray_accessor("fa")
class FADataArrayAccessor:
    """
    xarray DataArray accessor for FA-specific operations.
    
    Provides plotting, profile extraction, and analysis methods.
    Access via ds['variable'].fa.plot()
    
    Example
    -------
    >>> ds = fx.open_dataset('file.fa')
    >>> ds['TEMPERATURE'].sel(level=67).fa.plot()  # Uses lat/lon automatically
    >>> profile = ds['TEMPERATURE'].fa.extract_profile(lon=2.35, lat=48.85)
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    # =========================================================================
    # Plotting Methods
    # =========================================================================
    
    def plot(self, 
             ax: Optional[plt.Axes] = None,
             figsize: Optional[tuple] = None,
             cmap: str = 'viridis',
             add_colorbar: bool = True,
             **kwargs) -> plt.Axes:
        """
        Plot the DataArray using lat/lon coordinates if available.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size if creating new axes
        cmap : str
            Colormap
        add_colorbar : bool
            Whether to add colorbar
        **kwargs
            Additional arguments passed to pcolormesh
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        da = self._obj.squeeze()
        
        # Check if we have lat/lon coordinates
        has_latlon = 'lat' in da.coords and 'lon' in da.coords
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        if has_latlon:
            # Use lat/lon for plotting
            lon = da.coords['lon'].values
            lat = da.coords['lat'].values
            mesh = ax.pcolormesh(lon, lat, da.values, cmap=cmap, **kwargs)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            # Fallback to xarray's default
            mesh = ax.pcolormesh(da.values, cmap=cmap, **kwargs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        if add_colorbar:
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
            if da.name:
                cbar.set_label(da.name)
        
        ax.set_title(da.name or 'Data')
        plt.tight_layout()
        
        return ax
    
    def contourf(self, 
                 levels: int = 20,
                 ax: Optional[plt.Axes] = None,
                 figsize: Optional[tuple] = None,
                 cmap: str = 'viridis',
                 add_colorbar: bool = True,
                 **kwargs) -> plt.Axes:
        """
        Plot filled contours using lat/lon coordinates if available.
        """
        da = self._obj.squeeze()
        has_latlon = 'lat' in da.coords and 'lon' in da.coords
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        if has_latlon:
            lon = da.coords['lon'].values
            lat = da.coords['lat'].values
            cf = ax.contourf(lon, lat, da.values, levels=levels, cmap=cmap, **kwargs)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            cf = ax.contourf(da.values, levels=levels, cmap=cmap, **kwargs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        if add_colorbar:
            cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
            if da.name:
                cbar.set_label(da.name)
        
        ax.set_title(da.name or 'Data')
        plt.tight_layout()
        
        return ax
    
    def pcolormesh(self, *args, **kwargs):
        """Alias for plot()."""
        return self.plot(*args, **kwargs)
    
    def contour(self,
                levels: int = 10,
                ax: Optional[plt.Axes] = None,
                figsize: Optional[tuple] = None,
                colors: str = 'black',
                **kwargs) -> plt.Axes:
        """
        Plot contour lines using lat/lon coordinates if available.
        """
        da = self._obj.squeeze()
        has_latlon = 'lat' in da.coords and 'lon' in da.coords
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        if has_latlon:
            lon = da.coords['lon'].values
            lat = da.coords['lat'].values
            cs = ax.contour(lon, lat, da.values, levels=levels, colors=colors, **kwargs)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            cs = ax.contour(da.values, levels=levels, colors=colors, **kwargs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_title(da.name or 'Data')
        plt.tight_layout()
        
        return ax

    def imshow(self,
               ax: Optional[plt.Axes] = None,
               figsize: Optional[tuple] = None,
               cmap: str = 'viridis',
               add_colorbar: bool = True,
               origin: str = 'lower',
               **kwargs) -> plt.Axes:
        """
        Plot using imshow (fast, no geographic coords).
        """
        da = self._obj.squeeze()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            
        im = ax.imshow(da.values, cmap=cmap, origin=origin, aspect='auto', **kwargs)
        
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            if da.name:
                cbar.set_label(da.name)
        
        ax.set_title(da.name or 'Data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        
        return ax

    # =========================================================================
    # Vertical Profile Extraction
    # =========================================================================
    
    def extract_profile(self, 
                        lon: float, 
                        lat: float,
                        method: str = 'nearest') -> xr.DataArray:
        """
        Extract a vertical profile at a specific geographic location.
        
        Parameters
        ----------
        lon : float
            Longitude of the profile location
        lat : float
            Latitude of the profile location
        method : str
            Interpolation method: 'nearest' or 'linear'
            
        Returns
        -------
        xarray.DataArray
            1D profile with level as the coordinate
            
        Example
        -------
        >>> profile = ds['TEMPERATURE'].fa.extract_profile(lon=2.35, lat=48.85)
        >>> profile.plot()  # Vertical profile plot
        """
        da = self._obj
        
        # Check if we have level dimension
        if 'level' not in da.dims:
            raise ValueError("DataArray must have a 'level' dimension for profile extraction")
        
        # Check if we have lat/lon coordinates
        if 'lat' not in da.coords or 'lon' not in da.coords:
            raise ValueError("DataArray must have 'lat' and 'lon' coordinates")
        
        lon_vals = da.coords['lon'].values
        lat_vals = da.coords['lat'].values
        
        if method == 'nearest':
            # Find nearest grid point
            dist = np.sqrt((lon_vals - lon)**2 + (lat_vals - lat)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            # Extract profile at this point
            profile_data = da.isel(y=min_idx[0], x=min_idx[1])
            actual_lon = float(lon_vals[min_idx])
            actual_lat = float(lat_vals[min_idx])
            
        elif method == 'linear':
            # Bilinear interpolation using scipy
            from scipy.interpolate import griddata
            
            profile_data_list = []
            for level in da.coords['level'].values:
                level_data = da.sel(level=level).values
                points = np.column_stack([lon_vals.ravel(), lat_vals.ravel()])
                value = griddata(points, level_data.ravel(), (lon, lat), method='linear')
                profile_data_list.append(float(value))
            
            profile_data = xr.DataArray(
                profile_data_list,
                coords={'level': da.coords['level']},
                dims=['level'],
                name=da.name
            )
            actual_lon = lon
            actual_lat = lat
        else:
            raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'linear'")
        
        # Add metadata
        profile_data.attrs = da.attrs.copy()
        profile_data.attrs['profile_lon'] = actual_lon
        profile_data.attrs['profile_lat'] = actual_lat
        profile_data.attrs['extraction_method'] = method
        
        return profile_data
    
    # =========================================================================
    # Subdomain Extraction
    # =========================================================================
    
    def extract_domain(self,
                       lon_range: Optional[Tuple[float, float]] = None,
                       lat_range: Optional[Tuple[float, float]] = None,
                       region: Optional[str] = None) -> xr.DataArray:
        """
        Extract a subdomain based on geographic bounds.
        
        Parameters
        ----------
        lon_range : tuple of (min, max), optional
            Longitude bounds
        lat_range : tuple of (min, max), optional
            Latitude bounds
        region : str, optional
            Predefined region name. Available: france, alps, pyrenees,
            britain, iberia, italy, germany, benelux, scandinavia, mediterranean
            
        Returns
        -------
        xarray.DataArray
            Subset of data within the specified bounds
            
        Example
        -------
        >>> # Extract France region
        >>> france = ds['TEMPERATURE'].fa.extract_domain(region='france')
        >>> 
        >>> # Or specify bounds manually
        >>> subset = ds['TEMPERATURE'].fa.extract_domain(
        ...     lon_range=(-5, 10), lat_range=(41, 52)
        ... )
        """
        da = self._obj
        
        # Get bounds from predefined region if specified
        if region is not None:
            region_lower = region.lower()
            if region_lower not in PREDEFINED_REGIONS:
                available = ', '.join(PREDEFINED_REGIONS.keys())
                raise ValueError(f"Unknown region: {region}. Available: {available}")
            
            region_def = PREDEFINED_REGIONS[region_lower]
            lon_range = region_def['lon_range']
            lat_range = region_def['lat_range']
        
        if lon_range is None or lat_range is None:
            raise ValueError("Must specify either region or both lon_range and lat_range")
        
        # Check for lat/lon coordinates
        if 'lat' not in da.coords or 'lon' not in da.coords:
            raise ValueError("DataArray must have 'lat' and 'lon' coordinates")
        
        lon = da.coords['lon']
        lat = da.coords['lat']
        
        # Create mask
        mask = (
            (lon >= lon_range[0]) & (lon <= lon_range[1]) &
            (lat >= lat_range[0]) & (lat <= lat_range[1])
        )
        
        # Apply mask - use where and then find bounding box to crop
        result = da.where(mask, drop=False)
        
        # Add metadata about extraction
        result.attrs = da.attrs.copy()
        result.attrs['subdomain_lon_range'] = lon_range
        result.attrs['subdomain_lat_range'] = lat_range
        if region:
            result.attrs['subdomain_region'] = region
        
        return result

    # =========================================================================
    # Animation
    # =========================================================================
    



@xr.register_dataset_accessor("fa")
class FADatasetAccessor:
    """
    xarray Dataset accessor for FA-specific operations.
    
    Provides wind field helpers and dataset-level operations.
    
    Example
    -------
    >>> ds = fx.open_dataset('file.fa')
    >>> ds['WIND_SPEED'] = ds.fa.wind_speed('CLSVENT.ZONAL', 'CLSVENT.MERIDIEN')
    >>> ds.fa.plot_wind('CLSVENT.ZONAL', 'CLSVENT.MERIDIEN')
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    # =========================================================================
    # Wind Field Helpers
    # =========================================================================
    

    

    

    
    # =========================================================================
    # Subdomain Extraction (Dataset level)
    # =========================================================================
    
    def extract_domain(self,
                       lon_range: Optional[Tuple[float, float]] = None,
                       lat_range: Optional[Tuple[float, float]] = None,
                       region: Optional[str] = None) -> xr.Dataset:
        """
        Extract a subdomain from the entire dataset.
        
        Parameters
        ----------
        lon_range : tuple of (min, max), optional
            Longitude bounds
        lat_range : tuple of (min, max), optional
            Latitude bounds
        region : str, optional
            Predefined region name. Available: france, alps, pyrenees,
            britain, iberia, italy, germany, benelux, scandinavia, mediterranean
            
        Returns
        -------
        xarray.Dataset
            Subset of data within the specified bounds
        """
        ds = self._obj
        
        # Get bounds from predefined region if specified
        if region is not None:
            region_lower = region.lower()
            if region_lower not in PREDEFINED_REGIONS:
                available = ', '.join(PREDEFINED_REGIONS.keys())
                raise ValueError(f"Unknown region: {region}. Available: {available}")
            
            region_def = PREDEFINED_REGIONS[region_lower]
            lon_range = region_def['lon_range']
            lat_range = region_def['lat_range']
        
        if lon_range is None or lat_range is None:
            raise ValueError("Must specify either region or both lon_range and lat_range")
        
        # Check for lat/lon coordinates
        if 'lat' not in ds.coords or 'lon' not in ds.coords:
            raise ValueError("Dataset must have 'lat' and 'lon' coordinates")
        
        lon = ds.coords['lon']
        lat = ds.coords['lat']
        
        # Create mask
        mask = (
            (lon >= lon_range[0]) & (lon <= lon_range[1]) &
            (lat >= lat_range[0]) & (lat <= lat_range[1])
        )
        
        # Apply mask to all data variables
        result = ds.where(mask, drop=False)
        
        # Add metadata
        result.attrs = ds.attrs.copy()
        result.attrs['subdomain_lon_range'] = lon_range
        result.attrs['subdomain_lat_range'] = lat_range
        if region:
            result.attrs['subdomain_region'] = region
        
        return result
