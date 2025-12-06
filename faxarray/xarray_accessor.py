"""
Custom xarray accessor for FA data plotting.

Registers a .fa accessor on xarray DataArrays that provides
automatic lat/lon plotting when data has those coordinates.
"""

import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional


@xr.register_dataarray_accessor("fa")
class FADataArrayAccessor:
    """
    xarray DataArray accessor for FA-specific plotting.
    
    Automatically uses lat/lon coordinates when plotting if available.
    Access via ds['variable'].fa.plot()
    
    Example
    -------
    >>> ds = fxr.open_dataset('file.fa')
    >>> ds['TEMPERATURE'].sel(level=67).fa.plot()  # Uses lat/lon automatically
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def plot(self, 
             ax: Optional[plt.Axes] = None,
             figsize: tuple = (10, 8),
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
                 figsize: tuple = (10, 8),
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
