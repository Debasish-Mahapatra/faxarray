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
    
    def animate(self,
                dim: str = 'time',
                interval: int = 500,
                cmap: str = 'viridis',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                figsize: Optional[Tuple[float, float]] = None,
                title_format: Optional[str] = None) -> 'matplotlib.animation.FuncAnimation':
        """
        Create an animation over a dimension.
        
        Parameters
        ----------
        dim : str
            Dimension to animate over (default: 'time')
        interval : int
            Delay between frames in milliseconds
        cmap : str
            Colormap
        vmin, vmax : float, optional
            Color scale limits. If None, uses data min/max
        figsize : tuple, optional
            Figure size
        title_format : str, optional
            Format string for title, e.g. "Time: {}" 
            
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Call .save('file.gif') to save.
            
        Example
        -------
        >>> ani = ds['TEMPERATURE'].isel(level=67).fa.animate(dim='time')
        >>> ani.save('temp_evolution.gif', writer='pillow')
        """
        import matplotlib.animation as animation
        
        da = self._obj
        
        if dim not in da.dims:
            raise ValueError(f"Dimension '{dim}' not found. Available: {list(da.dims)}")
        
        has_latlon = 'lat' in da.coords and 'lon' in da.coords
        
        # Compute vmin/vmax from full data if not provided
        if vmin is None:
            vmin = float(da.min())
        if vmax is None:
            vmax = float(da.max())
        
        # Set up figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get first frame
        frame0 = da.isel({dim: 0}).squeeze()
        
        if has_latlon:
            lon = da.coords['lon'].values
            lat = da.coords['lat'].values
            mesh = ax.pcolormesh(lon, lat, frame0.values, 
                                 cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            mesh = ax.pcolormesh(frame0.values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        fig.colorbar(mesh, ax=ax, shrink=0.8, label=da.name or '')
        
        # Title
        dim_vals = da.coords[dim].values
        if title_format:
            title = ax.set_title(title_format.format(dim_vals[0]))
        else:
            title = ax.set_title(f'{da.name or "Data"} | {dim}={dim_vals[0]}')
        
        def update(frame_idx):
            frame = da.isel({dim: frame_idx}).squeeze()
            mesh.set_array(frame.values.ravel())
            if title_format:
                title.set_text(title_format.format(dim_vals[frame_idx]))
            else:
                title.set_text(f'{da.name or "Data"} | {dim}={dim_vals[frame_idx]}')
            return [mesh, title]
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(dim_vals),
            interval=interval, blit=False
        )
        
        plt.tight_layout()
        return ani


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
    
    def wind_speed(self, 
                   u_var: str, 
                   v_var: str,
                   name: str = 'WIND_SPEED') -> xr.DataArray:
        """
        Compute wind speed from U and V components.
        
        Parameters
        ----------
        u_var : str
            Name of U (zonal/eastward) wind variable
        v_var : str
            Name of V (meridional/northward) wind variable
        name : str
            Name for the resulting DataArray
            
        Returns
        -------
        xarray.DataArray
            Wind speed in same units as input
            
        Example
        -------
        >>> ds['WIND_SPEED'] = ds.fa.wind_speed('CLSVENT.ZONAL', 'CLSVENT.MERIDIEN')
        """
        ds = self._obj
        
        if u_var not in ds.data_vars:
            raise ValueError(f"Variable '{u_var}' not found in dataset")
        if v_var not in ds.data_vars:
            raise ValueError(f"Variable '{v_var}' not found in dataset")
        
        u = ds[u_var]
        v = ds[v_var]
        
        speed = np.sqrt(u**2 + v**2)
        speed.name = name
        speed.attrs = {
            'long_name': 'Wind Speed',
            'standard_name': 'wind_speed',
            'units': u.attrs.get('units', 'm s-1'),
        }
        
        return speed
    
    def wind_direction(self,
                       u_var: str,
                       v_var: str,
                       name: str = 'WIND_DIR',
                       convention: str = 'from') -> xr.DataArray:
        """
        Compute wind direction from U and V components.
        
        Parameters
        ----------
        u_var : str
            Name of U (zonal/eastward) wind variable
        v_var : str
            Name of V (meridional/northward) wind variable
        name : str
            Name for the resulting DataArray
        convention : str
            'from' (meteorological, direction wind comes from) or
            'to' (direction wind is blowing to)
            
        Returns
        -------
        xarray.DataArray
            Wind direction in degrees (0-360)
            
        Example
        -------
        >>> ds['WIND_DIR'] = ds.fa.wind_direction('CLSVENT.ZONAL', 'CLSVENT.MERIDIEN')
        """
        ds = self._obj
        
        if u_var not in ds.data_vars:
            raise ValueError(f"Variable '{u_var}' not found in dataset")
        if v_var not in ds.data_vars:
            raise ValueError(f"Variable '{v_var}' not found in dataset")
        
        u = ds[u_var]
        v = ds[v_var]
        
        if convention == 'from':
            # Meteorological convention: direction wind comes FROM
            direction = (270 - np.degrees(np.arctan2(v, u))) % 360
        elif convention == 'to':
            # Direction wind is blowing TO
            direction = (90 - np.degrees(np.arctan2(v, u))) % 360
        else:
            raise ValueError(f"Unknown convention: {convention}. Use 'from' or 'to'")
        
        direction.name = name
        direction.attrs = {
            'long_name': 'Wind Direction',
            'standard_name': 'wind_from_direction' if convention == 'from' else 'wind_to_direction',
            'units': 'degrees',
        }
        
        return direction
    
    def plot_wind(self,
                  u_var: str,
                  v_var: str,
                  style: str = 'quiver',
                  skip: int = 10,
                  ax: Optional[plt.Axes] = None,
                  figsize: Optional[Tuple[float, float]] = None,
                  scale: Optional[float] = None,
                  color: str = 'black',
                  **kwargs) -> plt.Axes:
        """
        Plot wind vectors as quiver or barbs.
        
        Parameters
        ----------
        u_var : str
            Name of U wind variable
        v_var : str
            Name of V wind variable
        style : str
            'quiver' for arrows, 'barbs' for wind barbs
        skip : int
            Plot every Nth grid point (to avoid overcrowding)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple, optional
            Figure size
        scale : float, optional
            Scale factor for arrows (quiver only)
        color : str
            Color for vectors
        **kwargs
            Additional arguments passed to quiver/barbs
            
        Returns
        -------
        matplotlib.axes.Axes
            
        Example
        -------
        >>> ds.fa.plot_wind('CLSVENT.ZONAL', 'CLSVENT.MERIDIEN', style='barbs', skip=5)
        """
        ds = self._obj
        
        if u_var not in ds.data_vars:
            raise ValueError(f"Variable '{u_var}' not found in dataset")
        if v_var not in ds.data_vars:
            raise ValueError(f"Variable '{v_var}' not found in dataset")
        
        u = ds[u_var].squeeze()
        v = ds[v_var].squeeze()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        has_latlon = 'lat' in ds.coords and 'lon' in ds.coords
        
        if has_latlon:
            lon = ds.coords['lon'].values[::skip, ::skip]
            lat = ds.coords['lat'].values[::skip, ::skip]
            x, y = lon, lat
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            ny, nx = u.shape[-2:]
            x = np.arange(0, nx, skip)
            y = np.arange(0, ny, skip)
            x, y = np.meshgrid(x, y)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        U = u.values[::skip, ::skip]
        V = v.values[::skip, ::skip]
        
        if style == 'quiver':
            quiver_kwargs = {'color': color}
            if scale is not None:
                quiver_kwargs['scale'] = scale
            quiver_kwargs.update(kwargs)
            ax.quiver(x, y, U, V, **quiver_kwargs)
        elif style == 'barbs':
            ax.barbs(x, y, U, V, color=color, **kwargs)
        else:
            raise ValueError(f"Unknown style: {style}. Use 'quiver' or 'barbs'")
        
        ax.set_title(f'Wind ({u_var}, {v_var})')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return ax
    
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
