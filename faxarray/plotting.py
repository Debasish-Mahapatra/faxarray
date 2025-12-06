"""
Plotting functionality for FA data.

Provides xarray-style plot accessors for FAVariable objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import FAVariable


class PlotAccessor:
    """
    xarray-style plot accessor for FAVariable.
    
    Provides convenient plotting methods that can be accessed via
    variable.plot() or variable.plot.contourf(), etc.
    
    Example
    -------
    >>> temp = fa['S001TEMPERATURE']
    >>> temp.plot()  # Quick pcolormesh
    >>> temp.plot.contourf(levels=20)  # Filled contours
    >>> temp.plot.contour(colors='black')  # Line contours
    """
    
    def __init__(self, variable: 'FAVariable'):
        self._variable = variable
    
    def __call__(self, 
                 ax: Optional[plt.Axes] = None,
                 figsize: Tuple[int, int] = (10, 8),
                 cmap: str = 'viridis',
                 colorbar: bool = True,
                 title: Optional[str] = None,
                 **kwargs) -> plt.Axes:
        """
        Quick plot using pcolormesh.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : tuple
            Figure size if creating new figure
        cmap : str
            Colormap name
        colorbar : bool
            Whether to add colorbar
        title : str, optional
            Plot title. Defaults to variable name.
        **kwargs
            Additional arguments passed to pcolormesh
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        return self.pcolormesh(ax=ax, figsize=figsize, cmap=cmap, 
                               colorbar=colorbar, title=title, **kwargs)
    
    def _get_plot_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get lon, lat, data arrays for plotting."""
        var = self._variable
        return var.lon, var.lat, var.data
    
    def _setup_axes(self, ax: Optional[plt.Axes], 
                    figsize: Tuple[int, int],
                    projection: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """Create or get axes for plotting."""
        if ax is not None:
            fig = ax.get_figure()
            return fig, ax
        
        if projection:
            try:
                import cartopy.crs as ccrs
                fig, ax = plt.subplots(figsize=figsize, 
                                        subplot_kw={'projection': ccrs.PlateCarree()})
                ax.coastlines()
                # Only draw labels on bottom and left to avoid overlap with title
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
            except ImportError:
                fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        return fig, ax
    
    def _add_colorbar(self, fig: plt.Figure, ax: plt.Axes, 
                      mappable, label: Optional[str] = None):
        """Add colorbar to plot."""
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
        if label:
            cbar.set_label(label)
    
    def pcolormesh(self,
                   ax: Optional[plt.Axes] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   cmap: str = 'viridis',
                   colorbar: bool = True,
                   title: Optional[str] = None,
                   use_cartopy: bool = True,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   **kwargs) -> plt.Axes:
        """
        Plot using pcolormesh.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        colorbar : bool
            Add colorbar
        title : str, optional
            Plot title
        use_cartopy : bool
            Use cartopy for geographic projection
        vmin, vmax : float, optional
            Color scale limits
        **kwargs
            Passed to pcolormesh
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        lon, lat, data = self._get_plot_data()
        fig, ax = self._setup_axes(ax, figsize, projection=use_cartopy)
        
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, 
                                     vmin=vmin, vmax=vmax,
                                     transform=ccrs.PlateCarree(), **kwargs)
            except ImportError:
                mesh = ax.pcolormesh(lon, lat, data, cmap=cmap,
                                     vmin=vmin, vmax=vmax, **kwargs)
        else:
            mesh = ax.pcolormesh(lon, lat, data, cmap=cmap,
                                 vmin=vmin, vmax=vmax, **kwargs)
        
        if colorbar:
            self._add_colorbar(fig, ax, mesh, self._variable.name)
        
        ax.set_title(title or self._variable.name, pad=10)
        
        # Only set labels if not using cartopy (cartopy adds its own)
        if not use_cartopy:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        return ax
    
    def contourf(self,
                 levels: int = 20,
                 ax: Optional[plt.Axes] = None,
                 figsize: Tuple[int, int] = (10, 8),
                 cmap: str = 'viridis',
                 colorbar: bool = True,
                 title: Optional[str] = None,
                 use_cartopy: bool = True,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 **kwargs) -> plt.Axes:
        """
        Plot using filled contours.
        
        Parameters
        ----------
        levels : int or array-like
            Number of contour levels or explicit levels
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        colorbar : bool
            Add colorbar
        title : str, optional
            Plot title
        use_cartopy : bool
            Use cartopy for geographic projection
        vmin, vmax : float, optional
            Color scale limits
        **kwargs
            Passed to contourf
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        lon, lat, data = self._get_plot_data()
        fig, ax = self._setup_axes(ax, figsize, projection=use_cartopy)
        
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap,
                                 vmin=vmin, vmax=vmax,
                                 transform=ccrs.PlateCarree(), **kwargs)
            except ImportError:
                cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap,
                                 vmin=vmin, vmax=vmax, **kwargs)
        else:
            cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax, **kwargs)
        
        if colorbar:
            self._add_colorbar(fig, ax, cf, self._variable.name)
        
        ax.set_title(title or self._variable.name, pad=10)
        
        if not use_cartopy:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        return ax
    
    def contour(self,
                levels: int = 10,
                ax: Optional[plt.Axes] = None,
                figsize: Tuple[int, int] = (10, 8),
                colors: str = 'black',
                title: Optional[str] = None,
                use_cartopy: bool = True,
                clabel: bool = True,
                **kwargs) -> plt.Axes:
        """
        Plot using line contours.
        
        Parameters
        ----------
        levels : int or array-like
            Number of contour levels or explicit levels
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size
        colors : str
            Contour line color
        title : str, optional
            Plot title
        use_cartopy : bool
            Use cartopy for geographic projection
        clabel : bool
            Add contour labels
        **kwargs
            Passed to contour
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        lon, lat, data = self._get_plot_data()
        fig, ax = self._setup_axes(ax, figsize, projection=use_cartopy)
        
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                cs = ax.contour(lon, lat, data, levels=levels, colors=colors,
                                transform=ccrs.PlateCarree(), **kwargs)
            except ImportError:
                cs = ax.contour(lon, lat, data, levels=levels, colors=colors, **kwargs)
        else:
            cs = ax.contour(lon, lat, data, levels=levels, colors=colors, **kwargs)
        
        if clabel:
            ax.clabel(cs, inline=True, fontsize=8)
        
        ax.set_title(title or self._variable.name, pad=10)
        
        if not use_cartopy:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        return ax
    
    def imshow(self,
               ax: Optional[plt.Axes] = None,
               figsize: Tuple[int, int] = (10, 8),
               cmap: str = 'viridis',
               colorbar: bool = True,
               title: Optional[str] = None,
               origin: str = 'lower',
               **kwargs) -> plt.Axes:
        """
        Plot using imshow (fast, no geographic projection).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        colorbar : bool
            Add colorbar
        title : str, optional
            Plot title
        origin : str
            Origin position ('lower' or 'upper')
        **kwargs
            Passed to imshow
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        fig, ax = self._setup_axes(ax, figsize, projection=False)
        
        lon, lat, data = self._get_plot_data()
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        
        im = ax.imshow(data, cmap=cmap, origin=origin, extent=extent, 
                       aspect='auto', **kwargs)
        
        if colorbar:
            self._add_colorbar(fig, ax, im, self._variable.name)
        
        ax.set_title(title or self._variable.name)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        return ax
    
    def save(self, filename: str, dpi: int = 150, **kwargs):
        """
        Save plot to file.
        
        Parameters
        ----------
        filename : str
            Output filename (e.g., 'plot.png', 'plot.pdf')
        dpi : int
            Resolution
        **kwargs
            Additional arguments passed to savefig
        """
        self()  # Create plot
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        plt.close()


def plot_multiple(variables: list,
                  ncols: int = 2,
                  figsize: Optional[Tuple[int, int]] = None,
                  cmap: str = 'viridis',
                  use_cartopy: bool = True,
                  **kwargs) -> plt.Figure:
    """
    Plot multiple variables in a grid.
    
    Parameters
    ----------
    variables : list of FAVariable
        Variables to plot
    ncols : int
        Number of columns
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    cmap : str
        Colormap
    use_cartopy : bool
        Use cartopy projection
    **kwargs
        Passed to pcolormesh
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    nplots = len(variables)
    nrows = (nplots + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                     subplot_kw={'projection': ccrs.PlateCarree()})
        except ImportError:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        var.plot(ax=axes[i], cmap=cmap, use_cartopy=use_cartopy, **kwargs)
    
    # Hide empty subplots
    for i in range(nplots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig
