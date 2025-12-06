#!/usr/bin/env python
"""
Validation test: Compare EPyGrAM and faxarray outputs.

This script:
1. Converts selected fields with EPyGrAM
2. Converts same fields with faxarray
3. Computes and plots differences
"""

import sys
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add parent dir to path for faxarray
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import epygram
import faxarray as fx

# Configuration
FA_FILE = '/home/dev/PROJECTS/proj-dat/pfABOFABOF+0001'
OUTPUT_DIR = '/home/dev/PROJECTS/Epygram-xarray/test'
FIELDS_TO_TEST = [
    'S001TEMPERATURE',
    'S010TEMPERATURE', 
    'S050TEMPERATURE',
    'SURFTEMPERATURE',
    'SURFPRESSION',
    'SURFGEOPOTEN',
]

def convert_with_epygram(fa_file, fields, output_dir):
    """Convert using EPyGrAM directly."""
    print("\n" + "="*60)
    print("STEP 1: Converting with EPyGrAM Python API")
    print("="*60)
    
    epygram.init_env()
    
    start = time.time()
    r = epygram.formats.resource(fa_file, 'r')
    
    data = {}
    lons = lats = None
    
    for fname in fields:
        print(f"  Reading {fname}...")
        f = r.readfield(fname)
        if hasattr(f, 'spectral') and f.spectral:
            f.sp2gp()
        data[fname] = f.getdata()
        if lons is None:
            lons, lats = f.geometry.get_lonlat_grid()
    
    r.close()
    read_time = time.time() - start
    
    # Create xarray dataset
    ds = xr.Dataset(
        {name.replace('.', '_'): (['y', 'x'], arr) for name, arr in data.items()},
        coords={'lat': (['y', 'x'], lats), 'lon': (['y', 'x'], lons)}
    )
    
    # Save
    output_file = os.path.join(output_dir, 'epygram_output.nc')
    ds.to_netcdf(output_file)
    total_time = time.time() - start
    
    print(f"\n  EPyGrAM time: {total_time:.2f}s")
    print(f"  Output: {output_file}")
    
    return output_file, ds


def convert_with_faxarray(fa_file, fields, output_dir):
    """Convert using faxarray."""
    print("\n" + "="*60)
    print("STEP 2: Converting with faxarray")
    print("="*60)
    
    start = time.time()
    fa = fx.open_fa(fa_file)
    
    # Get subset as xarray
    ds = fa.to_xarray(variables=fields, progress=False)
    
    # Save
    output_file = os.path.join(output_dir, 'faxarray_output.nc')
    ds.to_netcdf(output_file)
    total_time = time.time() - start
    
    print(f"\n  faxarray time: {total_time:.2f}s")
    print(f"  Output: {output_file}")
    
    fa.close()
    return output_file, ds


def compare_and_plot(ds_epygram, ds_faxarray, fields, output_dir):
    """Compare the two outputs and plot differences."""
    print("\n" + "="*60)
    print("STEP 3: Comparing outputs")
    print("="*60)
    
    # Normalize field names (EPyGrAM replaces . with _)
    fields_normalized = [f.replace('.', '_') for f in fields]
    
    # Create comparison plots
    n_fields = len(fields_normalized)
    fig, axes = plt.subplots(n_fields, 4, figsize=(16, 4*n_fields))
    
    if n_fields == 1:
        axes = axes.reshape(1, -1)
    
    all_close = True
    
    for i, fname in enumerate(fields_normalized):
        print(f"\n  Comparing {fname}...")
        
        epygram_data = ds_epygram[fname].values
        faxarray_data = ds_faxarray[fname].values
        
        # Compute difference
        diff = faxarray_data - epygram_data
        
        # Statistics
        max_abs_diff = np.nanmax(np.abs(diff))
        mean_diff = np.nanmean(diff)
        rmse = np.sqrt(np.nanmean(diff**2))
        
        print(f"    Max absolute difference: {max_abs_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        print(f"    RMSE: {rmse:.2e}")
        
        # Check if close
        if max_abs_diff > 1e-10:
            print(f"    ⚠️  WARNING: Non-zero difference detected!")
            all_close = False
        else:
            print(f"    ✓ Values match (diff < 1e-10)")
        
        # Plot
        lon = ds_epygram['lon'].values
        lat = ds_epygram['lat'].values
        
        # EPyGrAM
        ax = axes[i, 0]
        im = ax.pcolormesh(lon, lat, epygram_data, cmap='viridis')
        ax.set_title(f'EPyGrAM: {fname}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # faxarray
        ax = axes[i, 1]
        im = ax.pcolormesh(lon, lat, faxarray_data, cmap='viridis')
        ax.set_title(f'faxarray: {fname}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Difference
        ax = axes[i, 2]
        vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        if vmax == 0:
            vmax = 1e-10
        im = ax.pcolormesh(lon, lat, diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Difference (faxarray - EPyGrAM)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Histogram of differences
        ax = axes[i, 3]
        ax.hist(diff.flatten(), bins=50, edgecolor='black')
        ax.set_title(f'Difference histogram')
        ax.set_xlabel('Difference')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved comparison plot: {plot_file}")
    
    plt.close()
    
    return all_close


def main():
    print("="*60)
    print("VALIDATION TEST: EPyGrAM vs faxarray")
    print("="*60)
    print(f"FA file: {FA_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Fields to test: {FIELDS_TO_TEST}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert with both methods
    epygram_file, ds_epygram = convert_with_epygram(FA_FILE, FIELDS_TO_TEST, OUTPUT_DIR)
    faxarray_file, ds_faxarray = convert_with_faxarray(FA_FILE, FIELDS_TO_TEST, OUTPUT_DIR)
    
    # Compare and plot
    all_close = compare_and_plot(ds_epygram, ds_faxarray, FIELDS_TO_TEST, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if all_close:
        print("✓ All fields match! faxarray produces identical output to EPyGrAM.")
    else:
        print("⚠️ Some differences detected. Check the comparison plot.")
    
    print(f"\nOutput files:")
    print(f"  - {epygram_file}")
    print(f"  - {faxarray_file}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'comparison_plot.png')}")


if __name__ == '__main__':
    main()
