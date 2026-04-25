#!/usr/bin/env python
"""Validate faxarray on a local FA file, with optional EPYGRAM comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faxarray as fx
from faxarray.backends.native_fa import NativeFAResource


DEFAULT_FA_FILE = Path("/Users/dev/PROJECTS/test-data/pfABOFABOF+0012")
DEFAULT_OUTPUT_DIR = Path("/Users/dev/PROJECTS/test-data/faxarray_validation")
DEFAULT_FIELDS_FILE = Path(__file__).with_name("fields_to_compare.txt")
DEFAULT_EXTRA_FIELDS = ["SURFGEOPOTEN"]


def load_fields(fields_file: Path) -> List[str]:
    fields = []
    if fields_file.exists():
        fields.extend(
            line.strip()
            for line in fields_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    for field in DEFAULT_EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def classify_fields(fa_file: Path, fields: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    resource = NativeFAResource(str(fa_file))
    supported = []
    skipped = {}
    for field in fields:
        encoding = resource.fieldencoding(field)
        if not encoding.get("exists"):
            skipped[field] = "missing"
        elif encoding.get("ftype") != "H2D":
            skipped[field] = f"unsupported field type {encoding.get('ftype')}"
        else:
            supported.append(field)
    return supported, skipped


def convert_with_faxarray(fa_file: Path, fields: List[str], output_dir: Path) -> Tuple[Path, xr.Dataset]:
    print("\nSTEP 1: Converting with faxarray")
    start = time.time()
    fa = fx.open_fa(str(fa_file))
    try:
        ds = fa.to_xarray(variables=fields, stack_levels=False, progress=False)
    finally:
        fa.close()

    output_file = output_dir / "faxarray_output.nc"
    ds.to_netcdf(output_file)
    elapsed = time.time() - start
    print(f"  Fields: {fields}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {output_file}")
    return output_file, ds


def convert_with_epygram(fa_file: Path, fields: List[str], output_dir: Path):
    try:
        import epygram
    except Exception as exc:
        print("\nSTEP 2: EPYGRAM comparison skipped")
        print(f"  EPYGRAM is not importable: {exc}")
        return None, None

    print("\nSTEP 2: Converting with EPYGRAM Python API")
    start = time.time()
    epygram.init_env()
    resource = epygram.formats.resource(str(fa_file), "r")

    data = {}
    lons = lats = None
    try:
        for field in fields:
            fa_field = resource.readfield(field)
            if hasattr(fa_field, "spectral") and fa_field.spectral:
                fa_field.sp2gp()
            data[field] = fa_field.getdata()
            if lons is None:
                lons, lats = fa_field.geometry.get_lonlat_grid()
    finally:
        resource.close()

    ds = xr.Dataset(
        {name.replace(".", "_"): (["y", "x"], arr) for name, arr in data.items()},
        coords={"lat": (["y", "x"], lats), "lon": (["y", "x"], lons)},
    )
    output_file = output_dir / "epygram_output.nc"
    ds.to_netcdf(output_file)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {output_file}")
    return output_file, ds


def plot_faxarray_fields(ds: xr.Dataset, fields: List[str], output_dir: Path) -> Path:
    print("\nSTEP 3: Plotting faxarray fields")
    n_fields = len(fields)
    fig, axes = plt.subplots(n_fields, 1, figsize=(8, 4 * n_fields), squeeze=False)
    lon = ds["lon"].values
    lat = ds["lat"].values

    for index, field in enumerate(fields):
        safe_name = field.replace(".", "_")
        data = ds[safe_name].squeeze().values
        ax = axes[index, 0]
        mesh = ax.pcolormesh(lon, lat, data, cmap="viridis")
        ax.set_title(safe_name)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()
    plot_file = output_dir / "faxarray_fields.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Output: {plot_file}")
    return plot_file


def compare_outputs(ds_faxarray: xr.Dataset, ds_epygram: xr.Dataset, fields: List[str]) -> bool:
    print("\nSTEP 4: Comparing faxarray and EPYGRAM")
    all_close = True
    for field in fields:
        safe_name = field.replace(".", "_")
        faxarray_data = ds_faxarray[safe_name].squeeze().values
        epygram_data = ds_epygram[safe_name].squeeze().values
        diff = faxarray_data - epygram_data
        max_abs_diff = float(np.nanmax(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        print(f"  {safe_name}: max_abs_diff={max_abs_diff:.3e}, rmse={rmse:.3e}")
        if max_abs_diff > 1.0e-8:
            all_close = False
    return all_close


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa-file", type=Path, default=DEFAULT_FA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fields-file", type=Path, default=DEFAULT_FIELDS_FILE)
    args = parser.parse_args(argv)

    fields = load_fields(args.fields_file)
    if not args.fa_file.exists():
        print(f"FA file does not exist: {args.fa_file}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    supported, skipped = classify_fields(args.fa_file, fields)

    print("VALIDATION TEST")
    print(f"FA file: {args.fa_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Requested fields: {fields}")
    print(f"Supported H2D fields: {supported}")
    if skipped:
        print("Skipped fields:")
        for field, reason in skipped.items():
            print(f"  {field}: {reason}")

    if not supported:
        print("No supported H2D fields were selected", file=sys.stderr)
        return 1

    faxarray_file, ds_faxarray = convert_with_faxarray(args.fa_file, supported, args.output_dir)
    plot_file = plot_faxarray_fields(ds_faxarray, supported, args.output_dir)
    epygram_file, ds_epygram = convert_with_epygram(args.fa_file, supported, args.output_dir)

    matched = None
    if ds_epygram is not None:
        matched = compare_outputs(ds_faxarray, ds_epygram, supported)

    print("\nSUMMARY")
    print(f"  faxarray NetCDF: {faxarray_file}")
    print(f"  faxarray plot: {plot_file}")
    if epygram_file is not None:
        print(f"  EPYGRAM NetCDF: {epygram_file}")
        print(f"  comparison matched: {matched}")
    return 0 if matched is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
