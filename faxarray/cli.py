"""
Command-line interface for faxarray.

Provides quick access to common operations from the terminal.
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args):
    """Show file information."""
    from .core import open_fa
    
    fa = open_fa(args.file)
    print(fa.info())
    print()
    
    if args.list_vars:
        print("Variables:")
        for i, var in enumerate(fa.variables):
            print(f"  {i+1:4d}. {var}")
    else:
        print(f"First 20 variables:")
        for i, var in enumerate(fa.variables[:20]):
            print(f"  {i+1:4d}. {var}")
        if len(fa.variables) > 20:
            print(f"  ... and {len(fa.variables) - 20} more")
            print(f"  (use --list-vars to see all)")
    
    fa.close()


def cmd_convert(args):
    """Convert FA to NetCDF."""
    from .core import open_fa
    
    fa = open_fa(args.input)
    
    compress = args.compress if args.compress != 'none' else None
    
    fa.to_netcdf(
        args.output,
        compress=compress,
        compress_level=args.level,
        progress=not args.quiet
    )
    
    fa.close()


def cmd_plot(args):
    """Quick plot of a field."""
    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    from .core import open_fa
    
    fa = open_fa(args.file)
    
    if args.field:
        var = fa[args.field]
    else:
        # Use first surface field or first field
        for name in fa.variables:
            if name.startswith('SURF'):
                var = fa[name]
                break
        else:
            var = fa[fa.variables[0]]
    
    print(f"Plotting: {var.name}")
    print(f"  Range: [{var.min():.4g}, {var.max():.4g}]")
    
    var.plot(
        cmap=args.cmap,
        use_cartopy=not args.no_cartopy
    )
    
    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to: {args.output}")
    else:
        plt.show()
    
    fa.close()


def cmd_benchmark(args):
    """Benchmark conversion speed."""
    import time
    from .core import open_fa
    
    fa = open_fa(args.file)
    print(f"File: {args.file}")
    print(f"Variables: {fa.nvars}")
    print(f"Grid: {fa.shape}")
    print()
    
    # Benchmark reading
    print("Benchmarking read speed...")
    start = time.time()
    fa.load(progress=True)
    read_time = time.time() - start
    print(f"  Read time: {read_time:.2f}s")
    print()
    
    # Benchmark NetCDF write (uncompressed)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=True) as tmp:
        print("Benchmarking NetCDF write (uncompressed)...")
        start = time.time()
        fa.to_netcdf(tmp.name, compress=None, progress=False)
        write_time = time.time() - start
        
        import os
        size_mb = os.path.getsize(tmp.name) / 1e6
        print(f"  Write time: {write_time:.2f}s")
        print(f"  Output size: {size_mb:.1f} MB")
    
    print()
    print(f"Total: {read_time + write_time:.2f}s")
    
    fa.close()


def fa2nc_main():
    """Standalone FA to NetCDF converter (fa2nc command)."""
    parser = argparse.ArgumentParser(
        prog='fa2nc',
        description='Convert FA files to NetCDF format'
    )
    parser.add_argument('input', help='Input FA file')
    parser.add_argument('output', help='Output NetCDF file')
    parser.add_argument('--compress', '-c', choices=['none', 'zlib'],
                        default='none', help='Compression (default: none)')
    parser.add_argument('--level', '-L', type=int, default=4,
                        help='Compression level 1-9 (default: 4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    try:
        from .core import open_fa
        fa = open_fa(args.input)
        compress = args.compress if args.compress != 'none' else None
        fa.to_netcdf(
            args.output,
            compress=compress,
            compress_level=args.level,
            progress=not args.quiet
        )
        fa.close()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 0


def cmd_convert_multi(args):
    """Convert multiple FA files to NetCDF with de-accumulation."""
    from .xarray_backend import open_mfdataset
    
    # Build list of variables to de-accumulate
    deaccum_vars = []
    
    # From -d flag (space or comma separated)
    if args.deaccumulate:
        for item in args.deaccumulate:
            # Handle comma-separated values
            deaccum_vars.extend(item.split(','))
    
    # From --dlist file
    if args.dlist:
        try:
            with open(args.dlist, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        deaccum_vars.append(line)
        except FileNotFoundError:
            print(f"Error: File not found: {args.dlist}", file=sys.stderr)
            return 1
    
    # Remove duplicates while preserving order
    seen = set()
    deaccum_vars = [v for v in deaccum_vars if not (v in seen or seen.add(v))]
    
    if not args.quiet:
        print(f"Converting {args.input} -> {args.output}")
        if deaccum_vars:
            print(f"  De-accumulating: {deaccum_vars}")
        print(f"  Chunk size: {args.chunk_hours} hour(s)")
    
    try:
        # Parse variables list
        var_list = None
        if args.variables:
            var_list = []
            for item in args.variables:
                var_list.extend(item.split(','))
        
        open_mfdataset(
            args.input,
            variables=var_list,
            deaccumulate=deaccum_vars if deaccum_vars else None,
            chunk_hours=args.chunk_hours,
            output_file=args.output,
            progress=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='faxarray',
        description='Fast, user-friendly interface for Météo-France FA files'
    )
    parser.add_argument('--version', action='version', version='faxarray 0.2.2')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show file information')
    info_parser.add_argument('file', help='FA file path')
    info_parser.add_argument('--list-vars', '-l', action='store_true',
                             help='List all variables')
    
    # convert command
    convert_parser = subparsers.add_parser('convert', help='Convert FA to NetCDF')
    convert_parser.add_argument('input', help='Input FA file')
    convert_parser.add_argument('output', help='Output NetCDF file')
    convert_parser.add_argument('--compress', '-c', choices=['none', 'zlib'],
                                default='none', help='Compression (default: none)')
    convert_parser.add_argument('--level', '-L', type=int, default=4,
                                help='Compression level 1-9 (default: 4)')
    convert_parser.add_argument('--quiet', '-q', action='store_true',
                                help='Quiet mode')
    
    # plot command
    plot_parser = subparsers.add_parser('plot', help='Quick plot')
    plot_parser.add_argument('file', help='FA file path')
    plot_parser.add_argument('--field', '-f', help='Field to plot')
    plot_parser.add_argument('--output', '-o', help='Save to file')
    plot_parser.add_argument('--cmap', default='viridis', help='Colormap')
    plot_parser.add_argument('--dpi', type=int, default=150, help='DPI for saved image')
    plot_parser.add_argument('--no-cartopy', action='store_true',
                             help='Disable cartopy projection')
    
    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark conversion')
    bench_parser.add_argument('file', help='FA file path')
    
    # convert-multi command
    multi_parser = subparsers.add_parser('convert-multi', 
        help='Convert multiple FA files to NetCDF with de-accumulation')
    multi_parser.add_argument('input', help='Input pattern (e.g., pf*+*)')
    multi_parser.add_argument('output', help='Output NetCDF file')
    multi_parser.add_argument('-d', '--deaccumulate', nargs='*', default=[],
                              help='Variables to de-accumulate (space or comma separated)')
    multi_parser.add_argument('--dlist', metavar='FILE',
                              help='File with list of variables to de-accumulate (one per line)')
    multi_parser.add_argument('--chunk-hours', type=int, default=1,
                              help='Hours to hold in memory at once (default: 1)')
    multi_parser.add_argument('-v', '--variables', nargs='*', default=[],
                              help='Variables to include (default: all). Use for low memory.')
    multi_parser.add_argument('--quiet', '-q', action='store_true',
                              help='Quiet mode')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        'info': cmd_info,
        'convert': cmd_convert,
        'plot': cmd_plot,
        'benchmark': cmd_benchmark,
        'convert-multi': cmd_convert_multi,
    }
    
    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
