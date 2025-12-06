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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='faxarray',
        description='Fast, user-friendly interface for Météo-France FA files'
    )
    parser.add_argument('--version', action='version', version='faxarray 0.1.0')
    
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        'info': cmd_info,
        'convert': cmd_convert,
        'plot': cmd_plot,
        'benchmark': cmd_benchmark,
    }
    
    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
