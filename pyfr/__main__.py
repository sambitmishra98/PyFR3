#!/usr/bin/env python
from argparse import ArgumentParser, FileType
from pathlib import Path
import re

import h5py
import mpi4py.rc
import numpy as np
mpi4py.rc.initialize = False

#from rich.traceback import install
#install(show_locals=True, width=250, )

from pyfr._version import __version__
from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.partitioners import (BasePartitioner, get_partitioner,
                               reconstruct_by_diffusion, 
                               reconstruct_partitioning, write_partitioning)
from pyfr.plugins import BaseCLIPlugin
from pyfr.progress import ProgressBar, ProgressSequenceAction
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.readers.stl import read_stl
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_extn, get_writer_by_name


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(help='sub-command help', metavar='command')

    # Common options
    ap.add_argument('-v', '--verbose', action='count',
                    help='increase verbosity')
    ap.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
    ap.add_argument('-p', '--progress', action=ProgressSequenceAction,
                    help='show progress')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.add_argument('-l', dest='lintol', type=float, default=1e-5,
                           help='linearisation tolerance')
    ap_import.set_defaults(process=process_import)

    # Partition subcommand
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition = ap_partition.add_subparsers()

    # List partitionings
    ap_partition_list = ap_partition.add_parser('list',
                                                help='partition list --help')
    ap_partition_list.add_argument('mesh', help='input mesh file')
    ap_partition_list.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_list.set_defaults(process=process_partition_list)

    # Get info about a partitioning
    ap_partition_info = ap_partition.add_parser('info',
                                                help='partition info --help')
    ap_partition_info.add_argument('mesh', help='input mesh file')
    ap_partition_info.add_argument('name', help='partitioning name')
    ap_partition_info.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_info.set_defaults(process=process_partition_info)

    # Add partitioning
    ap_partition_add = ap_partition.add_parser('add',
                                               help='partition add --help')
    ap_partition_add.add_argument('mesh', help='input mesh file')
    ap_partition_add.add_argument('np', help='number of partitions or a colon '
                                  'delimited list of weights')
    ap_partition_add.add_argument('name', nargs='?', help='partitioning name')
    ap_partition_add.add_argument('-f', '--force', action='count',
                                  help='overwrite existing partitioning')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition_add.add_argument(
        '-p', dest='partitioner', choices=partitioners,
        help='partitioner to use'
    )
    ap_partition_add.add_argument(
        '-e', dest='elewts', action='append', default=[],
        metavar='shape:weight', help='element weighting factor or "balanced"'
    )
    ap_partition_add.add_argument(
        '--popt', dest='popts', action='append', default=[],
        metavar='key:value', help='partitioner-specific option'
    )
    ap_partition_add.set_defaults(process=process_partition_add)

    # Reconstruct partitioning
    ap_partition_reconstruct = ap_partition.add_parser(
        'reconstruct', help='partition reconstruct --help'
    )
    ap_partition_reconstruct.add_argument('mesh', help='input mesh file')
    ap_partition_reconstruct.add_argument('soln', help='input solution file')
    ap_partition_reconstruct.add_argument('name', help='partitioning name')
    ap_partition_reconstruct.add_argument(
        '-f', '--force', action='count', help='overwrite existing partitioning'
    )
    ap_partition_reconstruct.set_defaults(
        process=process_partition_reconstruct
    )

    # Reconstruct partitioning by solution file
    ap_partition_diffuse = ap_partition.add_parser(
        'diffuse', help='partition diffuse --help'
    )

    ap_partition_diffuse.add_argument('mesh', help='input mesh file')
    ap_partition_diffuse.add_argument('name', help='partitioning name')
    ap_partition_diffuse.add_argument(
        '-f', '--force', action='count', help='overwrite existing partitioning'
    )

    ap_partition_diffuse.set_defaults(
        process=process_partition_diffuse
    )

    # Remove partitioning
    ap_partition_remove = ap_partition.add_parser(
        'remove', help='partition remove --help'
    )
    ap_partition_remove.add_argument('mesh', help='input mesh file')
    ap_partition_remove.add_argument('name', help='partitioning')
    ap_partition_remove.set_defaults(process=process_partition_remove)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='input mesh file')
    ap_export.add_argument('solnf', help='input solution file')
    ap_export.add_argument('outf', help='output file')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='ftype', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-f', '--field', dest='fields', action='append',
                           metavar='FIELD', help='what fields should be '
                           'output; may be repeated, by default all fields '
                           'are output')
    output_options = ap_export.add_mutually_exclusive_group(required=False)
    output_options.add_argument('-d', '--divisor', type=int,
                                help='sets the level to which high order '
                                'elements are divided; output is linear '
                                'between nodes, so increased resolution '
                                'may be required')
    output_options.add_argument('-k', '--order', type=int, dest='order',
                                help='sets the order of high order elements')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='output number precision; '
                           'defaults to single')
    ap_export.add_argument('-b', '--boundary', dest='boundaries',
                           action='append', metavar='BOUNDARY',
                           help='boundary to output; may be repeated')
    ap_export.set_defaults(process=process_export)

    # Region subcommand
    ap_region = sp.add_parser('region', help='region --help')
    ap_region = ap_region.add_subparsers()

    # Add region
    ap_region_add = ap_region.add_parser('add', help='region add --help')
    ap_region_add.add_argument('mesh', help='input mesh file')
    ap_region_add.add_argument('stl', type=FileType('rb'), help='STL file')
    ap_region_add.add_argument('name', help='region name')
    ap_region_add.set_defaults(process=process_region_add)

    # List regions
    ap_region_list = ap_region.add_parser('list', help='region list --help')
    ap_region_list.add_argument('mesh', help='input mesh file')
    ap_region_list.add_argument('-s', '--sep', default='\t', help='separator')
    ap_region_list.set_defaults(process=process_region_list)

    # Remove region
    ap_region_remove = ap_region.add_parser('remove',
                                            help='region remove --help')
    ap_region_remove.add_argument('mesh', help='input mesh file')
    ap_region_remove.add_argument('name', help='region name')
    ap_region_remove.set_defaults(process=process_region_remove)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('-b', '--backend', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('-p', '--pname', help='partitioning to use')

    # Plugin commands
    for scls in subclasses(BaseCLIPlugin, just_leaf=True):
        scls.add_cli(sp.add_parser(scls.name, help=f'{scls.name} --help'))

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh, args.progress)
    else:
        extn = Path(args.inmesh.name).suffix
        reader = get_reader_by_extn(extn, args.inmesh, args.progress)

    # Write out the mesh
    reader.write(args.outmesh, args.lintol)


def process_partition_list(args):
    with h5py.File(args.mesh, 'r') as mesh:
        print('name', 'parts', sep=args.sep)

        for name, part in sorted(mesh['partitionings'].items()):
            nparts = len(part['eles'].attrs['regions'])
            print(name, nparts, sep=args.sep)


def process_partition_info(args):
    with h5py.File(args.mesh, 'r') as mesh:
        # Read the partition region info from the mesh
        regions = mesh[f'partitionings/{args.name}/eles'].attrs['regions']

        # Print out the header
        print('part', *mesh['eles'], sep=args.sep)

        # Compute and output the number of elements in each partition
        for i, neles in enumerate(regions[:, 1:] - regions[:, :-1]):
            print(i, *neles, sep=args.sep)


def process_partition_add(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        # Determine the element types
        etypes = list(mesh['eles'])

        # Partition weights
        if ':' in args.np:
            pwts = [int(w) for w in args.np.split(':')]
        else:
            pwts = [1]*int(args.np)

        # Element weights
        if args.elewts == ['balanced']:
            ewts = None
        elif len(etypes) == 1:
            ewts = {etypes[0]: 1}
        else:
            ewts = (ew.split(':') for ew in args.elewts)
            ewts = {e: int(w) for e, w in ewts}

        # Ensure all weights have been provided
        if ewts is not None and len(ewts) != len(etypes):
            missing = ', '.join(set(etypes) - set(ewts))
            raise ValueError(f'Missing element weights for: {missing}')

        # Get the partitioning name
        pname = args.name or str(len(pwts))
        if not re.match(r'\w+$', pname):
            raise ValueError('Invalid partitioning name')

        # Check it does not already exist unless --force is given
        if pname in mesh['partitionings'] and not args.force:
            raise ValueError('Partitioning already exists; use -f to replace')

        # Partitioner-specific options
        opts = dict(s.split(':', 1) for s in args.popts)

        # Create the partitioner
        if args.partitioner:
            part = get_partitioner(args.partitioner, pwts, ewts, opts=opts)
        else:
            parts = sorted(cls.name for cls in subclasses(BasePartitioner))
            for name in parts:
                try:
                    part = get_partitioner(name, pwts, ewts)
                    break
                except OSError:
                    pass
            else:
                raise RuntimeError('No partitioners available')

        # Partition the mesh
        pinfo = part.partition(mesh, args.progress)

        # Write out the new partitioning
        with args.progress.start('Write partitioning'):
            write_partitioning(mesh, pname, pinfo)


def process_partition_reconstruct(args):
    with (h5py.File(args.mesh, 'r+') as mesh,
          h5py.File(args.soln, 'r') as soln):
        # Validate the partitioning name
        if not re.match(r'\w+$', args.name):
            raise ValueError('Invalid partitioning name')

        # Check it does not already exist unless --force is given
        if args.name in mesh['partitionings'] and not args.force:
            raise ValueError('Partitioning already exists; use -f to replace')

        # Reconstruct the partitioning used in the solution
        pinfo = reconstruct_partitioning(mesh, soln, args.progress)

        # Write out the new partitioning
        with args.progress.start('Write partitioning'):
            write_partitioning(mesh, args.name, pinfo)


def process_partition_diffuse(args):
    # Validate the partitioning name
    if not re.match(r'\w+$', args.name):
        raise ValueError('Invalid partitioning name')

    init_mpi()

    comm, rank, root = get_comm_rank_root()

    # If comm.size > 1 throw error
    if comm.size < 2:
        raise ValueError('Diffusion is meaningless for nranks > 1.')

    reader = NativeReader(args.mesh)
    read_only_mesh = reader.mesh
    reader.close()

    # Reconstruct the partitioning used in the solution
    vparts = reconstruct_by_diffusion(read_only_mesh, args.name, args.progress)

    if rank == root:
        with args.progress.start('Repartition'):
            if rank == root:
                with (h5py.File(args.mesh, 'r+') as mesh):
                    # Check it does not already exist unless --force is given
                    if args.name in mesh['partitionings'] and not args.force:
                        raise ValueError('Partitioning already exists; use -f to replace')

                    con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

                    pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps,
                                                                con, vparts)

                    # Write out the new partitioning
                    with args.progress.start('Write partitioning'):
                            write_partitioning(mesh, args.name, pinfo)

def process_partition_remove(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        mparts = mesh['partitionings']

        if args.name not in mparts:
            raise ValueError(f'Partitioning {args.name} does not exist')

        del mparts[args.name]


def process_region_add(args):
    # Read the STL file
    stl = read_stl(args.stl)

    # See if the surface is closed
    edges = np.vstack([stl[:, 1:3], stl[:, 2:4], stl[:, [3, 1]]])
    edges.view('f4,f4,f4').sort(axis=1)
    closed = (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()

    # Validate the name
    if not re.match(r'\w+$', args.name):
        raise ValueError('Invalid region name')

    with h5py.File(args.mesh, 'r+') as mesh:
        g = mesh.require_group('regions/stl')

        if args.name in g:
            del g[args.name]

        g[args.name] = stl
        g[args.name].attrs['closed'] = closed


def process_region_list(args):
    with h5py.File(args.mesh, 'r') as mesh:
        print('name', 'tris', 'closed', sep=args.sep)

        for name, tris in sorted(mesh.get('regions/stl', {}).items()):
            print(name, len(tris), str(tris.attrs['closed']).lower(),
                  sep=args.sep)


def process_region_remove(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        rparts = mesh.get('regions/stl')

        if rparts is None or args.name not in rparts:
            raise ValueError(f'Region {args.name} does not exist')

        del rparts[args.name]


def process_export(args):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()

    kwargs = {
        'prec': np.dtype(args.precision).type,
        'order': args.order,
        'divisor': args.divisor,
        'fields': args.fields,
        'boundaries': args.boundaries
    }

    # Get writer instance by specified type or outf extension
    if args.ftype:
        writer = get_writer_by_name(args.ftype, args.meshf, args.solnf,
                                    **kwargs)
    else:
        extn = Path(args.outf).suffix
        writer = get_writer_by_extn(extn, args.meshf, args.solnf, **kwargs)

    # Write the output file
    writer.write(args.outf)


def _process_common(args, soln, cfg):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()

    # Read the mesh
    reader = NativeReader(args.mesh, pname=args.pname)
    mesh = reader.mesh

    # Load a provided solution, if any
    if soln is not None:
        soln = reader.load_soln(soln)

    # If we do not have a config file then take it from the solution
    if cfg is None:
        cfg = soln['config']

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Construct the solver
    solver = get_solver(backend, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and rank == root:
        pbar = ProgressBar()
        pbar.start(solver.tend, start=solver.tstart, curr=solver.tcurr)

        # Register a callback to update the bar after each step
        solver.plugins.append(lambda intg: pbar(intg.tcurr))

    # Execute!
    solver.run()


def process_run(args):
    _process_common(args, None, Inifile.load(args.cfg))


def process_restart(args):
    cfg = Inifile.load(args.cfg) if args.cfg else None
    _process_common(args, args.soln, cfg)


if __name__ == '__main__':
    main()
