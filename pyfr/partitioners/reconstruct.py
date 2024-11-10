from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.partitioners.base import BasePartitioner
from pyfr.progress import NullProgressSequence
from pyfr.loadrelocator import LoadRelocator
from pyfr.readers.native import _Mesh

from pyfr.mpiutil import get_comm_rank_root

def reconstruct_partitioning(mesh, soln, progress=NullProgressSequence):
    if mesh['mesh-uuid'][()] != soln['mesh-uuid'][()]:
        raise ValueError('Invalid solution for mesh')

    prefix = Inifile(soln['stats'][()].decode()).get('data', 'prefix')
    sparts = defaultdict(list)

    # Read the partition data from the solution
    for k in soln[prefix]:
        if (m := re.match(r'(p(?:\d+)-(\w+))-parts$', k)):
            parts = soln[f'{prefix}/{k}'][:]

            idxs = soln.get(f'{prefix}/{m[1]}-idxs')
            if idxs is None:
                idxs = np.arange(len(parts))

            sparts[m[2]].append((idxs, parts))

    # Group the data together by element type
    for etype, sp in sparts.items():
        idxs, parts = map(np.concatenate, zip(*sp))

        sparts[etype] = parts[np.argsort(idxs)]

    vparts = np.concatenate([p for _, p in sorted(sparts.items())])

    # Construct the global connectivity array
    with progress.start('Construct global connectivity array'):
        con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

    # Ensure that the solution has not been subset
    if len(vparts) != len(ecurved):
        raise ValueError('Can not reconstruct partitioning from subsetted '
                         'solution')

    # Construct the partitioning data
    with progress.start('Construct partitioning'):
        pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps,
                                                       con, vparts)

    return pinfo

def reconstruct_by_diffusion(mesh: _Mesh, name: str, progress=NullProgressSequence):

    comm, rank, root = get_comm_rank_root()

    print(f'Creating mesh named {name} into mesh object: {mesh.fname}')

    with progress.start('Initialise relocator'):

        load_relocator = LoadRelocator(mesh, tol=0.1, low_elem=1024)
        mesh_name = 'diffusion'
        load_relocator.move_priority = list(range(comm.size))
        print('move_priority:', load_relocator.move_priority)
        # Create a new mesh with the same connectivity as the original mesh
        load_relocator.mm.copy_mmesh('base', mesh_name)
        load_relocator.curr_nelems = comm.allgather(load_relocator.mm.get_mmesh(mesh_name).nelems)
        t_nelems = load_relocator.mm.gnelems/comm.size

        # If equipartition mesh
        t_nelems_byrank = [t_nelems]*comm.size

    with progress.start('Start diffusion'):
        mesh, ii = load_relocator.diffuse_computation('compute', t_nelems_byrank)

    # Group partition number and element idx from mesh of each rank
    sparts = {etype: [] for etype in mesh.etypes}
    eidxs: dict[str,list[int]] = load_relocator.mm.get_mmesh(mesh_name).eidxs
    for etype, idxs in eidxs.items():
        sparts[etype] = (idxs, rank*np.ones(len(idxs), dtype=np.int32))


    # Gather sparts data from all ranks, by element type
    sparts = {k: comm.gather(v, root=root) for k, v in sparts.items()}
    if rank == root:
        # Within each element type, sort the data by the zeroeth element of the tuple, which is the element idx. 
        for etype, sp in sparts.items():
            idxs, parts = map(np.concatenate, zip(*sp))

            sparts[etype] = parts[np.argsort(idxs)] 

        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts
