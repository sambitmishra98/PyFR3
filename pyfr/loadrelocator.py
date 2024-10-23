from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
import time

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, AlltoallMixin, mpi
from pyfr.nputil import iter_struct

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfr.readers.native import _Mesh

from termcolor import colored


class LoadRelocator:
    def __init__(self, base_mesh: _Mesh):
        self.update_metamesh(base_mesh)
        self.nelems_tot = self.count_all_eidxs(base_mesh.eidxs)
    
    @staticmethod
    def count_all_eidxs(mesh_eidxs):
        comm, rank, root = get_comm_rank_root()
        total_elements = sum(len(e) for e in mesh_eidxs.values())
        return comm.allreduce(total_elements, op=mpi.SUM)

    def update_metamesh(self, base_mesh: _Mesh):
        self.mmesh = base_mesh.copy()
        self.mmesh.etypes = sorted(self.mmesh.etypes)

        self.mmesh.eidxs       = self.preproc_edict(self.mmesh.eidxs      , edim=0)
        self.mmesh.eles        = self.preproc_edict(self.mmesh.eles       , edim=0)
        self.mmesh.spts_curved = self.preproc_edict(self.mmesh.spts_curved, edim=0)
        self.mmesh.spts_nodes  = self.preproc_edict(self.mmesh.spts_nodes , edim=0)
        self.mmesh.spts        = self.preproc_edict(self.mmesh.spts       , edim=1)

        self.mpi_relocator = AlltoallRelocator(self.mmesh)

        if base_mesh is None:
            del self.new_mesh_copy

    def update_to_new_indices(self, move_by_etype):

        new_mesh_eidxs = self.get_preordered_eidxs(move_by_etype)

        self.new_mesh_copy = self.mmesh.copy()
        self.new_mesh_copy.eidxs = self.preproc_edict(new_mesh_eidxs, edim=0)

        # Update the result of modified list of elements within mesh.
        self.mpi_relocator.update_mesh(self.new_mesh_copy)

    @staticmethod
    def check_edict(edict: dict[str, np.ndarray], etypes):
        if not isinstance(edict, dict):
            raise ValueError("Input must be a dictionary.")
        
        missing_etypes = [etype for etype in etypes if etype not in edict]
        if missing_etypes:
            raise ValueError(f"Input dict must contain all etypes, we are "
                            f"missing --> {missing_etypes}")

        for etype, ary in edict.items():
            if not isinstance(ary, np.ndarray):
                raise ValueError(f"ary['{etype}'] must be a np.array.")
            if ary.ndim == 0:
                raise ValueError(f"ary['{etype}'] np.array cannot be scalar.")

    def preproc_edict(self, edict_in: dict[str, np.ndarray],*, edim: int):
        """
        Ensure dictionary contains all etypes. 
        If missing etype, add an empty array similar to the existing arrays.
        """

        comm, rank, root = get_comm_rank_root()

        # Make a copy to avoid modifying the original dictionary
        edict = deepcopy(edict_in)

        # For each etype in edict, collect dtype and shape
        for etype in self.mmesh.etypes:
            ary = edict.get(etype)

            if ary is not None:
                ary = ary.copy()
                create_ary = False
                # Ensure element index is the first dimension
                if edim != 0:
                    if ary.ndim < edim:
                        raise ValueError(f"Expected {edim+1}+ dimensions array")
                    ary = np.moveaxis(ary, edim, 0)

                ary_dtype = ary.dtype
                ary_shape = ary.shape

            else:
                create_ary = True
                ary_dtype = None
                ary_shape = None

            # If one of the ranks has nothing, create empty array in it
            if comm.allreduce(create_ary, op=mpi.MAX):
                # Get the dtype and shape of those arrays that are not empty
                ary_dtypes = comm.allgather(ary_dtype)
                ary_shapes = comm.allgather(ary_shape)
                
                # We can use any non-none dtype and shape
                for dtype, shape in zip(ary_dtypes, ary_shapes):
                    if dtype is not None and shape is not None:
                        break
                else:
                    raise ValueError(f"All empty arrays for etype: {etype}.")
                
                # Create an empty array with the same dtype and shape
                if create_ary:
                    edict[etype] = np.empty((0,) + shape[1:], dtype=dtype)
                else:
                    # Ensure the non-empty arrays have the same dtype
                    edict[etype] = ary.astype(dtype)

            else:
                edict[etype] = ary

        self.check_edict(edict, self.mmesh.etypes)

        # Ensure etypes keys are alphabetically ordered
        edict = {etype: edict[etype] for etype in sorted(edict.keys())}

        return edict

    def postproc_edict(self, edict: dict[str, np.ndarray], *, edim: int):

        edict_out = {}

        for etype in edict:
            if edict[etype].size > 0:
                if edim != 0 and edict[etype].ndim > edim:
                    edict_out[etype] = np.moveaxis(edict[etype], 0, edim)
                else:
                    edict_out[etype] = edict[etype]

        return edict_out

    @property
    def mesh(self):

        # If the mesh does not exist, return None
        if not hasattr(self, 'new_mesh_copy'):
            return None

        # Make a copy of the mesh to avoid modifying the original
        output_mesh = self.new_mesh_copy.copy()

        # postproc the mesh before returning
        output_mesh.eidxs = self.postproc_edict(output_mesh.eidxs, edim=0)
        output_mesh.eles = self.postproc_edict(output_mesh.eles, edim=0)
        output_mesh.spts_curved = self.postproc_edict(output_mesh.spts_curved, edim=0)
        output_mesh.spts_nodes = self.postproc_edict(output_mesh.spts_nodes, edim=0)
        output_mesh.spts = self.postproc_edict(output_mesh.spts, edim=1)

        return output_mesh

    def move(self, etypes: list[str], eids: list[int], startranks: list[int], 
                   endranks:list[int]) -> dict:
        comm, rank, root = get_comm_rank_root()
        
        # Initialise with empty lists
        move_by_etype = {nrank: {etype: [] for etype in self.mmesh.etypes} for nrank in range(comm.size) if nrank != rank}

        for etype, eid, startrank, endrank in zip(etypes, eids, startranks, endranks):
            if rank == startrank:
                move_by_etype[endrank][etype].append(eid)

        return move_by_etype


    def observe(self, perfs):
        comm, rank, root = get_comm_rank_root()
        
        # Function to gather all values from a dictionary
        def gather_all(data_dict):
            return {key: np.array(comm.allgather(value)) for key, value in data_dict.items()}

        # Gather all performance and time metrics
        gathered_perfs = gather_all(perfs)

        # Gather the number of elements from all processes
        self.nelems = np.array(comm.allgather(sum(len(e) for e in self.mmesh.eidxs.values())))

        self.cost = np.array(gathered_perfs['target']) / sum(gathered_perfs['target'])

        return sum(self.nelems)*self.cost[rank] / sum(self.cost)

    def diffuse(self, target_nelems):
        comm, rank, root = get_comm_rank_root()
        csize = comm.size

        nelems_diff = np.array(comm.allgather(self.nelems[rank]-target_nelems)) 
        print(nelems_diff)

        gcon_p = self._build_global_connectivity()

        ncon_p_dict = {nrank: {etype: len(e) for etype, e in inters.items()} for nrank, inters in gcon_p.items()}
        ncon_p_sum = sum(sum(inters.values()) for inters in ncon_p_dict.values())

        nelems_from_nrank_dict = {
            nrank: nelems_diff[rank]*sum(inters.values())/ncon_p_sum for nrank, inters in ncon_p_dict.items()}
        nelems_from_nrank_dict[rank] = 0                                                        # Own rank element movement zero

        LoadRelocator.cprint(nelems_from_nrank_dict, name='nelems_from_nrank_dict')
        
        nelems_from_nrank       = np.zeros((csize, csize))
        nelems_from_nrank[rank] = [nelems_from_nrank_dict[nrank] for nrank in range(csize)] # Fill the matrix with the number of elements to move from each rank to another
        nelems_from_nrank       = comm.allreduce(nelems_from_nrank, op=mpi.SUM)                 # sum across all ranks
        nelems_from_nrank       = (nelems_from_nrank - nelems_from_nrank.T) / 2                 # Get (M - Mᵀ)/2, so send[nrank] = recv[nnrank]

        nelems_from_nrank = np.round(nelems_from_nrank).astype(int) # Round off all elements to the nearest integer
        nelems_from_nrank = np.maximum(nelems_from_nrank, 0)        # Remove all negatives

        neles_from_nrank = nelems_from_nrank[rank,    :]

        move_elements = self._select_elements_to_move(gcon_p, ncon_p_dict, neles_from_nrank)

        self.update_to_new_indices(move_elements)

    def _build_global_connectivity(self):
        return {
            nrank: {
                etype: np.array([
                    self.mmesh.eidxs[etype][inter[1]] 
                       for inter in inters if inter[0] == etype
                    ]) for etype in self.mmesh.etypes
                } for nrank, inters in self.mmesh.con_p.items()
            } 

    def _select_elements_to_move(self, gcon_p, ncon_p_dict, neles_from_nrank):
        ncon_p_sum = sum(sum(inters.values()) for inters in ncon_p_dict.values())
        
        # Set a probability of movement for each element.
        move_elements = {}
        for nrank in gcon_p:
            #prob = neles_from_nrank[nrank] /ncon_p_sum
            #bool_by_etype = {etype: np.random.rand(ncon_p_dict[nrank][etype]) < min(1, prob) for etype in self.mmesh.etypes}
            #move_elements[nrank] = {
            #    etype: np.unique(np.sort(np.array([
            #        intelem 
            #        for intelem, move in zip(gcon_p[nrank][etype], bool_by_etype[etype]) if move]))) 
            #            for etype in self.mmesh.etypes
            #    }

            #LoadRelocator.cprint(move_elements[nrank], name=f'OPTION 1: move_elements[{nrank}]')

            move_elements[nrank] = {}
            for etype in self.mmesh.etypes:
                elements = gcon_p[nrank][etype]
                num_elements = len(elements)
                num_to_move = int(neles_from_nrank[nrank] * 
                                  (ncon_p_dict[nrank][etype] / ncon_p_sum))
                if num_elements > 0 and num_to_move > 0:
                    selected_indices = np.random.choice(
                        num_elements, min(num_to_move, num_elements), 
                        replace=False)
                    move_elements[nrank][etype] = elements[selected_indices]
                else:
                    move_elements[nrank][etype] = np.array([], dtype=int)

        # Prioritise decreasing load on the costliest ranks first
        sorted_ranks = sorted(gcon_p, key=lambda x: self.cost[x], reverse=False)

        # Step 8: Remove duplicates by iterating over sorted ranks
        for i, nrank in enumerate(sorted_ranks):
            for etype in self.mmesh.etypes:
                for nnrank in sorted_ranks[i+1:]:
                    move_elements[nnrank][etype] = np.setdiff1d(
                        move_elements[nnrank][etype], move_elements[nrank][etype]
                    )

        return move_elements

    def get_preordered_eidxs(self, move_to_nrank: dict[int, dict[str, np.ndarray]]):
        """
        Get new element indices on each rank after relocation.
        
        """

        comm, rank, root = get_comm_rank_root()
        
        # Sync expected element movement across all ranks
        moved_from_nrank = self._send_recv(move_to_nrank)

        new_mesh_eidxs = deepcopy(self.mmesh.eidxs)
        # preprocs new_mesh_eidxs
        new_mesh_eidxs = self.preproc_edict(new_mesh_eidxs, edim=0)

        for etype in self.mmesh.etypes:
            elements_to_remove = np.sort(np.concatenate([elements[etype] for elements in move_to_nrank.values()]))
            new_mesh_eidxs[etype] = np.setdiff1d(new_mesh_eidxs[etype], elements_to_remove)

        for etype in self.mmesh.etypes:
            elements_to_add = np.sort(np.concatenate([elements[etype] for elements in moved_from_nrank.values()]))
            new_mesh_eidxs[etype] = np.unique(np.concatenate((new_mesh_eidxs[etype], elements_to_add)).astype(np.int32))

        # Count the overall number of elements
        ntotal_final = self.count_all_eidxs(new_mesh_eidxs)

        # Ensure the total number of elements is preserved
        if rank == root and self.nelems_tot != ntotal_final:
            raise ValueError(f"Total number of elements changed during relocation. "
                            f"Expected {self.nelems_tot}, got {ntotal_final}.")

        return new_mesh_eidxs

    def _send_recv(self, send_elements: dict[int, dict[str, np.ndarray]]):
        comm, rank, root = get_comm_rank_root()

        # Collect ranks to send to
        ranks_to_send = list(send_elements.keys())

        # Share ranks_to_send lists among all ranks
        all_ranks_to_send = comm.allgather(ranks_to_send)

        # Determine ranks to receive from
        ranks_to_recv = []
        for i, ranks in enumerate(all_ranks_to_send):
            if rank in ranks and i != rank:
                ranks_to_recv.append(i)

        # Initiate non-blocking receives
        recv_reqs = []
        recv_elements = {}
        for nrank in ranks_to_recv:
            # Prepare to receive data
            req = comm.irecv(source=nrank, tag=0)
            recv_reqs.append((nrank, req))

        # Initiate non-blocking sends
        send_reqs = []
        for nrank, etype_elements in send_elements.items():
            if rank != nrank:
                # Send data
                req = comm.isend(etype_elements, dest=nrank, tag=0)
                send_reqs.append(req)

        # Wait for all receives to complete
        for nrank, req in recv_reqs:
            etype_elements = req.wait()
            recv_elements[nrank] = etype_elements

        # Wait for all sends to complete
        comm.Barrier()

        return recv_elements


    def relocate(self, edict: dict[str, np.ndarray],*, edim) -> dict[str, np.ndarray]:

        edict = self.preproc_edict(edict, edim=edim)
        self.check_edict(edict, self.mmesh.etypes)
        relocated_dict = self.mpi_relocator.relocate(edict)
        return self.postproc_edict(relocated_dict, edim=edim)

    @staticmethod
    def cprint(text, name='blank'):
        comm, rank, root = get_comm_rank_root()
        rank_colors = ['green', 'blue', 'red', 'yellow', 'cyan', 'magenta', 'white']
        color = rank_colors[rank % len(rank_colors)]
        rank_color = lambda msg: colored(msg, color=color)

        # Helper function for recursive formatting
        def format_recursive(obj, indent=0):

            ind = '    ' * indent  # 4 spaces per indentation level
            if isinstance(obj, Mapping):
                lines = []
                for k, v in obj.items():
                    formatted_v = format_recursive(v, indent + 1)
                    lines.append(f"{ind}{k}: {formatted_v}")
                return '{\n' + '\n'.join(lines) + f'\n{ind}}}'
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                lines = []
                for v in obj:
                    formatted_v = format_recursive(v, indent + 1)
                    lines.append(f"{ind}{formatted_v}")
                return '[\n' + '\n'.join(lines) + f'\n{ind}]'
            elif isinstance(obj, np.ndarray):
                with np.printoptions(
                    precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
                ):
                    array_str = np.array2string(obj, separator=', ')
                return array_str
            else:
                return repr(obj)

        # Ensure outputs are printed in order by rank without sleep
        for r in range(comm.size):
            if rank == r:
                print(rank_color(f"{name}"))

                if isinstance(text, dict):
                    formatted_text = format_recursive(text)
                    print(rank_color(formatted_text))

                elif isinstance(text, list) and all(isinstance(t, np.ndarray) for t in text):
                    with np.printoptions(
                        precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
                    ):
                        for t in text:
                            print(rank_color(t), end="\n\n")

                elif isinstance(text, np.ndarray):
                    with np.printoptions(
                        precision=6, suppress=True, threshold=np.inf, linewidth=np.inf, nanstr='nan'
                    ):
                        print(rank_color(text))

                else:
                    # Use recursive formatter for complex objects
                    formatted_text = format_recursive(text)
                    print(rank_color(formatted_text))

                if rank == comm.size - 1:
                    print("-" * 100)
            time.sleep(r/100)
            comm.Barrier()


class AlltoallRelocator(AlltoallMixin):

    def __init__(self, mmesh: _Mesh):
        self.comm, self.rank, self.root = get_comm_rank_root()

        # Neighbours are all ranks except the current rank
        self.neighbours = [i for i in range(self.comm.size) if i != self.comm.rank]

        self.etypes = mmesh.etypes
        self.mmesh_eidxs = self._gather_mesh_eidxs(mmesh.eidxs)

        # Initialize indexing dictionaries
        self._scount = {}
        self._sdisp = {}
        self._rcount = {}
        self._rdisp = {}
        self._sidxs = {}
        self._ridxs = {}

        self._inv_scount = {}
        self._inv_sdisp = {}
        self._inv_rcount = {}
        self._inv_rdisp = {}
        self._inv_sidxs = {}
        self._inv_ridxs = {}

    def update_mesh(self, new_mesh_copy: _Mesh):
        # Make new_mesh_copy_eidxs accessible in other methods
        self.new_mesh_copy_eidxs = self._gather_mesh_eidxs(new_mesh_copy.eidxs)

        for etype in self.etypes:
            self.relocation_indexing(etype, self.new_mesh_copy_eidxs, invert=False)
            self.relocation_indexing(etype, self.new_mesh_copy_eidxs, invert=True)

        self._recreate_mesh(new_mesh_copy, reordered=False)

        # Reorder eidxs to ensure the following order:
        # [MPI-curved, MPI-linear, internal-curved, internal-linear]
        target_new_eidxs = self._reorder_elements(new_mesh_copy)

        # Repeat mesh reconstruction with reordered eidxs
        self._recreate_mesh(new_mesh_copy, reordered=False, invert=True)
        new_mesh_copy.eidxs = target_new_eidxs
        self.new_mesh_copy_eidxs = self._gather_mesh_eidxs(new_mesh_copy.eidxs)
        for etype in self.etypes:
            self.relocation_indexing(etype, self.new_mesh_copy_eidxs, invert=False)
            self.relocation_indexing(etype, self.new_mesh_copy_eidxs, invert=True)
        self._recreate_mesh(new_mesh_copy, reordered=True)
        
    def _recreate_mesh(self, mesh, reordered, invert=False):

        # Reconstruct everything needed
        mesh.eles        = self.relocate(mesh.eles,        invert=invert)
        mesh.spts_curved = self.relocate(mesh.spts_curved, invert=invert)

        if not invert:
            if reordered:
                mesh.spts       = self.relocate(mesh.spts      )
                mesh.spts_nodes = self.relocate(mesh.spts_nodes)
            self._reconstruct_con(mesh, mesh.eles)
            self.set_mpi_elems(mesh)

    def set_mpi_elems(self, mesh):

        # Find all elements that lie on the MPI boundary
        ifmpi = {etype: [] for etype in self.etypes}

        for nrank, ncon in mesh.con_p.items():
            for etype, eidx, fidx in ncon:
                ifmpi[etype].append(eidx)

        # Create a sorted list of unique elements
        for etype in self.etypes:
            ifmpi[etype] = np.unique(ifmpi[etype])

        # Towards lexsort we need a boolean array for each element type
        mesh.spts_internal = {etype: np.ones(mesh.eles[etype].shape[0], dtype=bool) for etype in self.etypes}
        
        for etype, eidxs in ifmpi.items():
            mesh.spts_internal[etype][eidxs.astype(int)] = False

    def _reorder_elements(self, mesh):
        # Reorder the elements, consider the following ordering
        # pidx = np.lexsort((ecurved, ~internal))

        new_target_eidxs = {etype: [] for etype in self.etypes}
                
        for etype in self.etypes:
            ordered_idx = np.lexsort(( mesh.spts_curved[etype], 
                                       mesh.spts_internal[etype])) 
            if len(ordered_idx) > 0:
                new_target_eidxs[etype] = mesh.eidxs[etype][ordered_idx]
            else:
                new_target_eidxs[etype] = mesh.eidxs[etype]

        return new_target_eidxs

    def _gather_mesh_eidxs(self, mesh_eidxs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:

        gathered_mesh_eidxs = {}
        for etype in self.etypes:

            # If exists, convert to numpy array. Else, use empty array
            elems = mesh_eidxs[etype] if etype in mesh_eidxs else np.array([], dtype=np.int32)

            if not isinstance(elems, np.ndarray):
                elems = np.array(elems)
            gathered_mesh_eidxs[etype] = self.comm.allgather(elems)
        return gathered_mesh_eidxs

    def relocation_indexing(self, etype, new_mesh_copy_eidxs, invert=False):
        if not invert:
            # Relocation
            base_mesh_eidxs = self.mmesh_eidxs
            end_mesh_eidxs = new_mesh_copy_eidxs
            _sidxs = self._sidxs
            _scount = self._scount
            _sdisp = self._sdisp
            _ridxs = self._ridxs
            _rcount = self._rcount
            _rdisp = self._rdisp
        else:
            # Unrelocation
            base_mesh_eidxs = new_mesh_copy_eidxs
            end_mesh_eidxs = self.mmesh_eidxs
            _sidxs = self._inv_sidxs
            _scount = self._inv_scount
            _sdisp = self._inv_sdisp
            _ridxs = self._inv_ridxs
            _rcount = self._inv_rcount
            _rdisp = self._inv_rdisp

        # Build the send indices
        _sidxs[etype] = [
            [int(e_number) for e_number in base_mesh_eidxs[etype][self.rank]
             if e_number in end_mesh_eidxs[etype][rank]]
            for rank in range(self.comm.size)
        ]
        _scount[etype] = np.array([len(_sidxs[etype][rank]) for rank in range(self.comm.size)])
        _sdisp[etype] = self._count_to_disp(_scount[etype])
        _sidxs[etype] = np.concatenate(_sidxs[etype]).astype(int)

        # Communicate to get receive counts and displacements
        _, (_rcount[etype], _rdisp[etype]) = self._alltoallcv(
            self.comm, _sidxs[etype], _scount[etype], _sdisp[etype]
        )

        # Allocate receive indices array
        _ridxs[etype] = np.empty(
            (_rcount[etype].sum(), *_sidxs[etype].shape[1:]),
            dtype=_sidxs[etype].dtype
        )

        # Perform all-to-all communication to exchange indices
        self._alltoallv(
            self.comm,
            (_sidxs[etype], (_scount[etype], _sdisp[etype])),
            (_ridxs[etype], (_rcount[etype], _rdisp[etype]))
        )
    def relocate(self, sary_dict: dict[str, np.ndarray],*, invert: bool=False) -> dict[str, np.ndarray]:
        return {etype: self._relocate_ary(etype, sary, invert=invert) for etype, sary in sary_dict.items()}

    def _relocate_ary(self, etype: str, ary: np.ndarray,*, invert=False) -> np.ndarray:
        """
        Relocate data across ranks along elements dimension for each etype.

        Args:
            etype (str): The element type.
            ary (np.ndarray): The array to relocate or unrelocate.
            invert (bool): Relocate if false, else unrelocate.

        Returns:
            np.ndarray: Relocated/unrelocated array.
        """

        if not invert:
            # Relocation
            base_idxs = self.mmesh_eidxs[etype][self.rank]
            end_idxs = self.new_mesh_copy_eidxs[etype][self.rank]
            send_idxs = self._sidxs[etype]
            recv_idxs = self._ridxs[etype]
            scount = self._scount[etype]
            sdisp = self._sdisp[etype]
            rcount = self._rcount[etype]
            rdisp = self._rdisp[etype]
        else:
            # Unrelocation
            base_idxs = self.new_mesh_copy_eidxs[etype][self.rank]
            end_idxs = self.mmesh_eidxs[etype][self.rank]
            send_idxs = self._inv_sidxs[etype]
            recv_idxs = self._inv_ridxs[etype]
            scount = self._inv_scount[etype]
            sdisp = self._inv_sdisp[etype]
            rcount = self._inv_rcount[etype]
            rdisp = self._inv_rdisp[etype]

        base_idxs_position = {idx: pos for pos, idx in enumerate(base_idxs)}

        try:
            base_to_send = [base_idxs_position[idx] for idx in send_idxs]
        except KeyError as e:
            raise KeyError(f"Index {e} not found in base_idxs_position for etype '{etype}' on rank {self.rank}")


        svals = ary[base_to_send]

        rvals = np.empty((rcount.sum(), *svals.shape[1:]), dtype=svals.dtype)
        
        self._alltoallv(self.comm, (svals, (scount, sdisp)),
                                   (rvals, (rcount, rdisp)),
                       )

        recv_idxs_position = {idx: pos for pos, idx in enumerate(recv_idxs)}
        recv_to_end = [recv_idxs_position[idx] for idx in end_idxs]
        rary = rvals[recv_to_end]

        return rary

    def _reconstruct_con(self, mesh, eles):
        mesh.bcon = {bc.split('/')[1]: [] for bc in mesh.codec if bc.startswith('bc/')}
        
        codec = mesh.codec
        eidxs = {k: v.tolist() for k, v in mesh.eidxs.items()}
        etypes = mesh.etypes

        # Create a map from global to local element numbers
        glmap = [{}]*len(etypes)
        for i, etype in enumerate(etypes):
            if etype in eidxs:
                glmap[i] = {k: j for j, k in enumerate(eidxs[etype])}

        # Create cidx indexed maps
        cdone, cefidx = [None]*len(codec), [None]*len(codec)
        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/(\d+)$', c)):
                etype, fidx = m[1], m[2]
                cdone[cidx] = set()
                cefidx[cidx] = (etype, etypes.index(etype), int(fidx))

        conl, conr = [], []
        bcon = {i: [] for i, c in enumerate(codec) if c.startswith('bc/')}
        resid = {}

        for etype, einfo in eles.items():

            try:
                i = etypes.index(etype)
                for fidx, eface in enumerate(einfo['faces'].T):
                    efcidx = codec.index(f'eles/{etype}/{fidx}')

                    for j, (cidx, off) in enumerate(iter_struct(eface)):
                        # Boundary
                        if off == -1:
                            bcon[cidx].append((etype, j, fidx))
                        # Unpaired face
                        elif j not in cdone[efcidx]:
                            # Lookup the element type and face number
                            ketype, ketidx, kfidx = cefidx[cidx]

                            # If our rank has the element then pair it
                            if (k := glmap[ketidx].get(off)) is not None:
                                conl.append((etype, j, fidx))
                                conr.append((ketype, k, kfidx))
                                cdone[cidx].add(k)
                            # Otherwise add it to the residual dict
                            else:
                                resid[efcidx, eidxs[etype][j]] = (cidx, off)
            except:
                pass

        # Add the internal connectivity to the mesh
        mesh.con = (conl, conr)

        for k, v in bcon.items():
            if v:
                mesh.bcon[codec[k][3:]] = v

        # Handle inter-partition connectivity
        if resid:
            self._reconstruct_mpi_con(mesh, glmap, cefidx, resid)

    def _reconstruct_mpi_con(self, mesh, glmap, cefidx, resid):
        comm, rank, root = get_comm_rank_root()

        # Create a neighbourhood collective communicator
        ncomm = comm.Create_dist_graph_adjacent(self.neighbours,
                                                self.neighbours)

        # Create a list of our unpaired faces
        unpaired = list(resid.values())

        # Distribute this to each of our neighbours
        nunpaired = ncomm.neighbor_allgather(unpaired)

        # See which of our neighbours unpaired faces we have
        matches = [[resid[j] for j in nunp if j in resid]
                   for nunp in nunpaired]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        for nrank, nmatch in zip(self.neighbours, nmatches):
            if rank < nrank:
                ncon = sorted([(resid[m], m) for m in nmatch])
                ncon = [r for l, r in ncon]
            else:
                ncon = sorted([(m, resid[m]) for m in nmatch])
                ncon = [l for l, r in ncon]

            nncon = []
            for cidx, off in ncon:
                etype, etidx, fidx = cefidx[cidx]
                nncon.append((etype, glmap[etidx][off], fidx))

            # Add the connectivity to the mesh
            mesh.con_p[nrank] = nncon
