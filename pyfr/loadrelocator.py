from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import re
import time

# Import deque
from collections import deque

import numpy as np
from tabulate import tabulate

from pyfr.mpiutil import get_comm_rank_root, AlltoallMixin, mpi
from pyfr.nputil import iter_struct
from pyfr.readers.native import _Mesh

# Remove later
import logging
from termcolor import colored
from typing import List, Dict

LOG_LEVEL=40

# Fix seed
np.random.seed(47)

def log_method_times(method):
    def wrapper(self, *args, **kwargs):
        tstart = time.perf_counter()
        result = method(self, *args, **kwargs)
        tdiff = time.perf_counter() - tstart

        l = self._logger
        mname = method.__name__

        # If more than 1 second, mark warning
        if   tdiff>10: l.critical(f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        elif tdiff> 5: l.error(   f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        elif tdiff> 1: l.warning( f"WALLTIME: \t {mname}: {tdiff:.4f} s")
        else: self._logger.debug( f"walltime: \t {mname}: {tdiff:.4f} s")
        return result
    return wrapper

def log_method_args(method):
    def wrapper(self, *args, **kwargs):
        arg_strs = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg_strs.append(f"{arg.shape}")
            elif isinstance(arg, dict):
                arg_strs.append(f"{len(arg)} keys")
            else:
                arg_strs.append(repr(arg))
        self._logger.debug(f"Calling {method.__name__} with args: {', '.join(arg_strs)}")
        return method(self, *args, **kwargs)
    return wrapper

def log_output_size(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if isinstance(result, np.ndarray):
            # If size is more than 1e6, log as warning
            if result.size > 1e8:
                self._logger.error(
                    f"{method.__name__} returned an array of size: {result.size}")
            if result.size > 1e7:
                self._logger.warning(
                    f"{method.__name__} returned an array of size: {result.size}")
            if result.size > 1e6:
                self._logger.info(
                    f"{method.__name__} returned an array of size: {result.size}")
            else:
                self._logger.debug(
                    f"{method.__name__} returned an array of size: {result.size}")
        
        elif isinstance(result, _Mesh):
            # Get total number of elements in the mesh
            total_elements = sum(len(e) for e in result.eidxs.values())
            self._logger.debug(
                f"{method.__name__} returned a mesh with total elements: {total_elements}")
        else:
            self._logger.debug(
                f"{method.__name__} returned a non-array.")
        return result
    
    return wrapper


@dataclass
class _MetaMesh(_Mesh):

    mmesh_name: str = field(default="")
    interconnector: dict[str, MeshInterConnector] = field(default_factory=dict)
    spts_internal: dict[str,np.ndarray[bool]] = field(default_factory=dict)
    recreated: bool = field(default=False)
    ary_here: bool = field(default=False)
    
    # Store variables that are re-used for connectivity functions
    glmap: dict = field(default_factory=dict)
    
    def copy(self, name):
        return replace(self, interconnector={}, mmesh_name=name)

    # If input _Mesh, return a _MetaMesh
    @classmethod
    def from_mesh(cls, mesh: _Mesh, *, name: str = ""):

        etypes = list(sorted(mesh.etypes))

        return cls(
            # Existing attributes
            fname=mesh.fname, raw=mesh.raw, 
            ndims=mesh.ndims, subset=mesh.subset, 
            creator=mesh.creator, codec=mesh.codec, uuid=mesh.uuid, 
            version=mesh.version,
            etypes=etypes, eidxs=cls.preproc_edict(etypes, mesh.eidxs),
            spts_curved=cls.preproc_edict(etypes, mesh.spts_curved),
            spts_nodes=cls.preproc_edict(etypes, mesh.spts_nodes),
            spts=cls.preproc_edict(etypes, mesh.spts, edim=1),
            con=mesh.con, con_p=mesh.con_p, bcon=mesh.bcon,
            eles=cls.preproc_edict(etypes, mesh.eles),
            
            # New attributes
            mmesh_name=name,
            interconnector={},
            spts_internal={},
            recreated=True,
            ary_here=False,
        )
        
    def create_glmap(self):
        self.glmap = [{}]*len(self.etypes)
        for i, etype in enumerate(self.etypes):
            if etype in self.eidxs:
                self.glmap[i] = {k: j for j, k in enumerate(self.eidxs[etype])}

    @property
    def nelems_etype(self):
        return {etype: len(self.eidxs[etype]) for etype in self.etypes}

    @property
    def nelems(self):
        return sum(self.nelems_etype.values())

    @property
    def gcon_p(self):
        """
            Dict of MPI-interface connections per rank and etype.
        """
        return {
                int(nrank): {
                    etype: np.array([self.eidxs[etype][inter[1]] 
                                     for inter in inters if inter[0] == etype ]) 
                       for etype in self.etypes 
                        } 
                    for nrank, inters in self.con_p.items()
               } 

    @property
    def ncon_p_nrank_etype(self):
        """
            Dict of the no. of MPI-interface connections per rank and etype.
        """
        return {
                int(nrank): {
                            etype: len(e) for etype, e in inters.items()
                            } 
                for nrank, inters in self.gcon_p.items()
                }

    @property
    def ncon_p_etype(self):
        """
            Dict of the no. of MPI-interface connections per etype.
        """
        return {
                etype: sum(inters[etype] for inters in self.ncon_p_nrank_etype.values())
                for etype in self.etypes
                }

    @property
    def ncon_p_nrank(self):
        """
            Dict of the no. of MPI-interface connections per rank, sum by etype.
        """
        return {
                int(nrank): sum(inters.values()) 
                for nrank, inters in self.ncon_p_nrank_etype.items()
                }

    @property
    def ncon_p(self):
        """
            Dict of total MPI-interface connections.
        """
        return sum(self.ncon_p_nrank.values())

    def to_mesh(self):

        postproc = lambda x,e=0: self.postproc_edict(x,edim=e)

        return _Mesh(
            fname=self.fname, raw=self.raw,
            ndims=self.ndims, subset=self.subset,
            creator=self.creator, codec = self.codec, uuid = self.uuid,
            version = self.version,
            etypes = self.etypes, eidxs = postproc(self.eidxs),
            spts_curved = postproc(self.spts_curved),
            spts_nodes = postproc(self.spts_nodes),
            spts = postproc(self.spts, 1),
            con = self.con, con_p = self.con_p, bcon = self.bcon,
            eles = postproc(self.eles),
        )

    @staticmethod
    def preproc_edict(etypes, edict_in: dict[str, np.ndarray],*, edim: int = 0):
        """
        Ensure dictionary contains all etypes. 
        If missing etype, add an empty array similar to the existing arrays.
        """

        comm, rank, root = get_comm_rank_root()

        # Make a copy to avoid modifying the original dictionary
        edict = deepcopy(edict_in)

        # For each etype in edict, collect dtype and shape
        for etype in etypes:
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

        # Ensure etypes keys are alphabetically ordered
        edict = {etype: edict[etype] for etype in sorted(edict.keys())}

        return edict

    @staticmethod
    def postproc_edict(edict: dict[str, np.ndarray], *, edim: int = 0):

        edict_out = {}

        for etype in edict:
            if edict[etype].size > 0:
                if edim != 0 and edict[etype].ndim > edim:
                    edict_out[etype] = np.moveaxis(edict[etype], 0, edim)
                else:
                    edict_out[etype] = edict[etype]

        return edict_out

    def collectall(self, param: str, *, 
                   reduce_op: str = None, 
                   reduce_axis: str = None):
        """
        Collects the specified attribute or property across all ranks, with optional reduction.

        Args:
            param (str): The attribute or property name to collect.
            reduce_op (str, optional): The reduction operation to apply ('sum', 'max', etc.). 
                                       If None, no reduction is applied.
            reduce_axis (str, optional): The axis along which to reduce ('rank', 'etype', 'both').
                                         Only applicable if reduce_op is specified.

        Returns:
            Any: The collected (and possibly reduced) data.
        """
        comm, rank, root = get_comm_rank_root()

        # Check if the attribute or property exists
        if not hasattr(self, param):
            raise AttributeError(f"{self.__class__.__name__} has no attribute or property '{param}'")

        # Retrieve the attribute or property value
        attr = getattr(self, param)

        # If it's a method (property with parameters), raise an error
        if callable(attr):
            raise TypeError(f"Attribute '{param}' is callable. "
                f"'collectall' expects a non-callable attribute or property.")

        # Gather the attribute across all ranks
        gathered_attr = comm.allgather(attr)

        # If no reduction is specified, return the gathered data
        if reduce_op is None:

            # Ensure the attribute is a dictionary
            if isinstance(attr, dict) and all(isinstance(k, str) for k in attr.keys()):
                gathered = {}

                for etype in self.etypes:
                    value = attr.get(etype)
                    if value is not None:
                        # Ensure consistent data types across ranks
                        gathered_values = comm.allgather(value)
                    else:
                        # If a rank doesn't have data for this etype, 
                        #   gather an empty array
                        # You can modify this as needed
                        # Here, we assume float dtype. 
                        #   Adjust based on your actual data type.
                        gathered_values = comm.allgather(np.array([], 
                                                                  dtype=float))
                    gathered[etype] = gathered_values

                return gathered

            else:
                return gathered_attr

        # Perform reduction
        if reduce_axis == 'rank':
            # Reduce across ranks
            reduced_attr = self._reduce_across_ranks(gathered_attr, reduce_op)
        elif reduce_axis == 'etype':
            # Reduce across etypes
            reduced_attr = self._reduce_across_etypes(gathered_attr, reduce_op)
        elif reduce_axis == 'both':
            # Reduce across both ranks and etypes
            reduced_attr = self._reduce_across_both(gathered_attr, reduce_op)
        else:
            raise ValueError(f"Invalid reduce_axis '{reduce_axis}'. "
                             f"Must be 'rank', 'etype', or 'both'.")

        return reduced_attr

    def _reduce_across_ranks(self, gathered_attr, reduce_op):
        comm, rank, root = get_comm_rank_root()

        # Determine the reduction operation
        mpi_op = self._get_mpi_op(reduce_op)

        # Local attribute
        local_attr = gathered_attr[rank]

        # Handle different data types
        if isinstance(local_attr, dict):
            # Assume dict with etypes or other keys
            reduced_attr = {}
            keys = set().union(*(d.keys() for d in gathered_attr))
            for key in keys:
                values = [d.get(key, 0) for d in gathered_attr]
                # Reduce values
                local_value = values[rank]
                reduced_value = comm.allreduce(local_value, op=mpi_op)
                reduced_attr[key] = reduced_value
        elif isinstance(local_attr, np.ndarray):
            # Reduce arrays element-wise
            # Stack arrays from all ranks
            reduced_attr = np.empty_like(local_attr)
            comm.Allreduce(local_attr, reduced_attr, op=mpi_op)
        elif isinstance(local_attr, (int, float, np.number)):
            # Reduce scalar values
            reduced_attr = comm.allreduce(local_attr, op=mpi_op)
        else:
            raise TypeError(f"Unsupported data type for reduction: {type(local_attr)}")

        return reduced_attr

    def _reduce_across_etypes(self, gathered_attr, reduce_op):
        # Assume each rank's attribute is a dict with etypes as keys
        # Reduce across etypes within each rank

        numpy_op = self._get_numpy_op(reduce_op)

        local_attr = gathered_attr[self.rank]

        if not isinstance(local_attr, dict):
            raise TypeError("Attribute must be a dictionary with etypes as keys to reduce across etypes.")

        # Reduce values across etypes
        reduced_value = numpy_op([v for v in local_attr.values()])

        return reduced_value

    def _reduce_across_both(self, gathered_attr, reduce_op):
        # First reduce across etypes within each rank
        reduced_across_etypes = self._reduce_across_etypes(gathered_attr, reduce_op)

        # Gather the reduced values from all ranks
        gathered_reduced = self.comm.allgather(reduced_across_etypes)

        # Now reduce across ranks
        reduced_attr = self._reduce_across_ranks(gathered_reduced, reduce_op)

        return reduced_attr

    def _get_mpi_op(self, reduce_op):
        if   reduce_op ==  'sum': return mpi.SUM
        elif reduce_op ==  'max': return mpi.MAX
        elif reduce_op ==  'min': return mpi.MIN
        elif reduce_op == 'prod': return mpi.PROD
        else: raise ValueError(f"Unsupported MPI reduction op: '{reduce_op}'.")

    def _get_numpy_op(self, reduce_op):
        if   reduce_op ==  'sum': return np.sum
        elif reduce_op ==  'max': return np.max
        elif reduce_op ==  'min': return np.min
        elif reduce_op == 'prod': return np.prod
        else: raise ValueError(f"Unsupported NumPy reduction op: '{reduce_op}'.")

@dataclass
class _MetaMeshes:
    """
    This class handles all the mmeshes created during the load balancing process.
    It tracks all mmeshes; _MetaMesh objects enhanced from _Mesh objects.
    """

    mmeshes: Dict[str, _MetaMesh] = field(default_factory=dict)

    etypes: List[str] = field(default_factory=list)
    fname: str = ""
    gnelems_etype: Dict[str, int] = field(default_factory=dict)
    gnelems: int = field(default=0)

    def add_mmesh(self, name: str, mmesh: _MetaMesh, if_base = False):

        # Ensure no other mmeshes exist before 1st mesh is added
        if if_base and self.mmeshes:
            raise ValueError(
                "Base mesh MUST be called once. The solution resides in here.")

        self.mmeshes[name] = mmesh
        self.mmeshes[name].mmesh_name = name

        if if_base:

            # Initialise logger
            self._logger = self.initialise_logger(__name__, LOG_LEVEL)

            comm, rank, root = get_comm_rank_root()

            self.etypes = mmesh.etypes
            self.fname = mmesh.fname
            self.gnelems_etype = {etype: comm.allreduce(len(mmesh.eidxs[etype]), 
                                                op=mpi.SUM) 
                                        for etype in mmesh.etypes}

            self.gnelems = sum(self.gnelems_etype.values())

            self.mmeshes[name].recreated = True
            self.mmeshes[name].ary_here = True
        else:
            self.mmeshes[name].ary_here = False

            if LOG_LEVEL < 30:
                # Check all those set by the base mmesh with this one
                if self.etypes != mmesh.etypes:
                    raise ValueError(
                        f"etypes mismatch: {self.etypes} != {mmesh.etypes}")
                
                if self.fname != mmesh.fname:
                    raise ValueError(
                        f"fname mismatch: {self.fname} != {mmesh.fname}")
                
                mmesh_gnelems_etype = {etype: 
                    len(mmesh.eidxs[etype]) for etype in mmesh.etypes}

                if self.gnelems_etype != mmesh_gnelems_etype:

                    raise ValueError(
                        f"nelems mismatch: {self.gnelems_etype} != {mmesh_gnelems_etype}")
    
    def update_mmesh(self, name: str, mmesh: _MetaMesh):
        self.mmeshes[name] = mmesh

    def get_mmesh(self, name: str) -> _MetaMesh:
        if name not in self.mmeshes:
            raise KeyError(f"Mesh '{name}' not found.")
        return self.mmeshes.get(name)

    def remove_mmesh(self, name: str):
        if name not in self.mmeshes:
            raise KeyError(f"Mesh '{name}' not found.")
        del self.mmeshes[name]

    def move_mmesh(self, src_name, dest_name):
        """
        Move src-mesh to dest-mesh.
        """

        # Check if src and dest exist
        if src_name not in self.mmeshes:
            raise KeyError(f"Mesh '{src_name}' not found.")
        else:
            if dest_name not in self.mmeshes:
                raise KeyError(f"Mesh '{dest_name}' not found.")

            # Ensure src is re-created
            if not self.mmeshes[src_name].recreated:
                raise ValueError(f"Mesh '{src_name}' has not been recreated.")

        # Delete all src --> dest connections
        for mesh_name, mmesh in self.mmeshes.items():
            if src_name in mmesh.interconnector:
                self._logger.info(f"Removing connection: src:{src_name} <-- mesh:{mesh_name}")
                del mmesh.interconnector[src_name]

        # Delete all dest --> src connections
        if dest_name in self.mmeshes[src_name].interconnector:
            self._logger.info(f"Removing connection: src:{dest_name} <-- dest:{src_name}")
            del self.mmeshes[src_name].interconnector[dest_name]

        self.mmeshes[dest_name] = self.mmeshes[src_name]
        self.remove_mmesh(src_name)
        self.mmeshes[dest_name].mmesh_name = dest_name

    def copy_mmesh(self, src_name, dest_name, new_eidxs = None):
        """
        Copy source meta-mesh to destination meta-mesh.
        """
        if src_name not in self.mmeshes:
            raise KeyError(f"Mesh '{src_name}' not found.")
        else:
            if dest_name in self.mmeshes:
                raise KeyError(f"Mesh '{dest_name}' already exists.")
        
            # If recreated is False, give error
            if not self.mmeshes[src_name].recreated: 
                raise ValueError(
                    f"Copying a mesh that has not yet been recreated! "
                    f"Mesh: {src_name}")

        self.mmeshes[dest_name] = self.mmeshes[src_name].copy(dest_name) 

        if new_eidxs is not None:
            self.mmeshes[dest_name].eidxs = self.mmeshes[src_name].preproc_edict(self.etypes, new_eidxs)
            self.connect_mmeshes(src_name, dest_name, both_ways=True)
            self.get_mmesh(src_name).interconnector[dest_name].recreate_reorder_recreate_mesh()
            self.mmeshes[dest_name].recreated = True

    def connect_mmeshes(self, src_name: str, dest_name: str, 
                       both_ways: bool = False):
        if not src_name in self.mmeshes:
            # Print all mmeshes first, then raise error
            print(f"Available mmeshes: {list(self.mmeshes.keys())}")
            raise KeyError(f"Missing mesh: {src_name}.")
            
        if not dest_name in self.mmeshes:
            print(f"Available mmeshes: {list(self.mmeshes.keys())}")
            raise KeyError(f"Missing mesh: {dest_name}.") 

        # For convinience, set variable names
        src_mmesh = self.mmeshes[src_name]
        dest_mmesh = self.mmeshes[dest_name]

        # Check if either has not been recreated
        if not src_mmesh.recreated:
            print(f"Available mmeshes: {list(self.mmeshes.keys())}")
            raise ValueError(f"{src_name} not yet recreated! ")

        if not dest_mmesh.recreated:
            self._logger.warning(f"{dest_name} not yet recreated! ")

        # If connection already exists, then error
        if dest_name in src_mmesh.interconnector:
            raise ValueError(f"Connection exists: {src_name} <-- {dest_name}")

        src_mmesh.interconnector[dest_name] = MeshInterConnector(
                                                        src_mmesh, dest_mmesh)
        if both_ways:
            self.connect_mmeshes(dest_name, src_name)

    def __str__(self):
        comm, rank, root = get_comm_rank_root()

        etypes = self.etypes
        num_ranks = comm.size
        columns = ['Mesh-name']

        # For each etype and each rank ...
        for etype in etypes:
            for r in range(num_ranks):
                columns.append(f'{etype}-{r}')

        for etype in etypes:
            columns.append(f'{etype}-all')

        columns.extend(['Total', 
                        #'Connections', # MESH CONNECTIONS 
                        # 'Recreated', 
                        'Solution Here'
                        ])

        # Now, collect data from all ranks
        # For each mesh, we need to collect per-rank element counts
        mesh_info_per_rank = {}
        for mesh_name, mesh in self.mmeshes.items():
            # For this rank, get the element counts per etype
            ecounts = {etype: len(mesh.eidxs.get(etype, [])) for etype in etypes}
            # Store in mesh_info_per_rank
            mesh_info_per_rank[mesh_name] = {'ecounts': ecounts}

        # Now, gather this data from all ranks
        all_mesh_info = comm.gather(mesh_info_per_rank, root=root)

        if rank == root:
            # Now, we have all_mesh_info as a list of dicts from each rank
            # We need to process this to build the table

            # Initialize a list for the rows
            rows = []

            for mesh_name in self.mmeshes.keys():
                # Initialize a dict to hold per-etype, per-rank counts
                etype_rank_counts = {}
                for etype in etypes:
                    etype_rank_counts[etype] = [0] * num_ranks

                # Collect per-rank counts
                for r, mesh_info in enumerate(all_mesh_info):
                    if mesh_name in mesh_info:
                        ecounts = mesh_info[mesh_name]['ecounts']
                        for etype in etypes:
                            etype_rank_counts[etype][r] = ecounts.get(etype, 0)
                    else:
                        # Mesh not present on this rank
                        pass

                # Compute per-etype totals across all ranks
                nelem_etype = {etype: 
                            sum(etype_rank_counts[etype]) for etype in etypes}

                # Compute total elements across all etypes and ranks
                total_elements = sum(nelem_etype.values())

                # MESH CONNECTIONS
                #mesh = self.mmeshes[mesh_name]
                #inter_mesh = list(mesh.interconnector.keys())
                #inter_mesh_str = ', '.join(inter_mesh) if inter_mesh else 'None'

                # Build the row
                row = [mesh_name]                       
                for etype in etypes:
                    for r in range(num_ranks):
                        row.append(str(etype_rank_counts[etype][r]))

                for etype in etypes:               
                    row.append(str(nelem_etype[etype]))
                row.append(str(total_elements))    
                #row.append(inter_mesh_str)             # MESH CONNECTIONS   
                #row.append(str(mesh.recreated))    
                row.append(str(mesh.ary_here))     
                rows.append(row)                   

            # Now, format the table
            table = tabulate(rows, headers=columns, tablefmt='grid')

            return table
        else:
            return ''

    def short_str(self):
        """
        A shortened version of mesh statistics towards logging.
        For now, only print the number of elements for 'compute' mesh.
        """
        self._logger.info(f"Compute-mesh: {self.get_mmesh('compute').nelems} elements")

    def initialise_logger(self, name, level):
        comm, rank, root = get_comm_rank_root()
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        file_handler = logging.FileHandler(
            f'logging/{name}-{rank}.log')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s  - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("_______________________________________________")
        logger.info(f"{name} initialized.") 

        return logger


class LoadRelocator():
    # Initialise as an empty list
    etypes = []

    def __init__(self, base_mesh: _Mesh, tol, K_p=1, 
                                              K_i=0, 
                                              K_d=0, 
                                              K_win=2):

        self._logger = self.initialise_logger(__name__, LOG_LEVEL)

        self.mm = _MetaMeshes()
        self.mm.add_mmesh('base', _MetaMesh.from_mesh(base_mesh), if_base=True)
        self.mm.copy_mmesh('base', 'compute')

        self.tol = tol

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.K_win = K_win

        self.targets_hist = deque(maxlen=2)

    def observe(self, mesh_name, perfs, times):

        comm, rank, root = get_comm_rank_root()
        
        # Just use one-liner here instead of function and then its using
        gathered_perfs = {key: np.array(comm.allgather(value)) for key, value in perfs.items()}
        gathered_times = {key: np.array(comm.allgather(value)) for key, value in times.items()}

        # COST OPTION 1: Target performance
        self.rank_cost = gathered_perfs['target'] / sum(gathered_perfs['target'])
        self.if_maximise = True

        # COST OPTION 2: Wait time
        # ISSUE: Making partitioning weights based on wait time biases parititoning from the start.
        #self.rank_cost = gathered_times['lost'] / sum(gathered_times['lost'])
        #self.if_maximise = False

        # COST OPTION 3: Lost performance
        #self.rank_cost = gathered_perfs['lost'] / sum(gathered_perfs['lost'])
        #self.if_maximise = False

        self._logger.info(f"Cost: {self.rank_cost}")

        sure_gains_nelems = self.mm.gnelems * self.rank_cost[rank]/sum(self.rank_cost)
        curr_nelems = self.mm.get_mmesh(mesh_name).nelems
        sure_gains_diff = sure_gains_nelems - curr_nelems

        # Use PID controller
        control_output  =  self.K_p * sure_gains_diff
        control_output +=  self.K_i * (self.targets_hist[-1] + self.targets_hist[-2] - 2*sure_gains_diff)/2 \
                         + self.K_d * (self.targets_hist[-1] - self.targets_hist[-2]) if len(self.targets_hist) == 2 else 0
        target_nelems = curr_nelems + control_output
        nelems_diff = (target_nelems - curr_nelems)*(self.if_maximise - 0.5)*2

        self.targets_hist.append(target_nelems)

        # Print all of them for debugging
        self. _logger.error(f"{sure_gains_diff = }, \t{control_output = } = {self.K_p * sure_gains_diff} + {self.K_i * np.mean(self.targets_hist)}, \t{target_nelems = }, \t{nelems_diff = }")

        # If any of the target elements are negative, throw error
        if target_nelems < 0:
            raise ValueError(f"Target elements cannot be negative: rank {rank}: {target_nelems}."
                             "Possible causes:\n"
                                "- Insufficiently loaded ranks.\n"
                                "- Improperly tuned PID controller.\n")

        # Check if the total number of elements across all ranks is still equal to gnelems
        if comm.allreduce(nelems_diff, op=mpi.SUM) > 1:
            raise ValueError(f"Total number of elements across all ranks is not equal to gnelems.")

        return target_nelems, nelems_diff

    @log_method_times
    def diffuse(self, mesh_name, target_nelems, nelems_diff, ary):
        comm, rank, root = get_comm_rank_root()

        if LOG_LEVEL < 30:
            # If ary does not exist in the compute mesh, error
            if not self.mm.get_mmesh(mesh_name).ary_here:
                raise ValueError("Solution does not exist in compute mesh.")
            
            ary_shape = [s.shape[2] for s in ary]
            mesh_shape = [len(self.mm.get_mmesh(mesh_name).eidxs[etype]) for etype in self.mm.etypes]
            
            # Compare
            if ary_shape != mesh_shape: 
                raise ValueError(f"ary shape {ary_shape} mismatch with mesh shape {mesh_shape}")
            else:
                self._logger.debug(f"ary shape matches mesh shape.")

            sum_ary_allreduced = np.array(comm.allreduce(np.sum(ary), op=mpi.SUM))

        # Create a mesh called mesh_name+'_base'. 
        self.mm.copy_mmesh(mesh_name, mesh_name+'_base')

        # Solution must now only be in the `mesh_name` mesh
        for m in self.mm.mmeshes:
            self.mm.mmeshes[m].ary_here = m == mesh_name
        
        ii = 0
        abs_nelems_diff = comm.allreduce(np.abs(nelems_diff), op=mpi.SUM)
        abs_target_nelems = comm.allreduce(target_nelems, op=mpi.SUM)
        if_diffuse_all = not abs_nelems_diff/abs_target_nelems < self.tol
        if_diffuse = not comm.allreduce(np.abs(nelems_diff/target_nelems) < self.tol/comm.size, op=mpi.MIN)
        if_high_movement = comm.allreduce(np.abs(nelems_diff) > 1, op=mpi.MIN)
        if_only_2_elems = comm.allreduce(target_nelems == 2, op=mpi.MIN)
        while if_diffuse_all and if_diffuse and if_high_movement and not if_only_2_elems:

            ii += 1
            nelems_diff = target_nelems - self.mm.get_mmesh(mesh_name+'_base').nelems

            self._logger.info(f"no. elements moved out: {nelems_diff}")
            self._logger.info(f"No. of interfaces: {self.mm.get_mmesh(mesh_name+'_base').ncon_p_nrank_etype}")

            to_nrank = self.figure_out_move_to_nrank(nelems_diff, self.mm.get_mmesh(mesh_name+'_base'))
            move_elems = self.reloc_interface_elems(to_nrank, self.mm.get_mmesh(mesh_name+'_base'))

            print(f"rank {rank} \t "
                  f"iter{ii}: "
                  f"Start: {self.mm.get_mmesh(mesh_name).nelems} "
                  f" \t Currently: {self.mm.get_mmesh(mesh_name+'_base').nelems} --> \t ("
                  f" {target_nelems - self.mm.get_mmesh(mesh_name+'_base').nelems}"
                  f") --> \t {target_nelems}", flush=True)

            # Using reference mesh, create a temporary mesh
            self.mm.copy_mmesh(mesh_name+'_base', mesh_name+'-temp', 
                               self.get_preordered_eidxs(move_elems, 
                                         self.mm.get_mmesh(mesh_name+'_base')))
            self.mm.move_mmesh(mesh_name+'-temp', mesh_name+'_base')


            abs_nelems_diff = comm.allreduce(np.abs(nelems_diff), op=mpi.SUM)
            abs_target_nelems = comm.allreduce(target_nelems, op=mpi.SUM)
            if_diffuse_all = not abs_nelems_diff/abs_target_nelems < self.tol

            if_diffuse = not comm.allreduce(np.abs(nelems_diff/target_nelems) < self.tol, op=mpi.MIN)
            if_high_movement = comm.allreduce(np.abs(nelems_diff) > 1, op=mpi.MIN)

        # Set new array and solution here
        self.mm.connect_mmeshes(mesh_name+'_base', mesh_name)

        # Re-create diffuion non-essential mesh attributes we skipped
        self.mm.get_mmesh(mesh_name+'_base').spts        = self.reloc(mesh_name, mesh_name+'_base', self.mm.get_mmesh(mesh_name).spts,      edim=0)
        self.mm.get_mmesh(mesh_name+'_base').spts_curved = self.reloc(mesh_name, mesh_name+'_base', self.mm.get_mmesh(mesh_name).spts_curved, edim=0)

        # Re-create ary
        new_mesh = self.mm.get_mmesh(mesh_name+ '_base').to_mesh()
        new_ary = list(self.reloc(mesh_name, mesh_name+'_base',  
                {m:s for m,s in zip(self.mm.etypes, ary)}, edim=2).values()
                           )

        # Finally, replace mesh_name with mesh_name+'_base'
        self.mm.move_mmesh(mesh_name+'_base', mesh_name)
            
        # Solution must now only be in the final mesh
        for m in self.mm.mmeshes:
            self.mm.mmeshes[m].ary_here = m == mesh_name
            
        print(self.mm)

        if LOG_LEVEL < 30:
            new_mesh_shape = [len(new_mesh.eidxs[etype]) for etype in new_mesh.etypes]
            new_ary_shape = [s.shape[2] for s in new_ary]

            if new_ary_shape != new_mesh_shape:
                raise ValueError(f"Solution shape {new_ary_shape} does not match mesh shape {new_mesh_shape}")
            else:
                self. _logger.debug(f"Successfully recreated mesh and solution.")

            new_sum_ary_allreduced = np.array(comm.allreduce(np.sum(new_ary), op=mpi.SUM))

            error = np.abs(new_sum_ary_allreduced - sum_ary_allreduced)/sum_ary_allreduced

            if np.any(error > 1e-10):
                raise ValueError(f"Ary-sum mismatch of the order of 1e-10. " 
                                f"Expected {sum_ary_allreduced}, got {new_sum_ary_allreduced}.")
            else:
                self._logger.debug(f"Sum solution after relocation matches sum solution before relocation.")

        return new_mesh, new_ary

    def reloc(self, src_mesh_name : str, dest_mesh_name: str,
                 edict: dict[str, np.ndarray],*, edim) -> dict[str, np.ndarray]:

        # Check that the numpy arrays in edict are of the same shape as the source mesh 

        # For convinience, set the _MetaMesh instance to a variable
        dest_mmesh = self.mm.get_mmesh(dest_mesh_name)

        edict = dest_mmesh.preproc_edict(self.mm.etypes, edict, edim=edim)
        relocated_dict = dest_mmesh.interconnector[src_mesh_name].relocate(edict)
        return dest_mmesh.postproc_edict(relocated_dict, edim=edim)

    def figure_out_move_to_nrank(self, nelems_diff: int, mesh: _MetaMesh):
        comm, rank, root = get_comm_rank_root()

        ncon_p_nrank_etype = mesh.ncon_p_nrank_etype
        ncon_p_nrank       = mesh.ncon_p_nrank
        ncon_p             = mesh.ncon_p

        nranks = comm.size

        move_to_nrank = np.zeros((nranks, nranks))
        for nrank, inters in ncon_p_nrank_etype.items():
            if nelems_diff != 0:
                move_to_nrank[rank,nrank] = -nelems_diff * ncon_p_nrank[nrank] / ncon_p

        move_to_nrank = comm.allreduce(move_to_nrank, op=mpi.SUM)
        move_to_nrank = np.maximum((move_to_nrank - move_to_nrank.T) / 2, 0)
        self._logger.info(f"Movement of elements to nrank: {move_to_nrank}")

        return move_to_nrank

    def reloc_interface_elems(self, move_to_nrank: np.ndarray, 
                                    mesh: _MetaMesh,
                                    ):

        comm, rank, root = get_comm_rank_root()

        gcon_p             = mesh.gcon_p
        ncon_p_nrank_etype = mesh.ncon_p_nrank_etype
        ncon_p             = mesh.ncon_p

        # Break into loops
        move_elems = {nrank: {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes} for nrank in mesh.con_p.keys()}
        for nrank in mesh.con_p.keys():
            for etype in self.mm.etypes:
                
                #move_nelems_nrank_etype = np.random.rand(ncon_p_nrank_etype[nrank][etype]) < min(1, move_to_nrank[rank][nrank] /ncon_p)

                # Number of elements available to move for this etype with nrank
                n_available = ncon_p_nrank_etype[nrank][etype]

                # Number of elements to move for this etype with nrank
                n_to_move = int(np.round(move_to_nrank[rank][nrank] * n_available / ncon_p))

                # Ensure n_to_move does not exceed available elements
                n_to_move = min(n_to_move, n_available)

                if n_available == 0:
                    # No elements to move
                    move_elemsf = np.array([], dtype=np.int32)
                elif n_to_move >= n_available:
                    # Move all elements
                    move_elemsf = gcon_p[nrank][etype]
                else:
                    # Randomly select n_to_move elements from available elements
                    move_elemsf = np.random.choice(gcon_p[nrank][etype], size=n_to_move, replace=False)

                move_elemsf = [intelem for intelem, move in zip(gcon_p[nrank][etype], move_elemsf) if move]

                move_elemsf = np.array(move_elemsf, dtype=np.int32)
                move_elemsf = np.sort(move_elemsf)
                move_elemsf = np.unique(move_elemsf)

                move_elems[nrank][etype] = move_elemsf
        
        # Empty if self
        move_elems[rank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
        move_elems_f = {nrank: move_elems[nrank] for nrank in sorted(move_elems.keys())}

        return self.remove_duplicate_movements(move_elems_f, mesh)

    def remove_duplicate_movements(self, 
                                   move_elems: dict[int, dict[str, np.ndarray]],
                                   mesh: _Mesh):
        """
        Remove duplicate movements from the move_to_nrank dictionary.
        """

        move_initial = {
            nrank: {
                etype: 
                    len(elems) for etype, elems in etypeelems.items()
                    } 
                        for nrank, etypeelems in move_elems.items()
                        }

        # Prioritise decreasing load on the costliest ranks first, use self.maximise
        sorted_ranks = sorted(mesh.con_p.keys(), key=lambda x: self.rank_cost[x], reverse=self.if_maximise)

        # Step 8: Remove duplicates by iterating over sorted ranks
        for i, nrank in enumerate(sorted_ranks):
            for etype in self.mm.etypes:
                for nnrank in sorted_ranks[i+1:]:
                    move_elems[nnrank][etype] = np.setdiff1d(
                        move_elems[nnrank][etype], move_elems[nrank][etype]
                    )

        move_final = {
                      nrank: {
                              etype: len(elems) 
                              for etype, elems in etypeelems.items()
                             } for nrank, etypeelems in move_elems.items()
                     }

        if LOG_LEVEL < 30:

            temp = {
                nrank: {
                    etype: move_initial[nrank][etype] - move_final[nrank][etype] 
                            for etype in self.mm.etypes
                    } for nrank in mesh.con_p.keys()
                }


            # Logg the number of elements moved to each rank, flattened 1D
            self._logger.info(f"Duplicates removed: { temp }")

        return move_elems
    
    @log_method_times
    def get_preordered_eidxs(self, 
                             move_to_nrank: dict[int, dict[str, np.ndarray]],
                             mesh: _MetaMesh) -> dict[str, np.ndarray[int]]:
        """
        Get new element indices on each rank after relocation.
        """

        comm, rank, root = get_comm_rank_root()

        moved_from_nrank0 = MeshInterConnector._send_recv0(self.mm.etypes, move_to_nrank)

        new_mesh_eidxs = deepcopy(mesh.eidxs)

        # preprocs new_mesh_eidxs
        new_mesh_eidxs = mesh.preproc_edict(self.mm.etypes, new_mesh_eidxs)

        for etype in self.mm.etypes:
            elements_to_remove = np.sort(np.concatenate([elements[etype] for elements in move_to_nrank.values()]))
            new_mesh_eidxs[etype] = np.setdiff1d(new_mesh_eidxs[etype], elements_to_remove)

        for etype in self.mm.etypes:
            elements_to_add = np.sort(moved_from_nrank0[etype])
            new_mesh_eidxs[etype] = np.unique(np.concatenate((new_mesh_eidxs[etype], elements_to_add)).astype(np.int32))

        if LOG_LEVEL < 30:

            # Count the overall number of elements
            ntotal_final = self.count_all_eidxs(new_mesh_eidxs)

            # Ensure the total number of elements is preserved
            if rank == root and sum(self.mm.gnelems_etype.values()) != ntotal_final:
                raise ValueError(f"Total number of elements changed during relocation. "
                                f"Expected {sum(self.mm.gnelems_etype.values())}, "
                                f"got {ntotal_final}.")

        return new_mesh_eidxs

    def get_gcon_p(self, mesh: _Mesh):
        return {
                nrank: {
                etype: np.array([mesh.eidxs[etype][inter[1]] 
                                 for inter in inters if inter[0] == etype ]) 
                       for etype in mesh.etypes 
                    } 
                for nrank, inters in mesh.con_p.items()
               } 

    def initialise_logger(self, name, level):
        comm, rank, root = get_comm_rank_root()
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        file_handler = logging.FileHandler(f'logging/{name}-{rank}.log')
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s  - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("_______________________________________________")
        logger.info(f"{name} initialized.") 

        return logger

    @staticmethod
    def count_all_eidxs(mesh_eidxs):
        comm, rank, root = get_comm_rank_root()
        total_elements = sum(len(e) for e in mesh_eidxs.values())
        return comm.allreduce(total_elements, op=mpi.SUM)

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
            #comm.Barrier()

class MeshInterConnector(AlltoallMixin):

    def __init__(self, mmesh: _MetaMesh, target: _MetaMesh = None):

        # Get loggers
        self._logger = self.initialise_logger(__name__, LOG_LEVEL)

        self.comm, self.rank, self.root = get_comm_rank_root()

        self.mmesh = mmesh
        self.etypes = list(mmesh.eidxs.keys())
        self.neighbours = [i for i in range(self.comm.size) if i != self.rank]

        self.mmesh_eidxs_gathered = mmesh.collectall('eidxs')

        if len(set(self.mmesh_eidxs_gathered.keys())) != len(self.etypes):
            raise ValueError(f"All ranks must have all etypes in mmesh.eidxs.")

        self._inv_scount = {}
        self._inv_sdisp = {}
        self._inv_rcount = {}
        self._inv_rdisp = {}
        self._inv_sidxs = {}
        self._inv_ridxs = {}

        if target is not None:

            self.target = target
            self.target_eidxs_gathered = target.collectall('eidxs')

            for etype in self.etypes:
                self.relocation_indexing(etype)

    def copy(self):
        """
        Create a deep copy of the MeshInterConnector instance.
        Note: We avoid copying MPI communicators and other non-copyable attributes.
        """
        new_copy = MeshInterConnector.__new__(MeshInterConnector)
        new_copy.comm = self.comm
        new_copy.rank = self.rank
        new_copy.root = self.root
        new_copy.neighbours = self.neighbours.copy()
        new_copy.etypes = self.etypes.copy()
        new_copy.mmesh_eidxs_gathered = deepcopy(self.mmesh_eidxs_gathered)

        new_copy._inv_scount = deepcopy(self._inv_scount)
        new_copy._inv_sdisp = deepcopy(self._inv_sdisp)
        new_copy._inv_rcount = deepcopy(self._inv_rcount)
        new_copy._inv_rdisp = deepcopy(self._inv_rdisp)
        new_copy._inv_sidxs = deepcopy(self._inv_sidxs)
        new_copy._inv_ridxs = deepcopy(self._inv_ridxs)

        # Copy other attributes
        if hasattr(self, '_logger'):
            new_copy._logger = self._logger  # Share the same logger

        return new_copy

    @staticmethod
    def _send_recv0(etypes, send_elements: dict[int, dict[str, np.ndarray]]):

        mixin = AlltoallMixin()

        comm, rank, root = get_comm_rank_root()

        recv_elements = {}
        
        # Convert to 1D array for each etype
        for etype in etypes:
            _scount = np.array([len(send_elements[nrank][etype]) for nrank in range(comm.size)], dtype=np.int32)
            _sdisp = mixin._count_to_disp(_scount)
            _sidxs = np.concatenate([send_elements[nrank][etype] for nrank in send_elements]).astype(np.int32)

            # Communicate to get recieve counts and displacements
            _, (_rcount, _rdisp) = mixin._alltoallcv(comm, 
                                                     _sidxs, _scount, _sdisp)

            # Allocate recieve indices array
            _ridxs = np.empty((_rcount.sum(), *_sidxs.shape[1:]), 
                              dtype=_sidxs.dtype)

            mixin._alltoallv(comm, 
                             (_sidxs, (_scount, _sdisp)),
                             (_ridxs, (_rcount, _rdisp)))

            #LoadRelocator.cprint(_ridxs, name=f"Rank {rank} received elements for {etype}")

            recv_elements[etype] = _ridxs

        return recv_elements

    @log_method_times
    def recreate_reorder_recreate_mesh(self):
        for etype in self.etypes:
            self.target.interconnector[self.mmesh.mmesh_name].relocation_indexing(etype)

        self._recreate_mesh_for_diffusion()                                     
        self._reconstruct_con_conp_bcon()                                       
        self._reorder_elements(self.target)                             
        self._invert_recreated_mesh()

        # WE ONLY RE-ORDER ELEMENTS HERE
        eidxs = self.target.collectall('eidxs')
        self.target_eidxs_gathered = eidxs
        self.target.interconnector[self.mmesh.mmesh_name].mmesh_eidxs_gathered = eidxs
        for etype in self.etypes:
            self.relocation_indexing(etype)
            self.target.interconnector[self.mmesh.mmesh_name].relocation_indexing(etype)

        self._recreate_mesh_for_diffusion()

        self._reconstruct_con_conp_bcon()

    @log_method_times
    def _invert_recreated_mesh(self):

        self.target.eles        = self.relocate(self.target.eles)
        self.target.spts_curved = self.relocate(self.target.spts_curved)

    @log_method_times
    def _recreate_mesh_for_diffusion(self):

        self.target.eles        = self.target.interconnector[self.mmesh.mmesh_name].relocate(self.target.eles)
        self.target.spts_curved = self.target.interconnector[self.mmesh.mmesh_name].relocate(self.target.spts_curved)

    @log_method_times
    def _recreate_mesh_after_diffusion(self):
        self.target.spts       = self.target.interconnector[self.mmesh.mmesh_name].relocate(self.target.spts)
        self.target.spts_nodes = self.target.interconnector[self.mmesh.mmesh_name].relocate(self.target.spts_nodes)

    @log_method_times
    def _reorder_elements(self, mesh: _Mesh):
        """
        Reorder the elements based on the following criteria:
        MPI-curved --> MPI-linear (treated as curved) --> curved --> linear
        """

        new_target_eidxs = {etype: [] for etype in self.etypes}

        for etype in self.etypes:
            ordered_idx = np.lexsort(( mesh.spts_curved[etype], 
                                       mesh.spts_internal[etype])) 
            if len(ordered_idx) > 0:
                new_target_eidxs[etype] = mesh.eidxs[etype][ordered_idx]
            else:
                new_target_eidxs[etype] = mesh.eidxs[etype]

        mesh.eidxs = new_target_eidxs

    @log_method_times
    def relocation_indexing(self, etype):
        """
        Create 1D arrays of send and receive indices for the given element type.
        """

        self._logger.info(f"Inverse-mapping mesh eidxs for {etype}")
        base_mesh_eidxs = self.target_eidxs_gathered
        end_mesh_eidxs = self.mmesh_eidxs_gathered
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

    def relocate(self, sary_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {etype: self._reloc_ary(etype, sary) for etype, sary in sary_dict.items()}

    def _reloc_ary(self, etype: str, ary: np.ndarray) -> np.ndarray:
        """
        Relocate data across ranks along elements dimension for each etype.

        Args:
            etype (str): The element type.
            ary (np.ndarray): The array to relocate or unrelocate.
            invert (bool): Relocate if false, else unrelocate.

        Returns:
            np.ndarray: Relocated/unrelocated array.
        """

        # Unrelocation
        base_idxs = self.target_eidxs_gathered[etype][self.rank]
        end_idxs = self.mmesh_eidxs_gathered[etype][self.rank]
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

    @log_method_times
    def _reconstruct_con_conp_bcon(self):

        mesh = self.target
        eles = self.target.eles

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

        # MPI connectivity

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

        # Find elements sharing an MPI interface
        ifmpi = {etype: [] for etype in self.etypes}

        for nrank, ncon in self.target.con_p.items():
            for etype, eidx, fidx in ncon:
                ifmpi[etype].append(eidx)

        # Create a sorted list of unique elements
        for etype in self.etypes:
            ifmpi[etype] = np.unique(ifmpi[etype])

        # `lexsort` needs boolean array of if-internal for each etype
        self.target.spts_internal = {etype: 
                                     np.ones(self.target.eles[etype].shape[0], 
                                            dtype=bool
                                            ) 
                                      for etype in self.etypes
                                    }
        
        for etype, eidxs in ifmpi.items():
            self.target.spts_internal[etype][eidxs.astype(int)] = False

    def _reconstruct_con_p_only(self):

        mesh = self.target
        eles = self.target.eles

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

        # MPI connectivity

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

        # Find elements sharing an MPI interface
        ifmpi = {etype: [] for etype in self.etypes}

        for nrank, ncon in self.target.con_p.items():
            for etype, eidx, fidx in ncon:
                ifmpi[etype].append(eidx)

        # Create a sorted list of unique elements
        for etype in self.etypes:
            ifmpi[etype] = np.unique(ifmpi[etype])

        # `lexsort` needs boolean array of if-internal for each etype
        self.target.spts_internal = {etype: 
                                     np.ones(self.target.eles[etype].shape[0], 
                                            dtype=bool
                                            ) 
                                      for etype in self.etypes
                                    }
        
        for etype, eidxs in ifmpi.items():
            self.target.spts_internal[etype][eidxs.astype(int)] = False

    def initialise_logger(self, name, level):
        comm, rank, root = get_comm_rank_root()
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        file_handler = logging.FileHandler(f'logging/{name}-{rank}.log')
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("_______________________________________________")
        logger.info(f"{name} initialized.") 

        return logger

    def exchange_element_indices(self, send_elements):
        """
        Exchange element indices between ranks using existing functionality.

        Args:
            send_elements: dict[int, dict[str, np.ndarray]], mapping destination rank
                           to dict of element types and element indices.

        Returns:
            recv_elements: dict[int, dict[str, np.ndarray]], mapping source rank
                           to dict of element types and element indices.
        """
        nranks = self.comm.size

        recv_elements = {}

        for etype in self.etypes:
            # Prepare send indices and counts
            svals = []
            scounts = np.zeros(nranks, dtype=int)

            for dest_rank in range(nranks):
                if dest_rank == self.rank:
                    continue
                elems = send_elements.get(dest_rank, {}).get(etype, np.array([], dtype=int))
                scounts[dest_rank] = len(elems)
                svals.append(elems)

            # Concatenate all elements to send
            svals = np.concatenate(svals) if svals else np.array([], dtype=int)

            # Use _alltoallcv to exchange counts and data
            rvals, (rcounts, rdisps) = self._alltoallcv(self.comm, svals, scounts)

            # Reconstruct recv_elements
            offset = 0
            for src_rank in range(nranks):
                if src_rank == self.rank:
                    continue
                count = rcounts[src_rank]
                if count > 0:
                    elems = rvals[offset:offset+count]
                    if src_rank not in recv_elements:
                        recv_elements[src_rank] = {}
                    recv_elements[src_rank][etype] = elems
                    offset += count

        return recv_elements