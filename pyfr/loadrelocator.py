from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import re

import numpy as np
from pyfr.cache import memoize
from tabulate import tabulate
from mpi4py import MPI

from pyfr.mpiutil import get_comm_rank_root, AlltoallMixin, mpi
from pyfr.nputil import iter_struct
from pyfr.readers.native import _Mesh

# Fix seed
BASE_TAG = 47
np.random.seed(BASE_TAG)
LOG_LEVEL=40

from pyfr.logger_utils import initialise_logger, log_method_times

@dataclass
class _MetaMesh(_Mesh):

    mmesh_name: str = field(default="")
    interconnector: dict[str, MeshInterConnector] = field(default_factory=dict)
    spts_internal: dict[str,np.ndarray[bool]] = field(default_factory=dict)
    recreated: bool = field(default=False)
    
    # Store variables that are re-used for connectivity functions
    glmap: dict = field(default_factory=dict)
    
    # Each MetaMesh has its own MPI communicator
    comm: MPI.Comm = field(default_factory=MPI.COMM_WORLD, 
                           repr=False, compare=False)

    ranks_map: dict[int, int] = field(default_factory=dict)

    def copy(self, name):
        return replace(self, interconnector={}, mmesh_name=name)

    # If input _Mesh, return a _MetaMesh
    @classmethod
    def from_mesh(cls, mesh: _Mesh):

        etypes = list(sorted(mesh.etypes))

        world_comm = get_comm_rank_root('world')[0]
        reader_comm = get_comm_rank_root('reader')[0]
        comm = get_comm_rank_root('compute')[0]

        etypes = world_comm.bcast(etypes, root=0)

        mesh.ndims = world_comm.bcast(mesh.ndims, root=0)
        mesh.subset = world_comm.bcast(mesh.subset, root=0)
        mesh.etypes = world_comm.bcast(mesh.etypes, root=0)

        if reader_comm == mpi.COMM_NULL:

            mesh.eidxs       = {etype: np.empty((0, 0)) for etype in etypes}
            mesh.spts        = {etype: np.empty((0, 0)) for etype in etypes} 
            mesh.spts_nodes  = {etype: np.empty((0, 0)) for etype in etypes} 
            mesh.spts_curved = {etype: np.empty((0, 0)) for etype in etypes} 
            mesh.eles        = {etype: np.empty((0, 0)) for etype in etypes} 
            
            mesh.con         = []
            mesh.con_p       = {}
            mesh.bcon        = {}

        eidxs = cls.preproc_edict(etypes, mesh.eidxs)
        spts_curved = cls.preproc_edict(etypes, mesh.spts_curved)
        spts_nodes = cls.preproc_edict(etypes, mesh.spts_nodes)
        spts = cls.preproc_edict(etypes, mesh.spts, edim=1)
        eles = cls.preproc_edict(etypes, mesh.eles)

        return cls(
            # Existing attributes
            fname=mesh.fname, raw=mesh.raw, 
            ndims=mesh.ndims, subset=mesh.subset, 
            creator=mesh.creator, codec=mesh.codec, uuid=mesh.uuid, 
            version=mesh.version,
            etypes=etypes, eidxs=eidxs,
            spts_curved=spts_curved, spts_nodes=spts_nodes, spts=spts,
            con=mesh.con, con_p=mesh.con_p, bcon=mesh.bcon,
            eles=eles,
            
            # New attributes
            mmesh_name=None,
            interconnector={},
            spts_internal={},
            recreated=True,
            
            comm=comm
        )

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

        comm, rank, root = get_comm_rank_root('compute')

        # Make a copy to avoid modifying the original dictionary
        edict = deepcopy(edict_in)

        # For each etype in edict, collect dtype and shape
        for etype in etypes:
            ary = edict.get(etype, None)

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

    def collectall(self, param: str, 
                   *, reduce_op: str = None, reduce_axis: str = None):
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

    def __eq__(self, other):
        if not isinstance(other, _MetaMesh):
            return False

        # Compare simple attributes
        simple_attrs = [
            'mmesh_name', 'recreated',
            'fname', 'raw', 'ndims', 'subset', 'creator',
            'codec', 'uuid', 'version', 'etypes'
        ]
        for attr in simple_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        # Compare dictionaries containing numpy arrays
        dict_array_attrs = [
            'eidxs', 'spts_curved', 'spts_nodes', 'spts',
            'eles', 'spts_internal'
        ]
        for attr in dict_array_attrs:
            if not self._compare_dicts_of_arrays(getattr(self, attr), getattr(other, attr)):
                return False

        # Compare 'con', which is a tuple of lists
        if self.con != other.con:
            return False

        # Compare 'con_p' and 'bcon', which are dictionaries
        if not self._compare_dicts(self.con_p, other.con_p):
            return False
        if not self._compare_dicts(self.bcon, other.bcon):
            return False

        # Compare 'interconnector', which is a dictionary of MeshInterConnector instances
        if not self._compare_interconnector(self.interconnector, other.interconnector):
            return False

        # Compare 'glmap', which is a list of dictionaries
        if not self._compare_list_of_dicts(self.glmap, other.glmap):
            return False

        return True

    def _compare_dicts_of_arrays(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            arr1 = dict1[key]
            arr2 = dict2[key]
            if arr1 is None and arr2 is None:
                continue
            if arr1 is None or arr2 is None:
                return False
            if not np.array_equal(arr1, arr2):
                return False
        return True

    def _compare_dicts(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]
            if val1 != val2:
                return False
        return True

    def _compare_interconnector(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            inter1 = dict1[key]
            inter2 = dict2[key]
            if not inter1 == inter2:
                return False
        return True

    def _compare_list_of_dicts(self, list1, list2):
        if len(list1) != len(list2):
            return False
        for dict1, dict2 in zip(list1, list2):
            if dict1 != dict2:
                return False
        return True

@dataclass
class _MetaMeshes:
    """
    This class handles all the mmeshes created during the load balancing process.
    It tracks all mmeshes; _MetaMesh objects enhanced from _Mesh objects.
    """

    mmeshes: dict[str, _MetaMesh] = field(default_factory=dict)

    etypes: list[str] = field(default_factory=list)
    fname: str = ""
    gnelems_etype: dict[str, int] = field(default_factory=dict)
    gnelems: int = field(default=0)

    def add_mmesh(self, name: str, mmesh: _MetaMesh, if_base: bool = False):

        # Ensure no other mmeshes exist before 1st mesh is added
        if if_base and self.mmeshes:
            raise ValueError(
                "Base mesh MUST be called once. The solution resides in here.")

        self.mmeshes[name] = mmesh
        self.mmeshes[name].mmesh_name = name

        if if_base:

            # Initialise logger
            self._logger = initialise_logger(__name__, LOG_LEVEL)

            comm, rank, root = get_comm_rank_root()

            self.etypes = mmesh.etypes
            self.fname = mmesh.fname
            self.gnelems_etype = {etype: comm.allreduce(len(mmesh.eidxs[etype]), 
                                                op=mpi.SUM) 
                                        for etype in mmesh.etypes}

            self.gnelems = sum(self.gnelems_etype.values())

            self.mmeshes[name].recreated = True
        else:

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

    def update_mmesh_comm(self, mesh_name: str, comm, ranks_map):

        self.mmeshes[mesh_name].comm = comm
        self.mmeshes[mesh_name].ranks_map = ranks_map

        for nrank, inters in list(self.mmeshes[mesh_name].con_p.items()):
            if not inters:
                del self.mmeshes[mesh_name].con_p[nrank]

    def connect_mmeshes(self, src_name: str, dest_name: str, 
                       both_ways: bool = False, *, overwrite = False):
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
        if dest_name in src_mmesh.interconnector and not overwrite:
            raise ValueError(f"Connection exists: {src_name} <-- {dest_name}")

        src_mmesh.interconnector[dest_name] = MeshInterConnector(
                                                        src_mmesh, dest_mmesh)
        if both_ways:
            self.connect_mmeshes(dest_name, src_name)

        # CRITICAL! COMMS OF BOTH MESHES ARE DIFFERENT! ENSURE CONSISTENT RANK NUMBERING!
        # create a mapping from base comm to this comm. This is reference for all
        
        # If does not exist, then set that location ad None
        # Example: 
        # Base always has fixed list of ranks [0, 1, 2, 3, 4, 5, 6, 7 .... ]
        # Compute may have a subset of ranks in some weird order [1, 0, None, 4]
        # This mapping will be {0: 1, 1: 0, 2: None, 3: 4}
        # WE have already done the above, and stored it in self.mmeshes[src_name].ranks_map
        # Now we need to connect src to dest, and store the same in MeshInterConnector object 

        smap = src_mmesh.ranks_map
        dmap = dest_mmesh.ranks_map

        # If dmap is an empty dict, then equate it to smap
        if not dmap:
            dmap = smap

        src_dest_ranks_map = {smap[k]: dmap[k] for k in smap}

        if None in src_dest_ranks_map.values():
            del src_dest_ranks_map[None]

        src_mmesh.interconnector[dest_name].ranks_map = src_dest_ranks_map

    def __str__(self):
        comm, rank, root = get_comm_rank_root()

        etypes = self.etypes
        nranks = comm.size
        columns = ['Mesh-name']

        # For each etype and each rank ...
        for etype in etypes:
            for r in range(nranks):
                columns.append(f'{etype}-{r}')

        for etype in etypes:
            columns.append(f'{etype}-all')

        columns.extend(['Total', 'Solution Here'])

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
                    etype_rank_counts[etype] = [0] * nranks

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
                    for r in range(nranks):
                        row.append(str(etype_rank_counts[etype][r]))

                for etype in etypes:               
                    row.append(str(nelem_etype[etype]))
                row.append(str(total_elements))    
                #row.append(inter_mesh_str)             # MESH CONNECTIONS   
                #row.append(str(mesh.recreated))    
                rows.append(row)                   

            # Now, format the table
            table = tabulate(rows, headers=columns, tablefmt='grid')

            return table
        else:
            return ''

class LoadRelocator():
    # Initialise as an empty list
    etypes = []

    def __init__(self, base_mesh: _Mesh, *,
                 bmmesh = 'base', cmmesh = 'compute', cnmmesh = 'compute_new'):

        self._logger = initialise_logger(__name__, LOG_LEVEL)

        self.mm = _MetaMeshes()
        self.mm.add_mmesh(bmmesh, _MetaMesh.from_mesh(base_mesh), if_base=True)

        # Print mmesh to check if it is added
        print(_MetaMesh.from_mesh(base_mesh).nelems, flush=True)

        self.mm.copy_mmesh(bmmesh, cmmesh)
        self.mm.copy_mmesh(cmmesh, cnmmesh)

        self.new_ranks = list(range(mpi.COMM_WORLD.size))

    def firstguess_target_nelems(self, weights: list[float]):
        '''
            Calculate the target elements per rank per rank weights.
            
            Args:
                mesh_nelems (list[int]): Elements per rank in mesh 
                weights (list[float]): Weights per rank.

            Returns:
                list[int]: Target elements per rank.
                
            Scenarios considered:
                1. Weights sum to 1.
                2. Target elements are integers that sum to total.

            Rule of thumb:
                1. Move any extra elements into rank with highest elements.
                - Assumption: Lowest wait-time rank with highest number of 
                              inter-rank interfaces load-balances the best.
        '''

        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights, dtype=float)

        t_nelems = np.round(self.mm.gnelems*weights).astype(int)

        # Move deficit into rank with highest elements
        t_nelems[np.argmax(t_nelems)] += self.mm.gnelems - sum(t_nelems)

        for t in t_nelems:
            if t == 0:
                raise ValueError("Rank removal functionality incomplete.")

        self.move_priority = sorted(range(len(t_nelems)), 
                                key=lambda k: t_nelems[k], reverse=True)

        # Convert to list of integers
        return t_nelems.tolist()

    @memoize
    def equipartition_diffuse(self, mesh_name):
        print("Expectation: Perform this once", flush=True)
        comm = self.mm.get_mmesh(mesh_name).comm

        self.move_priority = list(range(comm.size))

        t_nelems = [self.mm.gnelems/comm.size]*comm.size

        return self.diffuse_computation(mesh_name, t_nelems)

    def initialise_empty_rank(self, mesh_name, from_rank=0, to_rank=3):
        """
        Re-initialize the mesh distribution by moving elements from 'from_rank' to 'to_rank'.

        Parameters:
            mesh_name (str): The name of the mesh to modify.
            from_rank (int): The rank number to move elements from.
            to_rank (int): The rank number to move elements to.
            etypes (list[str], optional): etypes considered. Default to all.
        """
        comm, rank, root = get_comm_rank_root('compute')

        self.mm.copy_mmesh(mesh_name, f'{mesh_name}_reinitialised')
        mm_re = self.mm.get_mmesh(f'{mesh_name}_reinitialised')

        if rank == from_rank:
            etype, idx_loc = self.locate_elements_to_move(mm_re)
        else:
            etype, idx_loc = None, None

        if rank == from_rank:
            comm.send(etype, dest=to_rank, tag=BASE_TAG)
        elif rank == to_rank:
            etype = comm.recv(source=from_rank, tag=BASE_TAG)    
        
        if rank == from_rank:
            comm.send(mm_re.eidxs      [etype][idx_loc], dest=to_rank, tag=BASE_TAG)
            comm.send(mm_re.eles       [etype][idx_loc], dest=to_rank, tag=BASE_TAG)
            comm.send(mm_re.spts       [etype][idx_loc], dest=to_rank, tag=BASE_TAG)
            comm.send(mm_re.spts_curved[etype][idx_loc], dest=to_rank, tag=BASE_TAG)
            comm.send(mm_re.spts_nodes [etype][idx_loc], dest=to_rank, tag=BASE_TAG)

        elif rank == to_rank:
            mm_re.eidxs      [etype] = comm.recv(source=from_rank, tag=BASE_TAG)      
            mm_re.eles       [etype] = comm.recv(source=from_rank, tag=BASE_TAG)       
            mm_re.spts       [etype] = comm.recv(source=from_rank, tag=BASE_TAG)       
            mm_re.spts_curved[etype] = comm.recv(source=from_rank, tag=BASE_TAG)
            mm_re.spts_nodes [etype] = comm.recv(source=from_rank, tag=BASE_TAG) 

        if rank == from_rank:
            mm_re.eidxs      [etype] = np.delete(mm_re.eidxs      [etype], idx_loc)
            mm_re.eles       [etype] = np.delete(mm_re.eles       [etype], idx_loc)
            mm_re.spts       [etype] = np.delete(mm_re.spts       [etype], idx_loc)
            mm_re.spts_curved[etype] = np.delete(mm_re.spts_curved[etype], idx_loc)
            mm_re.spts_nodes [etype] = np.delete(mm_re.spts_nodes [etype], idx_loc)

        self.mm.remove_mmesh(mesh_name)
        self.mm.remove_mmesh(mesh_name+'_new')
        self.mm.copy_mmesh(mesh_name+'_reinitialised', mesh_name, mm_re.eidxs)
        self.mm.copy_mmesh(mesh_name, mesh_name+'_new')
        print(self.mm, flush=True)  

    def locate_elements_to_move(self, mm_re: _MetaMesh) -> dict[str, np.ndarray]:
        etype, idx = self.locate_mpi_interface_element(mm_re)
        return etype, np.array([int(np.where(mm_re.eidxs[etype] == idx)[0])])

    def locate_mpi_interface_element(self, mmesh: _MetaMesh) -> tuple[str, np.ndarray]:
        """
        Locate a seed element for the re-initialization of a rank.
        Element choises:
        1. intersection of given `from_rank` with longest two edges of con_p
        2. intersection of `from_rank` at some MPI interface
        """
        ordered_comm = sorted(mmesh.ncon_p_nrank, 
                                key=lambda x: mmesh.ncon_p_nrank[x], 
                                reverse=True)

        for i, nrank in enumerate(ordered_comm):
            for etype in mmesh.etypes:
                nrank_conp = mmesh.gcon_p[nrank][etype]

                for nnrank in ordered_comm[i+1:]:
                    nnrank_conp = mmesh.gcon_p[nnrank][etype]
                    idxs = np.intersect1d(nrank_conp, nnrank_conp)

                    if idxs:
                        return etype, idxs[0]

            # Just take the last element we find in nrank_conp
            if not idxs:
                return etype, nrank_conp[0]

    @log_method_times
    def diffuse_computation(self, mesh_name, t_nelems, cli = False):
        '''
            Iteratively diffuse elements until reaching target within each rank.
        '''

        comm, rank, root = get_comm_rank_root('compute')

        # Reset last iteration's leftover mesh.
        self.mm.move_mmesh(mesh_name+'_new', mesh_name)
        self.mm.copy_mmesh(mesh_name, mesh_name+'_new')

        curr_nelems = np.array(comm.allgather(self.mm.get_mmesh(mesh_name).nelems))
        current_nelems = np.array(comm.allgather(self.mm.get_mmesh(mesh_name+'_new').nelems))
        previous_nelems = np.zeros_like(current_nelems)

        # Create a matrix that lets us decide whether or not to move elements across interface between i and j ranks
        self.if_move_along_interface = np.ones((comm.size, comm.size))

        # Create a movement-multiplier for elements movement
        self.move_multiplier = np.ones((comm.size, comm.size))

        ccc = current_nelems

        # If ccc is not equal to previous_nelems, then continue
        while not np.array_equal(ccc, previous_nelems):

            previous_nelems = ccc
            ccc = comm.allgather(t_nelems[rank] - self.mm.get_mmesh(mesh_name+'_new').nelems)

            print("Current nelems: ", ccc, "Previous nelems: ", previous_nelems, flush=True)

            nelems_diff = t_nelems - np.array(comm.allgather(self.mm.get_mmesh(mesh_name+'_new').nelems))

            # If any rank has elements to move that is lesser than 1% of initial t_nelems, then set the rows and columns in self.if_move_along_interface to 0
            for i, diff in enumerate(nelems_diff):
                if abs(diff) < 0.01 * curr_nelems[i]:
                    self.if_move_along_interface[i, :] = 0
                    self.if_move_along_interface[:, i] = 0

            move_to_nrank = self.get_move_to_nrank(nelems_diff[rank], self.mm.get_mmesh(mesh_name+'_new'))

            # Find out how many elements can be moved from row-rank to column-rank (all positive)
            self.movable_to_nrank = self.get_movable_to_nrank(self.mm.get_mmesh(mesh_name+'_new'))

            # Multiply with if_move_along_interface 
            # to force-stop movement across some interfaces
            self.movable_to_nrank = np.multiply(self.movable_to_nrank, 
                                                self.if_move_along_interface)

            print(f"rank {rank}  \t "
                  f"Start: {self.mm.get_mmesh(mesh_name).nelems} "
                  f" \t Now: {self.mm.get_mmesh(mesh_name+'_new').nelems} --> \t ("
                  f" {t_nelems[rank] - self.mm.get_mmesh(mesh_name+'_new').nelems}"
                  f") --> \t {t_nelems[rank]}", flush=True)

            # Move elements
            move_elems = self.reloc_interface_elems(move_to_nrank, self.mm.get_mmesh(mesh_name+'_new'))
            self.mm.copy_mmesh(mesh_name+'_new', mesh_name+'-temp', 
                               self.get_preordered_eidxs(move_elems, self.mm.get_mmesh(mesh_name+'_new')))
            self.mm.move_mmesh(mesh_name+'-temp', mesh_name+'_new')

        self.mm.connect_mmeshes(mesh_name+'_new', mesh_name)

        if not cli:
            # CLI only needs eidxs, if-curved and if-mpi.
            self.mm.get_mmesh(mesh_name+'_new').spts = self.reloc(mesh_name, mesh_name+'_new', 
                                               self.mm.get_mmesh(mesh_name).spts, 
                                               edim=0)
            self.mm.get_mmesh(mesh_name+'_new').spts_curved = self.reloc(mesh_name, mesh_name+'_new', 
                                               self.mm.get_mmesh(mesh_name).spts_curved, 
                                               edim=0)

        # Re-create ary
        new_mesh = self.mm.get_mmesh(mesh_name+ '_new').to_mesh()
        
        print(self.mm, flush=True)

        self.new_ranks = [r for r in range(comm.size) if t_nelems[r] > 0]
        
        return new_mesh, 1

    def reloc(self, src_mesh_name : str, dest_mesh_name: str,
                 edict: dict[str, np.ndarray],*, edim) -> dict[str, np.ndarray]:

        dest_mmesh = self.mm.get_mmesh(dest_mesh_name)

        edict = dest_mmesh.preproc_edict(self.mm.etypes, edict, edim=edim)
        relocated_dict = dest_mmesh.interconnector[src_mesh_name].relocate(edict)
        return dest_mmesh.postproc_edict(relocated_dict, edim=edim)

    def get_move_to_nrank(self, nelems_diff: int, mesh: _MetaMesh):
        '''
            Save element movements as a matrix, 
            Elements move from row-rank to column-rank.
        '''

        comm, rank, root = get_comm_rank_root()

        move_to_nrank = np.zeros((comm.size, comm.size))
        for nrank, inters in mesh.ncon_p_nrank_etype.items():
            if nelems_diff != 0:
                if mesh.ncon_p > 0:
                    move_to_nrank[rank, nrank] = -nelems_diff * mesh.ncon_p_nrank[nrank] / mesh.ncon_p
                else:
                    move_to_nrank[rank, nrank] = 0
        move_to_nrank = comm.allreduce(move_to_nrank, op=mpi.SUM)
        move_to_nrank = np.maximum((move_to_nrank - move_to_nrank.T) / 2, 0)

        return move_to_nrank

    def get_movable_to_nrank(self, mesh):

        comm, rank, root = get_comm_rank_root()

        # Store available movable elements as a matrix 
        movable_to_nrank = np.zeros((comm.size, comm.size))

        for nrank in mesh.con_p.keys():
            for etype in self.mm.etypes:
                n_available = mesh.ncon_p_nrank_etype[nrank][etype]
                movable_to_nrank[rank, nrank] = n_available
                movable_to_nrank[nrank, rank] = n_available

        return movable_to_nrank

    def reloc_interface_elems(self, move_to_nrank: np.ndarray, mesh: _MetaMesh):

        comm, rank, root = get_comm_rank_root()

        # Break into loops
        move_elems = {nrank: {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes} for nrank in mesh.con_p.keys()}

        for nrank in mesh.con_p.keys():
            for etype in self.mm.etypes:


                #n_available = mesh.ncon_p_nrank_etype[nrank][etype]
                # Instead of taking from mesh object, take from the matrix
                n_available = self.movable_to_nrank[rank, nrank]
                
                if n_available == 0:
                    # No rank interfaces, element movement impossible.
                    move_elemsf = np.array([], dtype=np.int32)
                else:
                    # Number of elements to move for this etype with nrank
                    n_to_move = np.round(move_to_nrank[rank][nrank]).astype(int)
                    # n_to_move = np.round(move_to_nrank[rank][nrank] * n_available / mesh.ncon_p).astype(int)

                    # Enforce Nₑ-to-move ≤ Nₑ-available-to-move
                    n_to_move = min(n_to_move, n_available)

                    if n_to_move in [1, 2]:
                        # If only 1 or 2 elements needed to move
                        # Move all elements 
                        move_elemsf = mesh.gcon_p[nrank][etype]
                    elif n_to_move == n_available:
                        # If elements to move > interface elements with nrank
                        # Move all elements 
                        move_elemsf = mesh.gcon_p[nrank][etype]
                    else:
                        # Select any n_to_move elements from interface elements

                        n_to_move = np.floor(n_to_move* self.move_multiplier[rank, nrank]).astype(int)

                        move_elemsf = np.random.choice(mesh.gcon_p[nrank][etype], 
                                                       size=n_to_move,
                                                       replace=False)

                        # Set move multiplier to half for that movement interface
                        self.move_multiplier[rank, nrank] /= 2 # 1.1
                        self.move_multiplier[nrank, rank] /= 2 # 1.1

                    move_elemsf = [intelem for intelem, move in zip(mesh.gcon_p[nrank][etype], move_elemsf) if move]


                    move_elemsf = np.array(move_elemsf, dtype=np.int32)
                    move_elemsf = np.sort(move_elemsf)
                    move_elemsf = np.unique(move_elemsf)

                move_elems[nrank][etype] = move_elemsf
        
        # Empty if self
        move_elems[rank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
        move_elems_f = {nrank: move_elems[nrank] for nrank in sorted(move_elems.keys())}

        # Add empty arrays if we are not moving anything to nranks
        for nrank in range(comm.size):
            if nrank not in move_elems_f:
                move_elems_f[nrank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}

        rem_dups = self.remove_duplicate_movements(move_elems_f, mesh)
        
        # Add empty arrays if we are not moving anything to nranks
        for nrank in range(comm.size):
            if nrank not in rem_dups:
                rem_dups[nrank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
        
        return rem_dups

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

        # Step 8: Remove duplicates by iterating over sorted ranks
        for i, nrank in enumerate(self.move_priority):
            for etype in self.mm.etypes:
                for nnrank in self.move_priority[i+1:]:

                    if etype not in move_elems[nnrank]:
                        move_elems[nnrank][etype] = np.empty(0, dtype=np.int32)
                    else:
                        move_elems[nnrank][etype] = np.setdiff1d(
                            move_elems[nnrank][etype], 
                            move_elems[ nrank][etype]
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
                raise ValueError(
                    f"Total number of elements changed during relocation. "
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

    @staticmethod
    def count_all_eidxs(mesh_eidxs):
        comm, rank, root = get_comm_rank_root()
        total_elements = sum(len(e) for e in mesh_eidxs.values())
        return comm.allreduce(total_elements, op=mpi.SUM)

class MeshInterConnector(AlltoallMixin):

    def __init__(self, mmesh: _MetaMesh, target: _MetaMesh = None):

        # Get loggers
        self._logger = initialise_logger(__name__, LOG_LEVEL)

        self.comm = mmesh.comm

        self.mmesh = mmesh
        self.etypes = list(mmesh.eidxs.keys())
        self.neighbours = [i for i in range(self.comm.size) if i != self.comm.rank]

        self.mmesh_eidxs_gathered = mmesh.collectall('eidxs')

        if len(set(self.mmesh_eidxs_gathered.keys())) != len(self.etypes):
            raise ValueError(f"All ranks must have all etypes in mmesh.eidxs.")

        self._scount = {}
        self._sdisp = {}
        self._rcount = {}
        self._rdisp = {}
        self._sidxs = {}
        self._ridxs = {}

        if target is not None:
            self.target = target
            self.target_eidxs_gathered = target.collectall('eidxs')
            self.set_relocation_idxs()

    def copy(self):
        """
        Create a deep copy of the MeshInterConnector instance.
        Note: We avoid copying MPI communicators and other non-copyable attributes.
        """
        new_copy = MeshInterConnector.__new__(MeshInterConnector)
        new_copy.comm                 = self.comm
        new_copy.neighbours           = self.neighbours.copy()
        new_copy.etypes               = self.etypes.copy()
        new_copy.mmesh_eidxs_gathered = deepcopy(self.mmesh_eidxs_gathered)

        new_copy._scount = deepcopy(self._scount)
        new_copy._sdisp  = deepcopy(self. _sdisp)
        new_copy._rcount = deepcopy(self._rcount)
        new_copy._rdisp  = deepcopy(self. _rdisp)
        new_copy._sidxs  = deepcopy(self. _sidxs)
        new_copy._ridxs  = deepcopy(self. _ridxs)

        # Copy other attributes
        if hasattr(self, '_logger'):
            new_copy._logger = self._logger  # Share the same logger

        return new_copy

    @staticmethod
    def _send_recv0(etypes, send_elements: dict[int, dict[str, np.ndarray]]):

        comm, rank, root = get_comm_rank_root()
        recv_elements = {}

        mixin = AlltoallMixin()
        
        # Convert to 1D array for each etype
        for etype in etypes:
            _scount = np.array([len(send_elements[nrank][etype]) for nrank 
                                in range(comm.size)], dtype=np.int32)
            _sdisp = mixin._count_to_disp(_scount)
            _sidxs = np.concatenate([send_elements[nrank][etype] for nrank in send_elements]).astype(np.int32)

            # Communicate to get recieve counts and displacements
            _, (_rcount, _rdisp) = mixin._alltoallcv(comm, 
                                                     _sidxs, _scount, _sdisp)

            # Allocate recieve indices array
            _ridxs = np.empty((_rcount.sum(), *_sidxs.shape[1:]), 
                              dtype=_sidxs.dtype)

            mixin._alltoallv(comm, (_sidxs, (_scount, _sdisp)),
                                   (_ridxs, (_rcount, _rdisp)))

            recv_elements[etype] = _ridxs

        return recv_elements

    @log_method_times
    def recreate_reorder_recreate_mesh(self):
        self.target.interconnector[self.mmesh.mmesh_name].set_relocation_idxs()

        self._recreate_mesh_for_diffusion()                                     
        self._reconstruct_con_conp_bcon()                                       
        self._reorder_elements(self.target)                             
        self._invert_recreated_mesh()

        # WE ONLY RE-ORDER ELEMENTS HERE
        eidxs = self.target.collectall('eidxs')
        self.target_eidxs_gathered = eidxs
        self.target.interconnector[self.mmesh.mmesh_name].mmesh_eidxs_gathered = eidxs
        self.set_relocation_idxs()
        self.target.interconnector[self.mmesh.mmesh_name].set_relocation_idxs()

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
    def _reorder_elements(self, mesh: _Mesh):
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
    def set_relocation_idxs(self):
        """
        Create 1D arrays of send and receive indices for the given element type.
        """

        comm, rank, root = get_comm_rank_root()

        for etype in self.etypes:
            self._logger.info(f"Mapping to mesh eidxs for {etype}")

            from_mesh_eidxs = self.target_eidxs_gathered[etype][self.comm.rank]
            to_mesh_eidxs  = self. mmesh_eidxs_gathered[etype]

            # Build the send indices
            _sidxs = [[int(e_number) for e_number in from_mesh_eidxs
                    if e_number in to_mesh_eidxs[r]] 
                      for r in range(comm.size)
                     ]
            _scount = np.array([len(_sidxs[rank]) for rank in range(comm.size)])
            _sdisp = self._count_to_disp(_scount)
            _sidxs = np.concatenate(_sidxs).astype(int)

            # Communicate to get receive counts and displacements
            _, (_rcount, _rdisp) = self._alltoallcv(comm, 
                                                    _sidxs, _scount, _sdisp)

            # Allocate receive indices array
            _ridxs = np.empty((_rcount.sum(), *_sidxs.shape[1:]), 
                                dtype=_sidxs.dtype
                             )

            # Perform all-to-all communication to exchange indices
            self._alltoallv(comm, (_sidxs, (_scount, _sdisp)),
                                  (_ridxs, (_rcount, _rdisp)))

            self. _ridxs[etype] = _ridxs 

            self. _sidxs[etype] = _sidxs 
            self._scount[etype] = _scount
            self. _sdisp[etype] = _sdisp 
            self._rcount[etype] = _rcount
            self. _rdisp[etype] = _rdisp 

    def relocate(self, sary_dict: dict[str, np.ndarray]):

        rary = {}
        for etype, sary in sary_dict.items():
            base_idxs = self.target_eidxs_gathered[etype][self.comm.rank]
            end_idxs  = self. mmesh_eidxs_gathered[etype][self.comm.rank]
            send_idxs = self. _sidxs[etype]
            recv_idxs = self. _ridxs[etype]
            scount    = self._scount[etype]
            sdisp     = self. _sdisp[etype]
            rcount    = self._rcount[etype]
            rdisp     = self. _rdisp[etype]

            base_idxs_position = {idx: pos for pos, idx in enumerate(base_idxs)}

            try:
                base_to_send = [base_idxs_position[idx] for idx in send_idxs]
            except KeyError as e:
                raise KeyError(f"Missing index {e} in base_idxs_position, "
                               f"etype: '{etype}', rank {self.comm.rank}")

            svals = sary[base_to_send]

            rvals = np.empty((rcount.sum(), *svals.shape[1:]), dtype=svals.dtype)
            
            self._alltoallv(self.comm, (svals, (scount, sdisp)),
                                       (rvals, (rcount, rdisp)),
                            )

            recv_idxs_position = {idx: pos for pos, idx in enumerate(recv_idxs)}
            recv_to_end = [recv_idxs_position[idx] for idx in end_idxs]

            rary[etype] = rvals[recv_to_end]
            
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
        #glmap = [{}]*len(etypes)
        glmap = [{} for _ in etypes]
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
                                            dtype=bool) 
                                      for etype in self.etypes
                                    }
        
        for etype, eidxs in ifmpi.items():
            self.target.spts_internal[etype][eidxs.astype(int)] = False
