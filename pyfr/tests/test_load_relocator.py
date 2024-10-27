from pyfr.mpiutil import get_comm_rank_root 
import numpy as np
import pytest

from pyfr.loadrelocator import LoadRelocator

def test_count_all_eidxs():
    comm, rank, root = get_comm_rank_root()
    mesh_eidxs = {
        'etype1': np.array([1, 2, 3]) if rank == 0 else np.array([]),
        'etype2': np.array([4, 5]) if rank == 0 else np.array([])
    }
    total_elements = LoadRelocator.count_all_eidxs(mesh_eidxs)
    # The total_elements should be the sum across all ranks
    assert total_elements == 5, f"Total elements should be 5, got {total_elements}"
