import atexit
import os
import sys

import numpy as np


comm = None
rank = None
root = None

def init_mpi():
    global comm, rank, root

    import mpi4py.rc
    from mpi4py import MPI

    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'

    # Manually initialise MPI
    MPI.Init()

    # Prevent mpi4py from calling MPI_Finalize
    mpi4py.rc.finalize = False

    comm = MPI.COMM_WORLD
    rank = comm.rank
    root = 0

    # Intercept any uncaught exceptions
    class ExceptHook:
        def __init__(self):
            self.exception = None

            self._orig_excepthook = sys.excepthook
            sys.excepthook = self._excepthook

        def _excepthook(self, exc_type, exc, *args):
            self.exception = exc
            self._orig_excepthook(exc_type, exc, *args)

    # Register our exception hook
    excepthook = ExceptHook()

    def onexit():
        if not MPI.Is_initialized() or MPI.Is_finalized():
            return

        # Get the current exception (if any)
        exc = excepthook.exception

        # If we are exiting normally then call MPI_Finalize
        if (comm.size == 1 or exc is None or
            isinstance(exc, (KeyboardInterrupt, SystemExit))):
            import gc
            gc.collect()

            MPI.Finalize()
        # Otherwise forcefully abort
        else:
            sys.stderr.flush()
            comm.Abort(1)

    # Register our exit handler
    atexit.register(onexit)


def get_initial_comm_rank_root():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm, comm.rank, 0

def get_comm_rank_root():
    global comm, rank, root

    return comm, rank, root


def get_local_rank():
    envs = [
        'MV2_COMM_WORLD_LOCAL_RANK',
        'OMPI_COMM_WORLD_LOCAL_RANK',
        'SLURM_LOCALID'
    ]

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        return comm.Split_type(MPI.COMM_TYPE_SHARED).rank


def scal_coll(colfn, v, *args, **kwargs):
    dtype = int if isinstance(v, (int, np.integer)) else float
    v = np.array([v], dtype=dtype)
    colfn(mpi.IN_PLACE, v, *args, **kwargs)
    return dtype(v[0])


def get_start_end_csize(comm, n):
    rank, size = comm.rank, comm.size

    # Determine how much data each rank is responsible for
    csize = max(-(-n // size), 1)

    # Determine which part of the dataset we should handle
    return min(rank*csize, n), min((rank + 1)*csize, n), csize


class AlltoallMixin:
    def _count_to_disp(self, count):
        return np.concatenate(([0], np.cumsum(count[:-1])))

    def _alltoallv(self, comm, sbuf, rbuf):
        svals = sbuf[0]

        # If we are dealing with scalar data then call Alltoallv directly
        if svals.dtype.names is None and svals.ndim == 1:
            comm.Alltoallv(sbuf, rbuf)
        # Else, we need to create a suitable derived datatype
        else:
            from mpi4py.util.dtlib import from_numpy_dtype

            dtype = svals.dtype

            if svals.ndim > 1:
                dtype = [('', dtype, svals.shape[1:])]

            dtype = from_numpy_dtype(dtype).Commit()

            try:
                comm.Alltoallv((*sbuf, dtype), (*rbuf, dtype))
            finally:
                dtype.Free()

    def _alltoallcv(self, comm, svals, scount, sdisps=None):
        # Exchange counts
        rcount = np.empty_like(scount)
        comm.Alltoall(scount, rcount)

        # Compute displacements
        rdisps = self._count_to_disp(rcount)
        sdisps = self._count_to_disp(scount) if sdisps is None else sdisps

        # Exchange values
        rvals = np.empty((rcount.sum(), *svals.shape[1:]), dtype=svals.dtype)
        rbuf = (rvals, (rcount, rdisps))
        self._alltoallv(comm, (svals, (scount, sdisps)), rbuf)

        return rbuf


class BaseGathererScatterer(AlltoallMixin):
    def __init__(self, comm, aidx):
        self.comm = comm

        # Determine array size
        n = aidx[-1] if len(aidx) else -1
        n = scal_coll(comm.Allreduce, n, op=mpi.MAX) + 1

        # Determine which part of the dataset we should handle
        self.start, self.end, csize = get_start_end_csize(comm, n)

        # Map each index to its associated rank
        adisps = np.searchsorted(aidx, csize*np.arange(comm.size))
        acount = np.diff(adisps, append=len(aidx))

        # Exchange the indices
        bidx, (bcount, bdisps) = self._alltoallcv(comm, aidx, acount, adisps)

        # Save the count and displacement information
        self.acountdisps = (acount, adisps)
        self.bcountdisps = (bcount, bdisps)

        # Return the index information
        return bidx


class Scatterer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

        # Eliminate duplicates from our index array
        ridx, self.rinv = np.unique(idx, return_inverse=True)

        self.sidx = super().__init__(comm, ridx) - self.start

        # Save the receive count
        self.cnt = len(ridx)

    def __call__(self, dset, didxs=(...,)):
        rcount, rdisps = self.acountdisps
        scount, sdisps = self.bcountdisps

        # Read the data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        # Unpack the data
        return rvals[self.rinv]


class Gatherer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

        # Sort our send array
        self.sinv = np.argsort(idx)
        self.sidx = idx[self.sinv]

        bidx = super().__init__(comm, self.sidx)

        # Determine how to sort the data we will receive
        self.rinv = np.argsort(bidx)
        self.ridx = bidx[self.rinv]

        # Note the source rank of each received element
        self.rsrc = np.repeat(np.arange(comm.size), self.bcountdisps[0])
        self.rsrc = self.rsrc[self.rinv].astype(np.int32)

        # Compute the total number of items and our offset
        self.cnt = cnt = len(self.ridx)
        self.tot = scal_coll(comm.Allreduce, cnt, op=mpi.SUM)
        self.off = scal_coll(comm.Exscan, cnt, op=mpi.SUM)
        self.off = self.off if comm.rank else 0

    def __call__(self, dset):
        scount, sdisps = self.acountdisps
        rcount, rdisps = self.bcountdisps

        # Sort the data we are going to be sending
        svals = np.ascontiguousarray(dset[self.sinv])

        # Allocate space for the data we will receive
        rvals = np.empty((self.cnt, *dset.shape[1:]), dtype=dset.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        # Sort our received data
        return rvals[self.rinv]


class SparseScatterer(AlltoallMixin):
    def __init__(self, comm, iset, aidx):
        self.comm = comm

        # Sort our indices
        ainv = np.argsort(aidx)
        bidx = aidx[ainv]

        # Determine the array size
        n = len(iset)

        # Determine which part of the dataset we should handle
        self.start, self.end, _ = get_start_end_csize(comm, n)

        # Read our portion of the sorted index table
        cidx = iset[self.start:self.end]

        # Tell other ranks what region we have
        region = np.array([cidx.min(initial=n), cidx.max(initial=n) + 1])
        minmax = np.empty(2*comm.size, dtype=int)
        comm.Allgather(region, minmax)

        # Determine which rank, if any, has each of our desired indices
        didx = np.array_split(bidx, np.searchsorted(bidx, minmax))[1::2]
        dcount = np.array([len(s) for s in didx])

        # Exchange indices
        eidx, (ecount, edisps) = self._alltoallcv(comm, np.concatenate(didx),
                                                  dcount)

        # See which of these indices are present
        mask = np.isin(eidx, cidx, assume_unique=True)
        sidx = eidx[mask]
        scount = np.array([m.sum() for m in np.array_split(mask, edisps[1:])])
        sdisps = self._count_to_disp(scount)

        # Make a note of which indices we have
        self.sidx = np.searchsorted(cidx, sidx)
        self.scountdisps = (scount, sdisps)

        # Exchange the present indices
        ridx, self.rcountdisps = self._alltoallcv(comm, sidx, scount,
                                                  sdisps)

        self.ridx = ainv[np.searchsorted(bidx, ridx)]
        self.cnt = self.rcountdisps[0].sum()

    def __call__(self, dset, didxs=(...,)):
        scount, sdisps = self.scountdisps
        rcount, rdisps = self.rcountdisps

        # Read and appropriately reorder our send data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, (scount, sdisps)),
                        (rvals, (rcount, rdisps)))

        return rvals


class _MPI:
    def __getattr__(self, attr):
        from mpi4py import MPI

        return getattr(MPI, attr)

    def update_comm(self, new_ranks: list[int]):
        from mpi4py import MPI
        global comm, rank, root

        # Decide which ranks to keep
        if rank not in new_ranks:
            color = MPI.UNDEFINED
        else:
            color = 0
 
        # Before splitting
        print(f"[Balance] Before split: Rank {rank}, Size {comm.Get_size()}", flush=True)
 
        comm = comm.Split(color, key=rank)
        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
            root = 0
        else:
            MPI.Finalize()
            sys.exit()

        # After splitting
        if comm != MPI.COMM_NULL:
            print(f"[Balance] After split: Rank {comm.rank}, Size {comm.size}", flush=True)
        else:
            print(f"[Balance] After split: Rank {rank}, Size N/A (Excluded)", flush=True)

        #return comm, [new_ranks.index(i) if i in new_ranks else None for i in range(MPI.COMM_WORLD.Get_size())] 
        # Base always has fixed list of ranks [0, 1, 2, 3, 4, 5, 6, 7 .... ]
        # Compute may have a subset of ranks in some weird order [1, 0, None, 2]
        # This mapping will be {0: 1, 1: 0, 2: None, 3: 2, 4: None, 5: None, 6: None, 7: None, ....}
        return comm, {b: new_ranks.index(b) if b in new_ranks else None for b in list(range(MPI.COMM_WORLD.Get_size()))}

mpi = _MPI()
